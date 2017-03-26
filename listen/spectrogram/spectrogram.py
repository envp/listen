import numpy as np
import scipy.ndimage
import copy

from listen.helpers.array_helpers import array_helpers as ahelp
from listen.helpers.filters import Filter


class Spectrogram(object):
    def __init__(self, fft_size, step_size, logscale, thresh):
        self.fftsize = int(fft_size)
        self.step = int(step_size)
        self.logscale = logscale
        self.thresh = thresh

    def overlap(self, X):
        """
        Create an overlapped version of X
        Parameters
        ----------
        X : ndarray, shape=(n_samples,)
            Input signal to window and overlap

        Returns
        -------
        X_strided : shape=(n_windows, window_size)
            2D array of overlapped X
        """
        if self.fftsize % 2 != 0:
            raise ValueError("Window size must be even!")
        # Make sure there are an even number of windows before stridetricks
        append = np.zeros((self.fftsize - len(X) % self.fftsize))
        X = np.hstack((X, append))

        ws = self.fftsize
        ss = self.step
        a = X

        valid = len(a) - ws
        nw = int(valid // ss)
        out = np.ndarray((nw, ws), dtype=a.dtype)

        for i in range(nw):
            # "slide" the window along the samples
            start = i * ss
            stop = start + ws
            out[i] = a[start: stop]

        return out

    def stft(self, X, mean_normalize=True, real=False, compute_onesided=True):
        """Computes STFT for 1D real valued input X
        """
        if real:
            local_fft = np.fft.rfft
            cut = -1
        else:
            local_fft = np.fft.fft
            cut = None
        if compute_onesided:
            cut = self.fftsize // 2
        if mean_normalize:
            X -= X.mean()

        X = self.overlap(X)

        size = self.fftsize
        win = 0.54 - .46 * np.cos(2 * np.pi * np.arange(size) / (size - 1))
        X = X * win[None]
        X = local_fft(X)[:, :cut]
        return X

    def compute_spectrum(self, data):
        """Creates a spectrogram using data passed
        """
        specgram = np.abs(self.stft(data, real=False, compute_onesided=True))

        if self.logscale:
            specgram = ahelp.linscale(specgram, left=1e-6, right=1)
            specgram = np.log10(specgram)
            specgram[specgram < -self.thresh] = -self.thresh
        else:
            specgram[specgram < self.thresh] = self.thresh

        return specgram

    def compute_mel_cepstrum(self, data, num_mfcc_comps, frange, compression=1):
        specgram = self.compute_spectrum(data)
        mel_filter, _ = Filter.create_mel_filter(
            self.fftsize, num_mfcc_comps, *frange)

        mel_cepstrum = specgram.dot(mel_filter).T
        if compression != 1:
            mel_cepstrum = scipy.ndimage.zoom(mel_cepstrum.astype(
                'float32'), [1, 1. / compression]).astype('float32')

        mel_cepstrum = mel_cepstrum[:, 1:-1]
        return mel_cepstrum
