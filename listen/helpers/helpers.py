import matplotlib.pyplot as plt
import numpy as np
import os

from scipy.io import wavfile
from listen.helpers.filters import Filter
from listen.spectrogram import spectrogram as spg


class helpers(object):
    @staticmethod
    def hz2mel(hz):
        """Convert a value in Hertz to Mels
        :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
        """
        return 2595 * np.log10(1 + hz / 700.0)

    @staticmethod
    def mel2hz(mel):
        """Convert a value in Mels to Hertz
        :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
        :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
        """
        return 700 * (10 ** (mel / 2595.0) - 1)

    @staticmethod
    def save_data(dataset=None,
                  fftsize=2048,
                  low=20, high=8000,
                  logscale=True,
                  spec_thresh=3,
                  n_mfcc=64,
                  start=20,
                  end=8000,
                  compression=2):
        """Saves figures and numpy arrays containing spectra to local directories
        """
        step = fftsize / 16

        for filepath in dataset:
            rate, data = wavfile.read(filepath)
            data = Filter.butter_bandpass_filter(
                data, low, high, rate, order=1)
            # Most of the spectrogram has been taken from :
            # https://github.com/jameslyons/python_speech_features
            # https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
            wav_spectrogram = spg.Spectrogram(
                fftsize, step, logscale, spec_thresh)

            spectrum = wav_spectrogram.compute_spectrum(data)
            cepstrum = wav_spectrogram.compute_mel_cepstrum(
                data, n_mfcc, (start, end), compression=compression)

            fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

            ax[0].imshow(spectrum.T, interpolation='nearest',
                         aspect='auto', cmap=plt.cm.afmhot, origin='lower')
            ax[0].set_title('Original Spectrum')

            ax[1].imshow(cepstrum, interpolation='nearest',
                         aspect='auto', cmap=plt.cm.afmhot, origin='lower')
            ax[1].set_title('Mel Cepstrum')
            plt.tight_layout()
            fig.savefig(filepath.replace('.wav', '.png'), dpi=150)
            plt.close(fig)
            np.save(filepath.replace('.wav', '.cepstrum'), cepstrum)
            print("Saved cepstra and figure for {}".format(
                os.path.basename(filepath)))
