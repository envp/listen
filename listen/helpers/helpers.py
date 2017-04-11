import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

import concurrent.futures
import multiprocessing
from listen.helpers.filters import Filter
from listen.spectrogram import spectrogram as spg

import collections

def __save_info_private(dataset,
                        fftsize=2048,
                        low=20, high=8000,
                        logscale=True,
                        spec_thresh=3,
                        n_mfcc=64,
                        start=20, end=8000,
                        compression=2):
    for filepath, truth in dataset:
        try:
            step = fftsize / 16
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

            fig = plt.figure()
            st = fig.suptitle("Utterance = {}".format('-'.join(truth)))
            fig.figsize = (10, 8)

            ax0 = fig.add_subplot(211)
            ax0.imshow(spectrum.T, interpolation='nearest',
                       aspect='auto', cmap=plt.cm.afmhot, origin='lower')
            ax0.set_title('Original Spectrum')

            ax1 = fig.add_subplot(212)
            ax1.imshow(cepstrum, interpolation='nearest',
                       aspect='auto', cmap=plt.cm.afmhot, origin='lower')
            ax1.set_title('Mel Cepstrum')

            fig.tight_layout()

            st.set_y(0.95)
            fig.subplots_adjust(top=0.85)

            fig.savefig(filepath.replace('.wav', '.png'), dpi=150)
            plt.close(fig)
            np.save(filepath.replace('.wav', '.cepstrum'), cepstrum)
            print("Saved cepstra and figure for {}".format(
                os.path.basename(filepath)))
        except Exception as e:
            print(e)

def save_data(dataset=None, **kwargs):
    """Saves figures and numpy arrays containing spectra to local directories
    """
    def grouper(iterable, n):
        args = [iter(iterable)] * n
        return itertools.zip_longest(*args)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=multiprocessing.cpu_count())
    futures = [
        executor.submit(__save_info_private, group, **kwargs)
        for group in grouper(dataset, 4)
    ]
    concurrent.futures.wait(futures)

def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el

def chunks(data, c, step=1):
    """
    Yields a list in chunks of size `s`
    :param data: List to get chunks from
    :param c: Chunk size
    :param step: Iteration step size
    :return: Generator for chunks
    """
    for i in range(0, len(data) - c, step):
        yield data[i: i + c]

def mean_smooth(xs, window):
    return np.convolve(xs, window, mode='same')
