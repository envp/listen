import numpy as np

from scipy.signal import hanning, savgol_filter
from listen.helpers import helpers

# minimum duration in milliseconds
MIN_SEGMENT_DURATION = 8

def segments(data, rate, min_duration=8, gamma=0.01):
    """
    Predicts speech segments from audio file
    :param xs: array_like
    :param rate: sample rate of audio file
    :param tol: tolerance in decibels
    :return: Segment boundaries for speech signal
    """
    wsize = (min_duration * rate) // 1000

    window = hanning(wsize)

    ste = np.zeros_like(data)
    es = np.zeros(2 * len(data))
    n = len(data)
    # zcr = np.zeros_like(data)

    xs = data / np.max(data)

    # Compute:
    # Short term energy of the signal over window
    for i in range(0, n - wsize, 1):
        ste[i] = np.linalg.norm(xs[i: i + wsize] * window, 2) / wsize

    # Compute:
    # Mean zero crossing rate of the signal over window
    # for i in range(n - wsize - 1):
    #     zcr[i] = np.sum(np.abs(np.sign(xs[i: i + wsize]) - np.sign(xs[i + 1: i + wsize + 1])))

    ste /= np.max(ste)

    # Mirror via lateral invesion about y-axis
    es[:n] = np.fliplr(ste)
    es[n:] = ste

    # Exponentiate
    np.power(es, gamma, out=es)


    # zcr /= np.max(zcr)

    # return ste, zcr, acr
