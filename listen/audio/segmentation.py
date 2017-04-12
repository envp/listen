import numpy as np
from scipy import fftpack
from scipy.signal import chebwin

from listen.helpers import helpers

MIN_SEGMENT_DURATION = 8


def segments(data, rate, min_duration=8, gamma=0.01, at=100, alpha=0.95):
    """
    Predicts speech segments from audio file
    :param xs: array_like
    :param rate: sample rate of audio file
    :param tol: tolerance in decibels
    :return: Segment boundaries for speech signal
    """
    wsize = (min_duration * rate) // 1000
    window = chebwin(wsize, at=at)

    level = 10 ** (at / -20)
    xs = np.zeros_like(data)
    n = len(data)
    # Pre-emphasis, 1st order FIR highpass filter for alpha < 1
    for i in range(1, len(data)):
        xs[i] = data[i].astype(np.float32) - alpha * data[i - 1].astype(np.float32)

    xs = xs / np.max(xs)
    ste = np.zeros_like(xs)

    # Zero pad excess
    xs = np.append(xs, [0] * wsize)

    for i in range(0, len(xs) - wsize, 1):
        ste[i] = np.linalg.norm(xs[i: i + wsize] * window, 2) / wsize

    mx = np.max(ste)
    ste /= mx
    # Lateral mirroring about Y-axis
    es = np.r_[ste[-1:0:-1], ste[1:]]

    # Clip to save ourselves from divide by zero
    # es[es < level] = level

    es = es ** -gamma

    fftpack.ifft(es, overwrite_x=True)
    # Keep causal part of signal
    es = es[n:]

    phase = np.zeros_like(es)
    es = np.append(es, [0] * wsize)
    for i in range(0, len(es) - wsize, 1):
        phase[i] = np.angle(np.sum(fftpack.fft(es[i: i + wsize] * window)))

    phase = -np.diff(phase)

    # Remove jagged edges and smooth
    for i in range(len(phase) - wsize):
        phase[i] = max(phase[i:i + wsize])

    for i in range(len(phase) - wsize):
        phase[i] = max(phase[i:i + wsize])
    phase = helpers.mean_smooth(phase, window=window)

    peaks = phase
    return phase
