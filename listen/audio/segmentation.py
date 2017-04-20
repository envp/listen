import numpy as np

from scipy.signal import hamming

from listen.utils.helpers import chunk_it

MIN_SEGMENT_DURATION = 8

def segment_speech(data, mask, wsize, smooth=True, fuzz=0.3, timecodes=False):
    silences, segments = compute_segments_approx(data, mask, wsize, smooth=smooth, fuzz=fuzz)
    segments = merge_segments(segments, 3 * wsize)
    if timecodes:
        return segments
    else:
        samples = []
        for segment in segments:
            samples.append(data[segment[0] : segment[1]])
        return samples

def get_segment_mask(data, mask, wsize, smooth=False, fuzz=0.2, energy=None, zxr=None, lowconf=True):
    """
    Creates a segment mask using the data and initial mask (preferably an approximation of the actual speech regions)
    :param data: Audio data as ndarray
    :param mask: Mask as an ndarray with same dimensions as data
    :param wsize: window size to use to create a 
    :param smooth: Smooth data using some form of averaging
    :param fuzz: Fuzzy threshold multiplier 
    :param energy: 
    :param zxr: 
    :param lowconf: 
    :return: 
    """
    N = len(data)
    zcr = np.zeros_like(data)
    ste = np.zeros_like(data)
    decision = np.zeros_like(data)
    ds = np.zeros_like(data, dtype=np.float32)
    window = hamming(wsize)
    data = data * mask
    if smooth:
        data = np.convolve(data, 8 * np.ones(wsize // 8) / wsize, mode='same')

    xs = np.r_[[0] * (wsize // 2) , data, [0] * (wsize // 2)]
    if energy is None:
        for i in range(wsize //2, N - wsize):
            ste[i - wsize // 2] = np.linalg.norm(xs[i - wsize //2: i + wsize //2] * window, 2)
    else:
        ste = energy
    if zxr is None:
        for i in range(wsize // 2, N - wsize - 1):
            s = np.sign(xs[i + 1 - wsize //2: i + wsize // 2 + 1]) - np.sign(xs[i - wsize // 2: i + wsize // 2])
            np.clip(s, -1, 1, out=s)
            zcr[i - wsize //2] = np.max(np.abs(s))
    else:
        zcr = zxr

    thresh_ste = (fuzz * np.max(ste), (1 - fuzz) * np.max(ste))
    thresh_zcr = (fuzz * np.max(zcr), (1 - fuzz) * np.max(zcr))

    for i in range(1, N):
        if ste[i] > thresh_ste[0]:
            decision[i] = 1
        elif (ste[i] > thresh_ste[1] and ste[i] < thresh_ste[0]) and zcr[i] > thresh_zcr[1]:
            decision[i] = 1
        elif ste[i] < thresh_ste[0] and zcr[i] < thresh_zcr[0]:
            decision[i] = 0
        elif ste[i] < thresh_ste[0] and (zcr[i] < thresh_zcr[1] and zcr[i] > thresh_zcr[0]):
            if lowconf:
                decision[i] = 0.5
            else:
                decision[i] = 1
        elif ste[i] < thresh_ste[0] and zcr[i] > thresh_zcr[0]:
            decision[i] = 0

    for i in range(N - wsize//2):
        # A hack that really happens to work:
        # Replace L-2 norm based smoothing with infinity norm based smoothing.
        decision[i] = np.max(decision[i: i + wsize //2])

    return (decision, ste, zcr)

def compute_segments_approx(data, initial_mask, window_size, smooth=True, fuzz=0.3):
    xs = data / np.max(data)
    n = xs.shape[0]

    # Perhaps configure this as a nb_iters parameter? Or would it be too wasteful, since 3 repeated applications
    # produces convergent segment boundaries

    # First get a approximation of what the segments will be using a fuzzy (ternary) decision
    mask, ste, zcr = get_segment_mask(xs, np.ones(n), window_size, smooth=True)

    # Use this approximation as a high-confidence mask to improve upon initial segmentation
    mask, ste, zcr = get_segment_mask(xs, np.ones(n), window_size, smooth=True, energy=ste, zxr=zcr, lowconf=False)

    # Use a different threshold to get a new mask
    mask, ste, zcr = get_segment_mask(xs, mask, window_size, smooth=True, fuzz=0.5)

    zones = []
    for i in range(len(mask) - 1):
        if mask[i] != mask[i+1]:
            zones.append(i)
    silences = [
        (zones[i], zones[i + 1]) for i in range(len(zones) - 1) if mask[zones[i]] == 1 and mask[zones[i + 1]] == 0
    ]
    segments = [
        (zones[i], zones[i + 1]) for i in range(len(zones) - 1) if mask[zones[i]] == 0 and mask[zones[i + 1]] == 1
    ]

    return silences, segments

def merge_segments(segment_list, tolerance):
    """
    Given a list of segments, recursively merges adjecent segments if they aren't at least `tolerance` apart.
    Pre-condition: The segments must be sorted, this method doesn't check this condition
    :param segment_list: List of segment to merge (a list o pairs)
    :param tolerance: absolute minimum difference between two endpoints of a segment for them to be considered distinct
    :return: 
    """
    if len(segment_list) == 1:
        return segment_list

    segs = []

    for pair in chunk_it(segment_list, 2):
        if len(pair) < 2:
            if not pair[0] in segs:
                segs.append(pair[0])
        else:
            if pair[1][0] - pair[0][1] < tolerance:
                segs.append(tuple([pair[0][0],  pair[1][1]]))
            else:
                if not pair[0] in segs:
                    segs.append(pair[0])
                if not pair[1] in segs:
                    segs.append(pair[1])

    # i.e. This iteration was wasted.
    if len(segs) == len(segment_list):
        return segs
    else:
        return merge_segments(segs, tolerance)

"""
TODO: Try to get the group delay based approach to work
"""
# def segments(data, rate, min_duration=8, gamma=0.01, at=100, alpha=0.95):
#     """
#     Predicts speech segments from audio file
#     :param xs: array_like
#     :param rate: sample rate of audio file
#     :param tol: tolerance in decibels
#     :return: Segment boundaries for speech signal
#     """
#     wsize = (min_duration * rate) // 1000
#     window = chebwin(wsize, at=at)
#
#     level = 10 ** (at / -20)
#     xs = np.zeros_like(data)
#     n = len(data)
#     # Pre-emphasis, 1st order FIR highpass filter for alpha < 1
#     for i in range(1, len(data)):
#         xs[i] = data[i].astype(np.float32) - alpha * data[i - 1].astype(np.float32)
#
#     xs = xs / np.max(xs)
#     ste = np.zeros_like(xs)
#
#     # Zero pad excess
#     hw = wsize // 2
#     xs = np.append(xs, [0] * hw)
#
#     for i in range(hw, len(xs) - hw, hw // 4):
#         ste[i] = np.linalg.norm(xs[i - hw: i + hw] * window, 2) / wsize
#
#     mx = np.max(ste)
#     ste /= mx
#     # Lateral mirroring about Y-axis
#     es = np.r_[ste[-1:0:-1], ste[1:]]
#
#     # Clip to save ourselves from divide by zero
#     es[es < level] = level
#
#     es = es ** -gamma
#
#     fftpack.ifft(es, overwrite_x=True)
#     # Keep causal part of signal
#     es = es[n:]
#
#     phase = np.zeros_like(es)
#     es = np.append(es, [0] * hw)
#     for i in range(hw, len(es) - hw, 1):
#         phase[i] = np.angle(np.sum(fftpack.fft(es[i - hw: i + hw] * window)))
#
#     phase = -np.diff(phase)
#
#     # Remove jagged edges and smooth
#     for i in range(len(phase) - wsize):
#         phase[i] = max(phase[i:i + wsize])
#
#     phase = helpers.mean_smooth(phase, window=window)
#
#     peaks = phase
#     return phase
