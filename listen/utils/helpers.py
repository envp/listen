import collections

import numpy as np


def flatten(l):
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, (str, bytes)):
            yield from flatten(el)
        else:
            yield el


def chunk_it(xs, chunk_size):
    for i in range(0, len(xs), chunk_size):
        yield xs[i:i + chunk_size]


def mean_smooth(xs, window):
    wsize = len(window)
    hw = wsize // 2
    pxs = np.append([0] * hw, xs)
    pxs = np.append(xs, [0] * hw)
    ds = np.zeros_like(xs)
    for i in range(hw, len(ds) - hw, 1):
        ds[i] = np.sum(pxs[i - hw: i + hw] * window)
    return ds
