"""
Module wrapping array helper functions
"""

import numpy as np

class array_helpers(object):
    @staticmethod
    def standardize(ary):
        """Returns array input in standard form
        
        Standardizes the array of samples to number of 
        standard deviations from the mean for each sample
        """
        mu = np.mean(ary)
        ss = np.var(ary)
        if ss != 0:
            return (ary - mu) / ss
        else:
            return np.zeros_like(ary)

    @staticmethod
    def linscale(ary, left=0, right=1):
        """Returns a array of linearly rescaled samples to fit 
        in the given range

        Rescales an array linearly so that it falls in [left, right]
        
        The mapping used is the function: 
        f(ary) = left + (right - left)(ary - ary.min)/(ary.max - ary.min)
        """
        mx = np.max(ary)
        mn = np.min(ary)
        if mx != mn:
            return left + (right - left) * ((ary - mn) / (mx - mn))
        else:
            return ary

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    