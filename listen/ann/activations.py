import numpy as np


class Activations():

    @staticmethod
    def eye(zs, deriv=False):
        # Just a debugging method to test the network without any non-linearities
        if deriv:
            return 1
        return zs
    @staticmethod
    def sigmoid(zs, deriv=False):
        if deriv:
            return zs * (1 - zs)
        return 1 / (1 + np.exp(-zs))

    @staticmethod
    def sign(zs, deriv=False):
        if deriv:
            return 1
        return np.sign(zs)

    @staticmethod
    def relu(zs, deriv=False):
        if deriv:
            return 1 * (zs > 0)
        # Slightly faster way to clip to 0 from below
        # Crux: X * True -> X and X * False -> 0 in python
        # since boolean is a subtype of int
        return zs * (zs > 0)

    @staticmethod
    def softmax(zs, deriv=False):
        """
        Expect to be given a one-hot vector

        """

        ex = np.exp(zs - np.max(zs))
        return ex / np.sum(ex)
