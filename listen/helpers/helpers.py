import numpy as np

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
