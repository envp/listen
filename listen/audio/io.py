from scipy.io import wavfile

def read(fname, mmap=True, backend='scipy'):
    """
    Reads a wav format file and returns a numpy array of samples and the sampling rate
    :param fname: Name of the wav
    :param mmap: Boolean to indicate if we need to use a memory map for reading the file 
    :return: 
    """
    if backend == 'scipy':
        return wavfile.read(fname)
    else:
        raise ValueError("Unsupported backend: '{}'".format(backend))
