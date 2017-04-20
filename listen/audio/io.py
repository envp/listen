from scipy.io import wavfile
import subprocess
from os import path

def read(fname, mmap=True, backend='scipy', rate=None):
    """
    Reads a wav format file and returns a numpy array of samples and the sampling rate
    :param fname: Name of the wav
    :param mmap: Boolean to indicate if we need to use a memory map for reading the file 
    :return: 
    """
    if backend == 'scipy':
        sr, data = wavfile.read(fname, mmap=mmap)

        # Asked to re-sample data!
        if not rate is None:
            # Binaries ftw
            # TODO: Replace this
            outfile = path.realpath(path.dirname(__file__) + '/out.wav')
            # fname = path.realpath(fname)
            process = subprocess.Popen('ffmpeg.exe -i {} -y -ar {} {}'.format(fname, rate, outfile), shell=True)
            process.wait()
            sr, data = wavfile.read(outfile, mmap=mmap)

    else:
        raise ValueError("Unsupported backend: '{}'".format(backend))

    return sr, data
