import os
import sys
import pickle
import glob
import numpy as np

from listen.audio import io
from listen.audio import segmentation

from listen.ann.denseffn import denseffn
from listen.ann.activations import Activations
from listen.utils.generate_dataset import generate_data

USAGE_STRING = """
    +==============================+
    |   A Speech to text program   |
    +==============================+
    \   ^__^
     \  (oo)\_______
        (__)\       )\/
            ||----W |
            ||     ||
    
    Usage:
    ======
    
    Arg # 1: One of 'train' or 'test'
    Arg # 2: Path to the WAVfile in PCM format to the script as a command line 
    
    Example:
    python {} test /path/to/pcm_encoded_wavfile.wav
""".format(sys.argv[0])

DATA_DIR = os.path.realpath('./listen/data/gen/pickled/')

def main():
    if len(sys.argv) < 3 and not sys.argv[1] == 'train':
        print(USAGE_STRING)
    else:
        mode = sys.argv[1]
        if mode == 'test':
            fname = sys.argv[2]
            # Minimum window duration
            MIN_DURATION = 20

            # TODO: Load the classifier weights

            # Use scipy's wavfile.read(...) to process our input
            # and re-sample it to 22050Hz (same as training examples)
            rate, data = io.read(fname, mmap=True, backend='scipy', rate=22050)

            # Ensure that window size is an even number
            wsize = (rate * MIN_DURATION) // 1000
            if wsize % 2 == 1:
                wsize += 1

            # Compute the speech segments to feed into classifier
            # This returns the speech samples to feed into the classifier by default
            speech_segments = segmentation.segment_speech(data, np.ones(len(data)), wsize)

            # Call the classifier
        elif mode == 'train':
            # batch = pickle.load(open(next(glob.iglob(os.path.join(DATA_DIR, '*.pkl'))), 'rb'))
            denseffn.main()

        else:
            print(USAGE_STRING)

if __name__ == '__main__':
    main()
