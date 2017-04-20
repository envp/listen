import sys
import numpy as np

from listen.audio import io
from listen.audio import segmentation

def main():
    if len(sys.argv) < 2:
        print("""
        +==============================+
        |   A Speech to text program   |
        +==============================+
        
        Usage:
        ======
        
        Pass the path to the WAVfile in PCM format to the script as a command line parameter.
        
        Example:
        python {} /path/to/pcm_encoded_wavfile.wav
        """.format(sys.argv[0]))
    else:
        fname = sys.argv[1]
        # Minimum window duration
        MIN_DURATION = 20

        # Load the classifier

        # Use scipy's wavfile.read(...) to process our input
        # and re-sample it to 22050Hz (same as training examples)
        rate, data = io.read(fname, mmap=True, backend='scipy', rate=22050)

        wsize = (rate * MIN_DURATION) // 1000
        # Ensure that window size is an even number
        if wsize % 2 == 1:
            wsize += 1

        # Compute the speech segments to feed into classifier
        # This returns the speech samples to feed into the classifier by default
        speech_segments = segmentation.segment_speech(data, np.ones(len(data)), wsize)

        # Call the classifier

if __name__ == '__main__':
    main()
