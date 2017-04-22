import os
import sys
import pickle
import gzip
import numpy as np

from librosa.effects import time_stretch
from scipy.interpolate import interp1d

from listen.audio import io
from listen.audio import segmentation
from listen.utils import generate_dataset
# from listen.utils.filters import Filter
from listen.ann.denseffn import denseffn
from listen.ann.activations import Activations
from listen.spectrogram.spectrogram import Spectrogram


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
    Arg # 2: Path to the WAVfile in PCM format to the script
             as a command line if last argument was 'test'

    Example:
    python {} test /path/to/pcm_encoded_wavfile.wav
""".format(sys.argv[0])

DATA_PKL_GZ = os.path.realpath('./listen/data/gen/pickled/ml.pkl.gz')
LABELS_FILE = os.path.realpath('./listen/data/gen/pickled/all_labels.pkl')
TRAINED_DUMP = os.path.realpath('./listen/data/gen/pickled/nnet.pkl')

EPOCHS = 1
RATE = 1
ACT_FUNC = Activations.relu


def main():
    if len(sys.argv) < 3 and not (sys.argv[1] == 'train' or sys.argv[1] == 'datagen'):
        print(USAGE_STRING)
    else:
        mode = sys.argv[1]
        if mode == 'test':
            fname = sys.argv[2]
            # Minimum window duration
            MIN_DURATION = 20
            spec = Spectrogram(2048, 128, 6)

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
            cepstra = []
            for segment in speech_segments:
                ts_ratio = len(segment) / (0.7 * rate)
                segment = segment / np.max(segment)
                segment = time_stretch(segment, ts_ratio)
                cep = spec.compute_mel_cepstrum(segment.astype('float32'), 54, (0, 8000))
                if cep.shape[0] != cep.shape[1]:
                    interpolant = interp1d(np.linspace(0, 1, cep.shape[1]), cep, axis=1)
                    cep = interpolant(np.linspace(0, 1, cep.shape[0]))
                cepstra.append(cep.ravel())

            # Feed cepstra into network sequentially:
            network = pickle.load(open(TRAINED_DUMP, "rb"))
            labels = pickle.load(open(LABELS_FILE, 'rb'))
            predictions = [network.predict(cep) for cep in cepstra]

            # Get index of 1 from 1-hot vectors to
            predictions = [np.argmax(p) for p in predictions]

            # Convert to words
            predictions = [labels['label_names'][p] for p in predictions]

            print("\n\nNetwork's estimate:\n\n", predictions)

            # Call the classifier
        elif mode == 'train':
            # denseffn.main()
            data_file = gzip.open(DATA_PKL_GZ)
            print('== Loading datasets...')
            data = pickle.load(data_file)
            train_x = data['train_x']
            train_y = data['train_y']
            validate_x = data['validate_x']
            validate_y = data['validate_y']
            print('== Done loading.')

            # Input dimension (2916 for current dataset)
            idim = train_x[0].shape[0]

            # Output dimension (1011)
            odim = train_y[0].shape[0]

            # Hidden layer dimensions
            hdims = (1000, 500)

            network = denseffn.DenseFFN(ACT_FUNC, idim, *hdims, odim)

            print(
                "Training network with validation for params: EPOCHS={}, RATE={}, ACTIVATION={}".format(
                    EPOCHS, RATE, ACT_FUNC
                )
            )
            result = network.train(
                train_x, train_y, validate_x, validate_y, epochs=EPOCHS, rate=RATE
            )

            print("\tTraining results={}".format(result))

            # Dump training results that we can use for classification later
            nnet_file = open(TRAINED_DUMP, 'wb')
            pickle.dump(network, nnet_file, protocol=pickle.HIGHEST_PROTOCOL)
            nnet_file.close()

        elif mode == 'datagen':
            generate_dataset.generate_data()

        else:
            print(USAGE_STRING)

if __name__ == '__main__':
    main()
