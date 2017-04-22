import numpy as np
import glob
import os.path as path
import pickle
import re
from tqdm import tqdm
from IPython import  embed
from scipy.io import wavfile
from scipy.interpolate import interp1d

from listen.spectrogram.spectrogram import Spectrogram
from listen.utils.filters import Filter
from listen.utils.helpers import chunk_it

def generate_data():
    """
    Utility functions
    """

    def db_to_float(db):
        return 10 ** (db / 20)

    def sizeof_fmt(num, suffix='B'):
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "{:3.1f}{}{}".format(num, unit, suffix)
            num /= 1024.0
        return "{:.1f}{}{}".format(num, 'Yi', suffix)

    """
    Constants.
    """

    DATA_DIR = path.realpath(path.dirname(__file__) + '/../data/gen/proc')
    DATASET_PATH = path.realpath(path.dirname(__file__) + '/../data/gen/pickled/')
    DATASET_FILE = DATASET_PATH + 'data_batch_full.pkl'
    MAX_NB_COPIES = 12

    # 1% of the dataset is silence
    SILENCE_RELATIVE_COUNT = 0.01
    SILENCE_RANGE_DB = (-60, -40)

    # Speech duration in seconds for pitch-invariant scaling
    SPEECH_DURATION = 0.7


    FFT_SIZE = 2048
    STEP_SIZE = FFT_SIZE // 8
    LOG_THRESHOLD = 6
    CHUNK_SIZE = 1000
    NB_MFCC_BINS = 40
    FREQ_RANGE = (0, 8000)
    """
    Generating the dataset.
    """

    # Get all the wav files
    wavs = list(glob.iglob(path.join(DATA_DIR, '**/*.wav'), recursive=True))

    # Print total size being read
    print("Reading: {}...".format(sizeof_fmt(sum(path.getsize(w) for w in wavs))))

    utterances = list(set(map(lambda w: re.match(r'[a-z]+', path.basename(w)).group(0), wavs)))

    # +1 for silence
    nb_classes = len(utterances) + 1

    # Dump the labels
    data_file = open(path.realpath(path.join(DATASET_PATH, 'all_labels.pkl')), "wb")
    pickle.dump({'label_names': ['_'] + utterances}, data_file, pickle.HIGHEST_PROTOCOL)
    data_file.close()
    print('Wrote labels to file.')

    glob_max = -1
    cepstra = []
    utterance_idx = []
    nb_data = 0

    # Shuffle input files
    np.random.shuffle(wavs)

    spec = Spectrogram(FFT_SIZE, STEP_SIZE, LOG_THRESHOLD)

    for chunk_idx, chunk in tqdm(enumerate(chunk_it(wavs, CHUNK_SIZE))):
        print("Processing chunk {}, size={}".format(chunk_idx, sizeof_fmt(sum(path.getsize(c) for c in chunk))))

        for wav in tqdm(chunk):
            rate, snd = wavfile.read(wav, mmap=True)
            m = np.max(snd)
            glob_max = max(m, glob_max)
            # Split file name to extract utterance
            utterance = re.match(r'[a-z]+', path.basename(wav)).group(0)
            uidx = utterances.index(utterance)
            # Poor man's data variance
            noise_db = np.random.uniform(*SILENCE_RANGE_DB)
            noise_level = m * db_to_float(noise_db)
            nb_copies = np.random.randint(1, MAX_NB_COPIES)
            for i in range(nb_copies):
                noise = np.random.normal(0, noise_level, len(snd))
                factor = SPEECH_DURATION * rate / len(snd)
                sound = Filter.time_stretch(snd + noise, factor)
                sound = sound / np.max(sound)
                cep = spec.compute_mel_cepstrum(sound.astype('float32'), nb_mfcc_bins=NB_MFCC_BINS, frange=FREQ_RANGE)
                # Ensure square shape with piecewise linear interpolation
                if cep.shape[0] != cep.shape[1]:
                    interpolant = interp1d(np.linspace(0, 1, cep.shape[1]), cep, axis=1)
                    cep = interpolant(np.linspace(0, 1, cep.shape[0]))

                # Center wrt mean
                cep = cep - np.mean(cep)
                cepstra.append(cep.ravel())

                # Reserve index 0 for use later, also create 1-hot vectors on the fly
                uvec = np.zeros(nb_classes, dtype='int8')
                uvec[uidx + 1] = 1
                utterance_idx.append(uvec)

            """
            Add silence as data
            """
            # Manually add noise as silence now
            nb_silent = int(SILENCE_RELATIVE_COUNT * len(cepstra))
            for i in range(nb_silent):
                noise_db = np.random.uniform(*SILENCE_RANGE_DB)
                noise_level = (10 ** (noise_db / 20)) * glob_max
                noise = np.random.normal(0, noise_level, len(snd))
                cep = spec.compute_mel_cepstrum(noise.astype('float32'), nb_mfcc_bins=NB_MFCC_BINS, frange=FREQ_RANGE)
                if cep.shape[0] != cep.shape[1]:
                    interpolant = interp1d(np.linspace(0, 1, cep.shape[1]), cep, axis=1)
                    cep = interpolant(np.linspace(0, 1, cep.shape[0]))

                # Append a flattened array
                cepstra.append(cep.ravel())
                # Silence is index 0
                uvec = np.zeros(nb_classes, dtype='int8')
                uvec[0] = 1
                utterance_idx.append(uvec)


        nb_data = len(cepstra)
        # Shuffle again to avoid keeping copies together
        cepstra, utterance_idx = zip(*np.random.permutation(list(zip(cepstra, utterance_idx))))
        print("Working with {0} ({1} unique) sounds, including silence. )".format(nb_data, len(chunk)))

    nb_data = len(cepstra)
    # Split into training and validation sets (25% for validation, 75% for training)
    dump_data = {
        'validate_x': cepstra[:nb_data//4],
        'validate_y': utterance_idx[:nb_data//4],
        'train_x': cepstra[nb_data//4:],
        'train_y': utterance_idx[nb_data//4:]
    }


    # Dump collective dataset
    data_file = open(DATASET_FILE, "wb")
    pickle.dump(dump_data, data_file, pickle.HIGHEST_PROTOCOL)
    data_file.close()

    print('Wrote to dataset: {}'.format(DATASET_FILE))
    print("Done.")

if __name__ == '__main__':
    generate_data()
