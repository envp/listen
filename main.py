# %%
# Project level imports
import numpy as np

from listen.data.an4.an4 import AN4
from listen.ann.activations import Activations
from listen.ann.denseffn.network import DenseFFN

from tqdm import tqdm


def main():
    # Set conversion = true to convert the raw files to wav
    # Set phones=True to numerize phones (including silence)
    an4data = AN4(conversion=False, phones=True)

    # Uncomment this to run the MFCC computations
    # and save spectral data to disk
    # helper.save_data(an4data.trainset)
    cepstra = []
    ts = []
    print('Loading speech MFCC data...')
    for fpath, t in tqdm(list(an4data.trainset)):
        cepstra.append(np.load(fpath.replace('.wav', '.cepstrum.npy')))
        ts.append(t)

    nb_mfcc_bins = cepstra[0].shape[0]

    # Including silence
    nb_phones = len(an4data.phonemes)

    """
    See: Vijayaditya Peddinti, Daniel Povey, Sanjeev Khudanpur
    "A time delay neural network architecture for efficient modeling of long
    temporal contexts"
    """


if __name__ == '__main__':
    main()
