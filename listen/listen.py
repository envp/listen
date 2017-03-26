# %%
%load_ext autoreload
%autoreload 2

#%%

# Packages we're using
import numpy as np
import matplotlib.pyplot as plt



# Project level imports
from listen.data.an4.an4 import AN4
from listen.spectrogram import spectrogram as spg
from listen.helpers.helpers import helpers

# %%

# Use only the first file for demo purposes
an4data = AN4(debug=False, conversion=False)
# filename = next(an4data.trainset.data)

# # Read the wavfile from the filename
# rate, data = wavfile.read(filename)

# data = Filter.butter_bandpass_filter(data, F_LO, F_HI, rate, order=1)


# # Most of the spectrogram has been taken from :
# # https://github.com/jameslyons/python_speech_features
# # https://timsainb.github.io/spectrograms-mfccs-and-inversion-in-python.html
# wav_spectrogram = spg.Spectrogram(FFT_SIZE, STEP_SIZE, LOGSCALE, SPEC_THRESH)

# spectrum = wav_spectrogram.compute_spectrum(data)
# cepstrum = wav_spectrogram.compute_mel_cepstrum(
#     data, NUM_MFCC_COMPONENTS, (F_START, F_END), compression=COMPRESSION)

# fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(8, 8))

# ax[0].matshow(spectrum.T, interpolation='nearest',
#               aspect='auto', cmap=plt.cm.afmhot, origin='lower')
# ax[0].set_title('Original Spectrum')

# ax[1].matshow(cepstrum, interpolation='nearest', aspect='auto',
#               cmap=plt.cm.afmhot, origin='lower')
# ax[1].set_title('Mel Cepstrum')

# plt.tight_layout()

helpers.save_data(data=an4data.trainset.data, )
