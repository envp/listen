# %%
%load_ext autoreload
%autoreload 2

#%%

# Packages we're using
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import copy
from scipy.io import wavfile
from scipy.signal import butter, lfilter

# Project level imports
from listen.data.an4.an4 import AN4
from listen.spectrogram import spectrogram as spg
from listen.helpers.filters import Filter
from listen.helpers.array_helpers import array_helpers as ahelp
# %%
### Parameters ###
FFT_SIZE = 2048  # window size for the FFT
STEP_SIZE = FFT_SIZE / 16  # distance to slide along the window (in time)
SPEC_THRESH = 4  # threshold for spectrograms (lower filters out more noise)
F_LO = 500  # Hz # Low cut for our butter bandpass filter
F_HI = 8000  # Hz # High cut for our butter bandpass filter
# For mels
# number of mel frequency channels
NUM_MFCC_COMPONENTS = 64
COMPRESSION = 10  # how much should we compress the x-axis (time)
F_START = 300  # Hz # What frequency to start sampling our melS from
F_END = 8000  # Hz # What frequency to stop sampling our melS from

# %%
# Use only the first file for demo purposes
an4data = AN4(debug=False, conversion=False)
filename = next(an4data.trainset.data)

# Read the wavfile from the filename
rate, data = wavfile.read(filename)

data = Filter.butter_bandpass_filter(data, F_LO, F_HI, rate, order=1)


# Most of the spectrogram has been taken from https://github.com/jameslyons/python_speech_features
wav_spectrogram = spg.Spectrogram(FFT_SIZE, STEP_SIZE, True, SPEC_THRESH)

cepstrum = wav_spectrogram.compute_mel_cepstrum(data, NUM_MFCC_COMPONENTS, (F_START, F_END), compression=COMPRESSION)
