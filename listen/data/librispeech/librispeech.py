# -*- coding: utf-8 -*-

import glob
import logging
import sys
import numpy as np
from multiprocessing import Pool
from os import path

from scipy.io import wavfile

from listen.spectrogram.spectrogram import Spectrogram

__author__ = "Vaibhav Yenamandra <vyenaman@ufl.edu>"

logger = logging.getLogger()
logging.basicConfig(format='[%(levelname)s] %(message)s')

DEV_DIR = path.realpath(path.join(path.dirname(__file__), 'LibriSpeech/dev-clean'))
TRAIN_DIR = path.realpath(path.join(path.dirname(__file__), 'LibriSpeech/train-clean-100'))
TEST_DIR = path.realpath(path.join(path.dirname(__file__), 'LibriSpeech/test-clean'))
VOCAB_FILE = path.realpath(path.join(path.dirname(__file__), 'librispeech-vocab.txt'))

def get_mfcc_from_path(fname):
    rate, data = wavfile.read(fname, mmap=True)
    return (data, rate, path.basename(fname).replace('.wav', ''))

def get_transcript_from_path(fname):
    transcript = {}
    with open(fname) as f:
        for l in f:
            t = l.strip('\n').split(' ')
            transcript[t[0].replace('.wav', '')] = t[1:]
    return transcript


def load_from_dir(dpath):
    """
    Load transcripts and filenames containing audio
    Returns a dictionary like: {filename:string --> transcript_text:string}
    :param dpath: Path of the directory containing the dataset
    :return: A dictionary of filenames mapped to transcription text, 
    along with a list of filenames
    """
    pool = Pool()

    if path.exists(dpath):
        wavs = list(
            glob.iglob(path.join(dpath, '**/*.wav'), recursive=True)
        )
        ts = list(
            glob.iglob(path.join(dpath, '**/*.trans.txt'), recursive=True)
        )
        val = {}
        for transcript in ts:
            with open(transcript, 'r') as tfile:
                for line in tfile:
                    ary = line.strip('\n').split(' ')
                    val[ary[0]] = ary[1:]
        return val, wavs
    else:
        raise ValueError("Couldn't find dataset: {}".format(dpath))


def load_data(dev=True, train=False, test=False):
    """
    Loads a standard dataset (dev / train / test) from librispeech
    :param dev: Load dev set
    :param train: Load train set
    :param test: Load test set
    :return: Transcription Dict of utterances and list of filepaths
    """
    ds = None
    ls = None
    if dev:
        ds, ls = load_from_dir(DEV_DIR)

    if test:
        ds, ls = load_from_dir(TEST_DIR)

    if train:
        ds, ls = load_from_dir(TRAIN_DIR)

    return ds, ls

