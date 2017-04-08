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
    pool = Pool()

    if path.exists(dpath):
        x = list(
            glob.iglob(path.join(dpath, '**/*.wav'), recursive=True)
        )
        y = list(
            glob.iglob(path.join(dpath, '**/*.trans.txt'), recursive=True)
        )
        num_files = len(x)
        xs = list(pool.imap(get_mfcc_from_path, x, 8))
        ys = {}

        for _, u in enumerate(pool.imap(get_transcript_from_path, y, 8)):
            ys = {**ys, **u}
        pool.close()
        return xs, ys

    else:
        print("Couldn't find dataset: {}".format(dpath))


def load_data(dev=True, train=False, test=False):
    if dev:
        xs, ys = load_from_dir(DEV_DIR)

    if test:
        xs, ys = load_from_dir(TEST_DIR)

    if train:
        xs, ys = load_from_dir(TRAIN_DIR)

    return xs, ys

