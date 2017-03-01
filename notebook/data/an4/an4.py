# -*- coding: utf-8 -*-

import os
import re

__author__ = "Vaibhav Yenamandra <vyenaman@ufl.edu>"


class DictionMap(object):
    """Wrapper for diction / pronunciation data

    This class wraps the pronunciation dictionary for an4.
    The precondition for this class is that the data has been downloaded and
    unpacked with this file
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.diction = self.parse_data(filepath)

    def __iter__(self):
        for k, v in self.diction.items():
            yield (k, v)

    def parse_data(self, fpath):
        phones = {}
        with open(fpath, 'r') as f:
            for line in f:
                word, *ph = line.split()
                # Find the first index of a parantheses (indicating duplication)
                # and slice after that.
                lparen_index = word.find('(')
                if lparen_index > 0:
                    word = word[:lparen_index]

                if word not in phones:
                    phones[word] = list()
                phones[word].append(ph)
        return phones


class AN4Data(object):
    """Wraps the actual audio data in AN4
    """

    def __init__(self, datapath, truthpath):
        self.data = [os.path.realpath(line + '.raw') for line in open(datapath, 'r')]
        self.truth = self.parse_data(truthpath)

    def parse_data(self, fpath):
        truth = dict()
        with open(fpath, 'r') as f:
            for line in f:
                res = re.search(
                    r"(<s>)*\s*(?P<truth>[\w\s]+)\s*(</s>)*\s*\((?P<fileid>.+)\)", line
                )
                truth[res.group('fileid')] = res.group('truth').split(' ')
        return truth


class AN4(object):
    """Wrapper for AN4 audio data and associated metadata.

    Ideally this should download and unpack the dataset instead of assuming
    that it already exists.
    """

    path_to_resource = lambda resname: os.path.realpath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), resname))

    DATASET_URL = 'http://www.speech.cs.cmu.edu/databases/an4/an4_raw.bigendian.tar.gz'

    # The dictionary file maps each word to it's possible pronunciations
    DICTIONARY_FILE = path_to_resource('etc/an4.dic')
    PHONEME_LIST = path_to_resource('etc/an4.phone')
    # 2-gram language model provided with the dataset
    # currently we aren't bothered about an4.ug.lm.DMP, an4.filler and an4.ug.lm since
    # we have no clue what these are for
    LANG_MODEL = path_to_resource('etc/an4.ug.lm')
    TRAIN_DATA = path_to_resource('etc/an4_train.fileids')
    TRAIN_TRUTH = path_to_resource('etc/an4_train.transcription')
    TEST_DATA = path_to_resource('etc/an4_test.fileids')
    TEST_TRUTH = path_to_resource('etc/an4_test.transcription')

    # Assert that resources exist
    # @TODO: This could be handled better
    assert os.path.exists(
        DICTIONARY_FILE), "No pronunciation dictionary at {}".format(DICTIONARY_FILE)

    assert os.path.exists(
        PHONEME_LIST), "No phoneme list found at {}".format(PHONEME_LIST)

    assert os.path.exists(
        TRAIN_DATA), "No training data found at {}".format(TRAIN_DATA)

    assert os.path.exists(
        TRAIN_TRUTH), "No training ground truth found at {}".format(TRAIN_TRUTH)

    assert os.path.exists(
        TEST_DATA), "No testing data found at {}".format(TEST_DATA)

    assert os.path.exists(
        TEST_TRUTH), "No testing ground truth found at {}".format(TEST_TRUTH)

    def __init__(self):
        self.dictions = DictionMap(self.DICTIONARY_FILE)
        self.phonemes = self.phoneme_loads(open(self.PHONEME_LIST, 'r'))
        self.train = AN4Data(self.TRAIN_DATA, self.TRAIN_TRUTH)
        self.test = AN4Data(self.TEST_DATA, self.TEST_TRUTH)

    def phoneme_loads(self, s):
        return [line for line in s]
