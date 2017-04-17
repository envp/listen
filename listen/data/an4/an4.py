# -*- coding: utf-8 -*-

import itertools
import logging
import os
import re
import subprocess

from listen.utils import helpers

__author__ = "Vaibhav Yenamandra <vyenaman@ufl.edu>"

logger = logging.getLogger()
logging.basicConfig(format='[%(levelname)s] %(message)s')


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
            yield k, v

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

    def __init__(self, datapath, truthpath, convert):
        def to_res_path(r): return os.path.realpath(
            os.path.join(
                os.path.dirname(__file__),
                'wav', os.path.dirname(r),
                os.path.basename(r).strip() + '.raw'
            )
        )

        self.data = map(to_res_path, [line for line in open(datapath, 'r')])
        self.truth = self.parse_data(truthpath)
        if convert:
            self.convert_to_wav()
        self.data = list(map(lambda s: s.replace('.raw', '.wav'), self.data))

    @staticmethod
    def parse_data(fpath):
        truth = dict()
        with open(fpath, 'r') as f:
            for line in f:
                res = re.search(
                    r"(<s>)*\s*(?P<truth>[\w\s]+)\s*(</s>)*\s*\((?P<fileid>.+)\)", line
                )
                truth[res.group('fileid') + '.wav'] = res.group('truth').lstrip().rstrip().split(' ')
        return truth

    def __iter__(self):
        for d in self.data:
            yield d, self.truth[os.path.basename(d)]

    def __getitem__(self, key):
        return self.data[key], self.truth[os.path.basename(self.data[key])]

    def male(self):
        return itertools.filterfalse(
            lambda s: not os.path.basename(
                os.path.dirname(s)).startswith('m'), self.data
        )

    def female(self):
        return itertools.filterfalse(
            lambda s: not os.path.basename(
                os.path.dirname(s)).startswith('f'), self.data
        )

    def file_ids(self):
        return map(lambda s: os.path.basename(s).replace('.wav', ''), self.data)

    def convert_to_wav(self):
        count = 0
        for f in self.data:
            outname = os.path.join(os.path.dirname(
                f), os.path.basename(f).replace('.raw', '.wav'))
            subprocess.run(
                'ffmpeg -y -f s16le -ar 16k -ac 1 -i {} {}'.format(f, outname))
            os.remove(f)
            count += 1
            logger.debug("Converted {} files, deleted corresponding raw".format(count))


class AN4(object):
    """Wrapper for AN4 audio data and associated metadata.

    Ideally this should download and unpack the dataset instead of assuming
    that it already exists.
    """

    def path_to_resource(resname):
        return os.path.realpath(
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

    def __init__(self, debug=False, conversion=False, phones=False):
        if debug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
        self.dictions = DictionMap(self.DICTIONARY_FILE)
        self.phonemes = self.phoneme_loads(open(self.PHONEME_LIST, 'r'))
        self.trainset = AN4Data(self.TRAIN_DATA, self.TRAIN_TRUTH, conversion)
        self.testset = AN4Data(self.TEST_DATA, self.TEST_TRUTH, conversion)
        [
            logger.debug("Loaded resource {}".format(r)) for r in [
            self.DICTIONARY_FILE,
            self.PHONEME_LIST,
            self.TRAIN_DATA,
            self.TRAIN_TRUTH,
            self.TEST_DATA,
            self.TEST_TRUTH
        ]
        ]
        if phones:
            for k, v in self.trainset.truth.items():
                self.trainset.truth[k] = list(map(
                    lambda u: self.phonemes.index(u),
                    helpers.flatten(
                       ['SIL'] + [self.dictions.diction[u] for u in v] + ["SIL"]
                    )
                ))

    @staticmethod
    def phoneme_loads(s):
        return [line.strip("\n") for line in s]
