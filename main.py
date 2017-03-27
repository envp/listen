#%%

import pprint
import random

from listen.ann.activations import Activations
# Project level imports
from listen.ann.denseffn.network import DenseFFN


class CircleDataSet(object):
    IN_VAL = 1
    OUT_VAL = 0

    def __init__(self, a, b, r, sp=0.5):
        self.a = a
        self.b = b
        self.r = r
        self.sp = sp
        self.xgen = random.Random(a * r * 1337)
        self.ygen = random.Random(b * r * 31337)

    def is_inside(self, x, y):
        return (x - self.a)**2 + (y - self.b)**2 - self.r**2 < 0

    def generate(self):
        while True:
            x = self.xgen.random()
            y = self.ygen.random()
            yield ((x, y), self.IN_VAL if self.is_inside(x, y) else self.OUT_VAL)

    def take(self, n):
        i = int(n * self.sp)
        o = n - i
        data = self.take_inside(i) + self.take_outside(o)
        random.shuffle(data)
        return data

    def take_inside(self, n):
        gen = self.generate()
        pts = []
        while len(pts) < n:
            p = next(gen)
            if self.IN_VAL == p[1]:
                pts.append(p)
        return pts

    def take_outside(self, n):
        gen = self.generate()
        pts = []
        while len(pts) < n:
            p = next(gen)
            if self.OUT_VAL == p[1]:
                pts.append(p)
        return pts


#
# HYPER PARAMATERS
#
DATASET_SIZE = 100
EPOCHS = 1000
RATE = 0.1
BATCH_SIZE = None
THRESHOLD = 1e-3
IN_FLAG = 1
OUT_FLAG = 0
ACT = Activations.SIGMOID
random.seed(1)


def main():
    # Repeatable psuedo-random results

    data = CircleDataSet(0.5, 0.6, 0.4)

    training, train_targets = list(zip(*data.take(DATASET_SIZE)))
    testing, test_targets = list(zip(*data.take(DATASET_SIZE)))
    network = DenseFFN(ACT, 2, 10, 1)

    print("== Training DATASET_SIZE={} (IN={}, OUT={}), EPOCHS={}, RATE={}, ACTIVATION='{}'".format(
        DATASET_SIZE, train_targets.count(
            IN_FLAG), train_targets.count(OUT_FLAG), EPOCHS, RATE, ACT
    ))
    result = network.train(
        training, train_targets, bsize=BATCH_SIZE, threshold=THRESHOLD, epochs=EPOCHS, rate=RATE
    )

    print("\tTraining results={}".format(result))

    print("== Testing network on {} points with {} inside and {} outside".format(
        DATASET_SIZE, train_targets.count(
            IN_FLAG), train_targets.count(OUT_FLAG)
    ))

    ps = network.test(testing, test_targets)
    accs = [abs(t - ps[i][0]) < 0.5 for i, t in enumerate(test_targets)]

    print("\tTesting accuracy={}\n".format(
        accs.count(True) / len(test_targets)))
    pprint.pprint(network.layers[0].neurons)


if __name__ == '__main__':
    main()
