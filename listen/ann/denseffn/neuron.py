import numpy as np
import random


class Neuron(object):
    def __init__(self, indim):
        self.weights = np.array([
            np.random.uniform(-5e-3, 5e-3) for __ in range(1 + indim)
        ])
        self.error = float("nan")
        self.delta = float("nan")
        self.output = float("nan")

    def compute(self, xs):
        # print(xs, '.', self.weights)
        return np.inner(self.weights, xs)

    def __str__(self):
        return "Neuron(wts={}, error={}, delta={}, derivative={}, output={})".format(
            self.weights, self.error, self.delta, self.output *
            (1 - self.output), self.output
        )

    def __repr__(self):
        return str(self)
