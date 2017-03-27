import numpy as np

from .neuron import Neuron
from ..activations import Activations

class Layer(object):
    def __init__(self, indim, outdim, isinput=False, isout=False, activation=Activations.SIGMOID):
        self.indim = indim
        self.outdim = outdim
        self.isout = isout
        self.isinput = isinput
        self.activation = getattr(self, activation.func_name())
        self.derivative = getattr(self, activation.derivfunc_name())
        self.output = np.array([])
        # Transpose of the weight matrix
        self.neurons = [Neuron(indim) for _ in range(outdim)]

    def __str__(self):
        return "Layer({})".format(self.neurons)

    def __repr__(self):
        return str(self)

    def feedforward(self, xs):
        # assert(len(xs) == self.indim), "input={}, expected={}".format(len(xs), self.indim)
        # Add a bias input:
        xs = np.insert(xs, 0, 1.0)
        for neuron in self.neurons:
            # print("activation={}, input={}, wts={}".format(a, neuron.weights, xs))
            # print(self, xs)
            neuron.output = self.activation(neuron.compute(xs))
        return [neuron.output for neuron in self.neurons]

    def sign(self, zs):
        if hasattr(zs, '__iter__'):
            zs = np.array(list(zs))
        return 0.5 * (np.sign(zs) + 1)

    def sign_derivative(self, zs):
        if hasattr(zs, '__iter__'):
            zs = np.array(list(zs))
        return np.ones_like(zs)

    def sigmoid(self, zs):
        if hasattr(zs, '__iter__'):
            zs = np.array(list(zs))
        return 1 / (1 + np.exp(-zs))

    def sigmoid_derivative(self, zs):
        return zs * (1 - zs)

    def tanh(self, zs):
        if hasattr(zs, '__iter__'):
            zs = np.array(list(zs))
        return np.tanh(zs)
