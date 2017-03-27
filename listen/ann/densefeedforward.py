import itertools
import random

import numpy as np

from enum import Enum


class Neuron(object):
    def __init__(self, indim):
        self.weights = np.array([
            random.random.uniform(-5e-3, 5e-3) for __ in range(1 + indim)
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


class Layer(object):
    def __init__(self, indim, outdim, isinput=False, isout=False, activation='sigmoid'):
        self.indim = indim
        self.outdim = outdim
        self.isout = isout
        self.isinput = isinput
        self.activation = getattr(self, activation.value)
        self.derivative = getattr(self, activation.value + '_derivative')
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


class DenseFeedForwardNetwork(object):
    layers = []

    def __init__(self, activ, *ldims):
        assert len(ldims) > 1
        indim = 1
        self.shape = ldims
        for i, ldim in enumerate(ldims):
            if i == len(ldims) - 1:
                break
            self.layers.append(Layer(ldim, ldims[i + 1], activation=activ))

        # Special treatment
        self.layers[-1].isout = True
        self.layers[0].isinput = True

    def __str__(self):
        return "DenseFFN({})".format('->'.join(map(str, self.layers)))

    def train(self, inputs, targets, epochs=1, rate=1, bsize=None, threshold=1e-3):
        cost = None
        n = len(inputs)
        acc = None
        for epoch in range(epochs):
            # Feedforward loop
            outputs = []
            cost = 0
            acc = 0
            if bsize:
                inputs = list(inputs)
                random.shuffle(inputs)
                inputs = inputs[:int(n * bsize)]
            for i, row in enumerate(inputs):
                outputs.append(self.predict(row))
                cost += 0.5 * (targets[i] - outputs[i][0])**2
                self.backpropagate(targets[i])
                self.update(rate, row)
                # print("== input={} - target={} - predicted={}".format(row, targets[i], outputs))

            cost /= n
            predicted = [1 if output[0] > 0.5 else 0 for output in outputs]
            acc = [p == t for p, t in zip(predicted, targets)].count(True) / n

            if (epoch + 1) % 50 == 0:
                print("== epoch={:04d}\terror={:.9f}\taccuracy={}".format(
                    epoch + 1, cost, acc
                ))

            if cost < threshold:
                break
        return {'accuracy': acc, 'error': cost}

    def predict(self, xs):
        pred = list(xs)
        for layer in self.layers:
            pred = layer.feedforward(pred)
            layer.output = pred
            # print("==layer==")
        return pred

    def backpropagate(self, expected):
        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            if not layer.isout:
                for j, neuron in enumerate(layer.neurons):
                    neuron.error = 0
                    for out_neuron in self.layers[i + 1].neurons:
                        neuron.error += out_neuron.weights[j] * \
                            out_neuron.delta
            else:
                for j, neuron in enumerate(layer.neurons):
                    neuron.error = expected - neuron.output
            for j, neuron in enumerate(layer.neurons):
                neuron.delta = neuron.error * layer.derivative(neuron.output)

    def update(self, rate, inputs):
        for i, layer in enumerate(self.layers):
            if not layer.isinput:
                inputs = np.array(
                    [neuron.output for neuron in self.layers[i - 1].neurons])

            inputs = np.insert(inputs, 0, 1.0)
            for neuron in layer.neurons:
                neuron.weights = neuron.weights + rate * neuron.delta * inputs

    def neuron(self, i, j):
        return self.layers[i].neurons[j]

    def test(self, inputs, targets):
        predictions = []
        for xs in inputs:
            predictions.append(self.predict(xs))
        return predictions


class Activations(Enum):
    SIGN = 'sign'
    SIGMOID = 'sigmoid'
