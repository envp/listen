import itertools
import random

import numpy as np

from .layer import Layer


class DenseFFN(object):
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
                np.random.shuffle(inputs)
                inputs = inputs[:int(n * bsize)]
            for i, row in enumerate(inputs):
                outputs.append(self.predict(row))
                cost += 0.5 * (targets[i] - outputs[i][0]) ** 2
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
                        neuron.error += out_neuron.weights[j] * out_neuron.delta
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

    def test(self, inputs):
        predictions = []
        for xs in inputs:
            predictions.append(self.predict(xs))
        return predictions
