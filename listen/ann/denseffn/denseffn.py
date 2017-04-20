import pprint
import random

import numpy as np

from listen.ann.activations import Activations


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
        return (x - self.a) ** 2 + (y - self.b) ** 2 - self.r ** 2 < 0

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
            self.weights, self.error, self.delta, self.output * (1 - self.output), self.output
        )

    def __repr__(self):
        return str(self)


class Layer(object):
    def __init__(self, indim, outdim, isinput=False, isout=False, activation='sigmoid'):
        self.indim = indim
        self.outdim = outdim
        self.isout = isout
        self.isinput = isinput
        self.activation = activation
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

    def train(self, inputs, targets, validations, validation_targets, epochs=1, rate=1, bsize=None, threshold=1e-3):
        n = len(inputs)
        acc = 0
        for epoch in range(epochs):
            # Feedforward loop
            for i, row in enumerate(inputs):
                self.predict(row)
                self.backpropagate(targets[i])
                self.update(rate, row)


            # 0-1 loss
            for i, row in enumerate(validations):
                predicted = self.predict(row)
                predicted = np.array(predicted)
                predicted = 1 * (predicted >= 0.5)
                acc += sum( 1 * (predicted == validation_targets[i]))

            acc /= len(validation_targets)

            if (epoch + 1) % 50 == 0:
                print("== epoch={:04d}\taccuracy={}".format(
                    epoch + 1, acc
                ))

        return {'accuracy': acc}

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
                neuron.delta = neuron.error * layer.activation(neuron.output, deriv=True)

    def update(self, rate, inputs):
        for i, layer in enumerate(self.layers):
            if not layer.isinput:
                inputs = np.array([neuron.output for neuron in self.layers[i - 1].neurons])

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


def main():
    #
    # HYPER PARAMATERS
    #
    DATASET_SIZE = 500
    EPOCHS = 500
    RATE = 0.1  
    THRESHOLD = 1e-3
    IN_FLAG = 1
    OUT_FLAG = 0
    random.seed(1)

    # Repeatable psuedo-random results
    data = CircleDataSet(0.5, 0.6, 0.4)

    training, train_targets = list(zip(*data.take(DATASET_SIZE)))
    n = len(training)
    validation = training[:n // 10]
    validation_targets = train_targets[:n // 10]
    training = training[n // 10:]
    train_targets = train_targets[n // 10:]
    testing, test_targets = list(zip(*data.take(DATASET_SIZE)))
    network = DenseFFN(Activations.sigmoid, 2, 10, 1)

    print("== Training DATASET_SIZE={} (IN={}, OUT={}), EPOCHS={}, RATE={}, ACTIVATION='{}'".format(
        DATASET_SIZE, train_targets.count(IN_FLAG), train_targets.count(OUT_FLAG), EPOCHS, RATE, Activations.sigmoid
    ))
    result = network.train(
        training, train_targets, validation, validation_targets, threshold=THRESHOLD, epochs=EPOCHS, rate=RATE
    )

    print("\tTraining results={}".format(result))

    print("== Testing network on {} points with {} inside and {} outside".format(
        DATASET_SIZE, train_targets.count(IN_FLAG), train_targets.count(OUT_FLAG)
    ))

    ps = network.test(testing, test_targets)
    accs = [abs(t - ps[i][0]) < 0.5 for i, t in enumerate(test_targets)]

    print("\tTesting accuracy={}\n".format(accs.count(True) / len(test_targets)))
    pprint.pprint(network.layers[0].neurons)


if __name__ == '__main__':
    main()
