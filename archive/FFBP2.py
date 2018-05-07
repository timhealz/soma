"""
Implementation of a Feed Forward Back Propagation (FFBP) Multilayer Perceptron Neural Network, consisting of
Perceptron and Network objects. Weights, biases, activation values, errors, and deltas are calculated and stored
within Perceptrons, while the Network object knits together layers of Perceptron objects based on a user-defined
structure. Feed forward and back propagation functions calculate activity values, compare network outputs to
desired outputs, and back propagate error deltas to update weights and biases. The network is iterated upon using
a trainer to minimize error.

Author: Tim Healy
"""

import numpy as np
import random as rand
from archive import activation_functions as af


class Perceptron():
    def __init__(self, num_inputs):
        """
        Each Perceptron within the network stores and calculates relevant information like weights, biases,
        activation values, errors, and deltas.

        Future release to have more flexibility around weight and bias value setting. These should be randomly
        generated if no weights and biases are given. Allowing user-inputted values for my class assignments.
        """
        self.weights = np.array([rand.uniform(0, 1) for x in range(num_inputs)])
        self.bias = rand.uniform(0, 1)

    def function(self, x, activation_function):
        """
        Chooses activation function based on user input.
        """
        if activation_function == 'ramp':
            y = af.ramp(x)
        else:
            y = af.sigmoid(x)
        return (y)

    def calc_output_activation(self, inputs):
        A = np.dot(inputs, self.weights) + self.bias
        self.activation_value = self.function(A, 'ramp')
        return (self.activation_value)

    def calc_activation(self, inputs):
        A = np.dot(inputs, self.weights) + self.bias
        self.activation_value = self.function(A, 'sigmoid')
        return (self.activation_value)

    def set_output_delta(self, inputs, desired_output, eta):
        """
        Since output layers and hidden layers have different calculations for their delta values, two different
        delta functions are defined. Weights and biases in the output layer are then updated using the appropriate
        delta value.

        Output delta = Error * (1 - Activation Value) * Activation value
        """
        inputs = np.array(inputs)
        self.error = desired_output - self.calc_output_activation(inputs)
        self.total_error = (1 / 2) * self.error ** 2
        self.delta = self.error * af.sigmoid(np.dot(self.weights, inputs) + self.bias)
        # (1 - self.activation_value) *
        self.updated_weights = self.weights + eta * self.delta * inputs
        self.bias = self.bias + eta * self.delta * 1
        output = str(self.total_error) + ", " + str(self.activation_value)
        print("     " + output)

    def set_hidden_delta(self, inputs, prev_deltas, prev_weights, eta):
        """
        Since output layers and hidden layers have different calculations for their delta values, two different
        delta functions are defined. Weights and biases in the output layer are then updated using the appropriate
        delta value.

        Hidden delta = (1 - Activation Value) * Activation value * SUM(previous delta * previous weight)
        """
        inputs = np.array(inputs)
        self.delta = (1 - self.activation_value) \
                     * self.activation_value * \
                     np.dot(prev_deltas, prev_weights)
        self.updated_weights = self.weights + eta * self.delta * inputs
        self.bias = self.bias + eta * self.delta * 1
        output = str(self.updated_weights) + ", " + str(self.bias)
        # print("     " + output)

class Network():
    def __init__(self, structure, num_inputs):
        """
        The Network object creates structures of Perceptron objects and feeds information stored within the perceptrons
        to other perceptrons. The "structure" input guides a lot of logic in the functions, since information is being
        fed from layer to layer.
        """
        self.structure = structure
        self.num_inputs = num_inputs
        self.network = []
        for i, j in enumerate(self.structure):
            if i == 0:
                layer = [Perceptron(num_inputs) for x in range(j)]
            else:
                layer = [Perceptron(self.structure[i - 1]) for x in range(j)]
            self.network.append(layer)

        self.output_vector = []

    def feed_forward(self, inputs):
        """
        The "FF" of FFBP. Starting with the first layer in the network, this function calculates activation values for
        each perceptron in the layer, and then "feeds" the newly calculated activations values forward into the next layer
        of perceptrons until it reaches the end of the network.
        """
        # print("Feeding Forward...")

        k = 0
        self.activations = []
        while k < len(self.structure):
            # print("     Layer " + str(k+1) + " Activations")
            if k == 0:
                self.activations.append([x.calc_activation(inputs) for x in self.network[k]])
            if k == len(self.structure)-1:
                self.activations.append([x.calc_output_activation(self.activations[k - 1]) for x in self.network[k]])
            else:
                self.activations.append([x.calc_activation(self.activations[k - 1]) for x in self.network[k]])
            # print("     " + str(self.activations[k]))
            k += 1

    def back_propagate(self, inputs, desired_output, eta):
        """
        The "BP" of FFBP. Starting with the last (output) layer of the network, this function compares the output(s) to the
        desired output(s), and calculates the error and delta values. These delta values are fed back into the previous
        layer, updating the weights and biases, until it reaches the beginning of the network.
        """
        # print("Back Propagating...")
        k = len(self.structure) - 1
        while k >= 0:
            if k == len(self.structure) - 1:
                print("     Total Error, Output Layer Activation Value")
                prev_outputs = self.activations[k - 1]
                [o.set_output_delta(prev_outputs, desired_output, eta) for o in self.network[k]]

            elif k == 0:
                # print("     Updated Layer " + str(k+1) + " Weights & Biases")
                prev_deltas = [o.delta for o in self.network[k + 1]]
                perceptrons = self.structure[k]
                for i in range(perceptrons):
                    prev_weights = [o.weights[i] for o in self.network[k + 1]]
                    self.network[k][i].set_hidden_delta(inputs, prev_weights, prev_deltas, eta)

            else:
                # print("     Updated Layer " + str(k+1) + " Weights & Biases")
                prev_deltas = [o.delta for o in self.network[k + 1]]
                prev_outputs = [o.activation_value for o in self.network[k - 1]]
                perceptrons = self.structure[k]
                for i in range(perceptrons):
                    prev_weights = [o.weights[i] for o in self.network[k + 1]]
                    [self.network[k][i].set_hidden_delta(prev_outputs, prev_weights, prev_deltas, eta)]

            k = k - 1

        for i, j in enumerate(x.structure):
            for k in range(j):
                self.network[i][k].weights = self.network[i][k].updated_weights

    def trainer(self, inputs, desired_output, eta):
        """
        Included trainer function that allows the user to train the network.
        """
        self.feed_forward(inputs)
        self.back_propagate(inputs, desired_output, eta)
        self.feed_forward(inputs)


x = Network([10, 10, 1], 2)
data = [[[1.98, 10], 0],
        [[1.80, 10], 1],
        [[1.05, 160], 2],
        [[1.45, 180], 1],
        [[1.80, 80], 1],
        [[1.96, 110], 1],
        [[0.4, 40], 2],
        [[2.05, 130], 1],
        [[0.90, 10], 1],
        [[2.5, 60], 0],
        [[1.6, 105], 2],
        [[1.05, 196], 1],
        [[0.52, 105], 2],
        [[1.80, 32], 1],
        [[2.3, 106], 0],
        [[2.4, 151], 1],
        [[2.5, 170], 1],
        [[0.50, 150], 2],
        [[1.1, 35], 1],
        [[0.85, 70], 2]]

test = data[2]
iterations = 20
for i in range(iterations):
    x.trainer(test[0], test[1], 1)

'''
test = data[0:10]
train = data[10:20]
iterations = 1000
eta = 2

for i in range(iterations):
    print("Iteration " + str(i))
    for j in train:
        x.trainer([j[0][0], j[0][1]], j[1], eta)

for i in test:
    x.feed_forward(i[0])
'''
