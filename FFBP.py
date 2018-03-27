"""
Implementation of a Feed Forward Back Propagation (FFBP) Multilayer Perceptron Neural Network, consisting of
Perceptron and Network objects. Weights, biases, activation values, errors, and deltas are calculated and stored
within Perceptrons, while the Network object knits together layers of Perceptron objects based on a user-defined
structure. Feed forward and back propagation functions calculate activity values, compare network outputs to
desired outputs, and back propagate error deltas to update weights and biases. The network is iterated upon using
a trainer to minimize error.

"""

import numpy as np

class Perceptron():
    def __init__(self, weights, bias):
        """
        :param weights: length of weight vector has to match number of inputs for each perceptrong
        :param bias: each perceptron also has a bias value that can be adjusted

        Future release to have more flexibility around weight and bias value setting. These should be randomly
        generated if no weights and biases are given. Allowing user-inputted values for my class assignments.
        """
        self.weights = np.array(weights)
        self.updated_weights = None
        self.bias = bias
        self.input_vector = None
        self.activation_value = None
        self.error = None
        self.total_error = None
        self.delta = None

    def sigmoid(self, z):
        """
        :param z: function variable. Signmoid function squeezes input into 0:1 scale.
        :return:

        Future release to enable user to pick different activation functions
        """
        fx = 1 / (1 + np.exp(-z))
        return(fx)

    def calc_activation(self, inputs):
        """

        :param inputs:
        :return:
        """
        self.input_vector = np.array(inputs)
        A = np.dot(inputs, self.weights) + self.bias
        self.activation_value = self.sigmoid(A)
        return(self.activation_value)

    def set_output_delta(self, inputs, desired_output, eta):
        inputs = np.array(inputs)
        self.error = desired_output - self.calc_activation(inputs)
        self.total_error = (1/2)*self.error**2
        self.delta = self.error * \
                     (1 - self.activation_value) * \
                     self.activation_value
        self.updated_weights = self.weights + eta * self.delta * inputs
        self.bias = self.bias + eta * self.delta * 1
        output = str(self.total_error) + ", " + str(self.updated_weights) + ", " + str(self.bias)
        print("     " + output)

    def set_hidden_delta(self, inputs, prev_deltas, prev_weights, eta):
        inputs = np.array(inputs)
        self.delta = (1 - self.activation_value) \
                     * self.activation_value * \
                     np.dot(prev_deltas, prev_weights)
        self.updated_weights = self.weights + eta * self.delta * inputs
        self.bias = self.bias + eta * self.delta * 1
        output = str(self.updated_weights) + ", " + str(self.bias)
        print("     " + output)

class Network():
    def __init__(self, structure, weights, bias):
        self.structure = structure
        self.weights = weights
        self.bias = bias

        self.network = []
        for i, j in enumerate(self.structure):
            layer = []
            k = 0
            while k < j:
                layer.append(Perceptron(self.weights[i][k], self.bias))
                k += 1
            self.network.append(layer)

        self.output_vector = []

    def apply_activation(self, input_vector, perceptron_vector):
        activation = lambda x: x.calc_activation(input_vector)
        apply_activation = np.vectorize(activation)
        return(apply_activation(perceptron_vector))

    def feed_forward(self, inputs):
        print("Feeding Forward...")
        k = 0
        layer_output = []
        while k < len(self.structure):
            print("     Layer " + str(k+1) + " Activations")
            if k == 0:
                layer_output = self.apply_activation(inputs, self.network[k])
            else:
                layer_output = self.apply_activation(layer_output, self.network[k])
            k += 1
            print("     " + str(layer_output))
        self.output_vector = layer_output

    def back_propagate(self, inputs, desired_output, eta):
        print("Back Propagating...")
        k = len(self.structure)-1
        while k >= 0:
            if k == len(self.structure)-1:
                print("     Total Error, Updated Layer " + str(k+1) + " Weights & Biases")
                prev_outputs = [o.activation_value for o in self.network[k-1]]
                [o.set_output_delta(prev_outputs, desired_output, eta) for o in self.network[k]]

            elif k == 0:
                print("     Updated Layer " + str(k+1) + " Weights & Biases")
                prev_deltas = [o.delta for o in self.network[k+1]]
                perceptrons = self.structure[k]
                for i in range(perceptrons):
                    prev_weights = [o.weights[i] for o in self.network[k+1]]
                    self.network[k][i].set_hidden_delta(inputs, prev_weights, prev_deltas, eta)

            else:
                print("     Updated Layer " + str(k+1) + " Weights & Biases")
                prev_deltas = [o.delta for o in self.network[k+1]]
                prev_outputs = [o.activation_value for o in self.network[k-1]]
                perceptrons = self.structure[k]
                for i in range(perceptrons):
                    prev_weights = [o.weights[i] for o in self.network[k+1]]
                    [self.network[k][i].set_hidden_delta(prev_outputs, prev_weights, prev_deltas, eta)]

            k = k-1

        for i, j in enumerate(x.structure):
            for k in range(j):
                self.network[i][k].weights = self.network[i][k].updated_weights

    def trainer(self, inputs, desired_output, iterations, eta):
        for i in range(iterations):
            print("Iteration " + str(i+1))
            if i == 0:
                self.feed_forward(inputs)
                self.back_propagate(inputs, desired_output, eta)
                self.feed_forward(inputs)
            else:
                self.back_propagate(inputs, desired_output, eta)
                self.feed_forward(inputs)