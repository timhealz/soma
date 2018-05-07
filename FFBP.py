import numpy as np
import random as rand


# Activation functions
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return (y)


def ramp(x):
    y = np.log(1 + np.exp(x))
    return (y)


class Perceptron():
    '''
    Perceptron object is where all of the calculations are performed.
    '''

    def __init__(self, num_inputs, type):
        '''
        'Type' determines whether the perceptron is of the input,
        hidden, or output variety. Weights and bias values are randomly
        generated based on how many inputs are fed into the network.
        '''
        self.type = type
        self.weights = np.array([rand.uniform(0, 1) for x in range(num_inputs)])
        self.bias = rand.uniform(0, 1)

    def calc_activation(self, inputs):
        '''
        Activation is calculated using sigmoid function for hidden / input
        layers; ramp function for output layer
        '''
        self.A = np.dot(inputs, self.weights) + self.bias

        if self.type == 2:
            # self.activation_value = sigmoid(self.A)
            self.activation_value = ramp(self.A)

        else:
            self.activation_value = sigmoid(self.A)

    def calc_output_delta(self, desired_output):
        '''
        Output delta calculation utilizin the sigmoid function, which is the
        dervative of the ramp function
        '''
        self.error = desired_output - self.activation_value
        self.total_error = (1 / 2) * self.error ** 2
        self.delta = self.error * sigmoid(self.A)

    def calc_hidden_delta(self, prev_deltas, prev_weights):
        '''
        Hidden delta calculation utlizing derivative of sigmoid activation
        function
        '''
        self.delta = (1 - self.activation_value) * self.activation_value * np.dot(prev_deltas, prev_weights)

    def update_weights_bias(self, inputs, eta):
        '''
        Updates the weights and biases, using the deltas calculated above
        '''
        self.updated_weights = np.add(self.weights, np.multiply(eta * self.delta, inputs))
        self.updated_bias = self.bias + eta * self.delta * 1


class Network():
    '''
    Network object sets up layers of perceptron objects and performs operations
    on the perceptrons utilizing the perceptron's built in functions.
    '''

    def __init__(self, num_inputs, structure):
        '''
        Structure input determines how many layers and perceptrons are in the network.
        '''
        self.structure = structure
        self.num_inputs = num_inputs
        self.network = []

        # Build network based on the inputted structure variable
        for i, j in enumerate(self.structure):

            if i == 0:
                layer = [Perceptron(num_inputs, 0) for x in range(j)]

            elif i == len(self.structure) - 1:
                layer = [Perceptron(self.structure[i - 1], 2) for x in range(j)]

            else:
                layer = [Perceptron(self.structure[i - 1], 1) for x in range(j)]

            self.network.append(layer)

    def feed_forward(self, inputs):
        '''
        Calculate activations for each layer of network
        '''
        self.activations = [];
        self.weights = [];
        self.bias = []

        for i, layer in enumerate(self.network):

            if i == 0:
                [perceptron.calc_activation(inputs) for perceptron in layer]

            else:
                input_activations = [perceptron.activation_value for perceptron in self.network[i - 1]]
                [perceptron.calc_activation(input_activations) for perceptron in layer]

            self.activations.append([perceptron.activation_value for perceptron in layer])
            self.weights.append([perceptron.weights for perceptron in layer])
            self.bias.append([perceptron.bias for perceptron in layer])

    def back_propagate(self, inputs, desired_output, eta):
        '''
        Lot of list comprehensions going on here to capture back propagation, but all
        that it's doing is essentially calculting deltas and updating weights for each
        layer of the network. It has to use different routines based on what kind of
        layer it's working on.
        '''
        k = len(self.structure) - 1
        self.updated_weights = []

        while k >= 0:

            # Output layer back propagation
            if k == len(self.structure) - 1:
                [perceptron.calc_output_delta(desired_output) for perceptron in self.network[k]]
                input_weights = [perceptron.activation_value for perceptron in x.network[k - 1]]
                [perceptron.update_weights_bias(input_weights, eta) for perceptron in self.network[k]]

            # Input layer back propagation
            elif k == 0:
                for i, perp in enumerate(x.network[k]):
                    prev_weights = [perceptron.weights[i] for perceptron in self.network[k + 1]]
                    prev_deltas = [perceptron.delta for perceptron in self.network[k + 1]]
                    self.network[k][i].calc_hidden_delta(prev_weights, prev_deltas)

                    perp.update_weights_bias(inputs, eta)

            # hidden layer back propagation
            else:
                for i, perp in enumerate(x.network[k]):
                    prev_weights = [perceptron.weights[i] for perceptron in self.network[k + 1]]
                    prev_deltas = [perceptron.delta for perceptron in self.network[k + 1]]
                    self.network[k][i].calc_hidden_delta(prev_weights, prev_deltas)

                    input_weights = [perceptron.activation_value for perceptron in x.network[k - 1]]
                    perp.update_weights_bias(input_weights, eta)

            self.updated_weights.insert(0, [perceptron.updated_weights for perceptron in self.network[k]])
            k = k - 1

        # replace old weights with updated weights
        for layer in self.network:
            for perceptron in layer:
                perceptron.weights = perceptron.updated_weights
                perceptron.bias = perceptron.updated_bias

    # execute FFBP
    def FFBP(self, inputs, desired_output, eta):
        self.feed_forward(inputs)
        self.back_propagate(inputs, desired_output, eta)
        self.feed_forward(inputs)
