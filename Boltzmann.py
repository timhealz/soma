import random as rand
import numpy as np
import copy

class Network():
    """
    Sets up a square network of boolean 1 and 0 values.
    """
    def __init__(self, nodes, weight_matrix):
        self.dim = nodes
        self.size = self.dim * self.dim
        self.network = np.array([[0 for i in range(self.dim)]
                    for i in range(self.dim)])
        self.network = np.transpose(self.network)

        n = list(range(self.dim))
        for j in range(self.dim):
            i = n.pop(rand.randint(0,len(n)-1))
            self.flip(i, j)

        self.pairs = list(zip(*np.nonzero(self.network)))
        self.weights = weight_matrix
        self.consensus = self.calculate_consensus()

    def flip(self, i, j):
        self.network[i][j] = np.abs(self.network[i][j] - 1)

    def calculate_cost(self, connection):
        """
        Helper function to calculate distance from any given node
        :param connection: two tuples representing nodes in matrix. [(0,1), (2,3)] for example
        :return:
        """
        weight_ij = (connection[1][0], connection[1][1])
        x_i = connection[0]
        x_j = connection[1]
        calc = self.weights[weight_ij] * self.network[x_i] * self.network[x_j]
        return(calc)

    def calculate_consensus(self):
        self.consensus = 0
        k = 0
        while k < self.dim-1:
            self.consensus += self.calculate_cost([self.pairs[k], self.pairs[k+1]])
            k += 1

        returning_connection = [self.pairs[self.dim-1], (self.pairs[0][0], self.dim-1)]
        self.consensus += self.calculate_cost(returning_connection)

        return(self.consensus)

    def shuffle(self):
        seed = rand.sample(list(range(self.dim)), 2)
        self.network[:, [seed[0], seed[1]]] = self.network[:, [seed[1], seed[0]]]
        self.pairs = tuple(zip(*np.nonzero(self.network)))
        self.calculate_consensus()

class Boltzmann():
    def __init__(self, nodes, weight_matrix):
        self.optimized = Network(nodes, weight_matrix)
        self.working = copy.deepcopy(self.optimized)

    def anneal(self):
        self.working = copy.deepcopy(self.optimized)
        self.working.shuffle()

        if self.optimized.consensus < self.working.consensus:
            self.optimized = self.optimized
        else:
            self.optimized = copy.deepcopy(self.working)

        print(self.optimized.network)
        print(self.optimized.consensus)

    def train(self, iterations):
        for i in range(iterations):
            self.anneal()

distances = np.array([[0, 10, 20, 5, 18],
                      [10, 0, 15, 32, 10],
                      [20, 15, 0, 25, 16],
                      [5, 32, 25, 0, 35],
                      [18, 10, 16, 35, 0]])
x = Network(5, distances)

x = Boltzmann(5, distances)
x.train(10)