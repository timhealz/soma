import random as rand
import numpy as np

class Network():
    """
    Sets up a square network of boolean 1 and 0 values.
    """
    def __init__(self, nodes, weight_matrix):
        self.dim = nodes
        self.size = self.dim * self.dim
        self.network = np.array([[0 for i in range(nodes)]
                    for i in range(nodes)])

        n = list(range(self.dim))
        self.pairs = []
        for j in range(self.dim):
            i = n.pop(rand.randint(0,len(n)-1))
            pair = (i, j)
            self.pairs.append(pair)
            self.flip(i, j)

        """
        self.network = np.hstack((self.network, np.array([0] * self.dim)[:, np.newaxis],))
        pair = (self.pairs[0][0], self.dim)
        self.flip(pair[0], pair[1])
        self.pairs.append(pair)
        """

        self.weights = weight_matrix
        self.cost = 0

    def flip(self, i, j):
        self.network[i][j] = np.abs(self.network[i][j] - 1)

    def calculate_cost(self, connection):
        weight_ij = (connection[0][0], connection[1][0])
        x_i = connection[0];
        x_j = connection[1]
        calc = self.weights[weight_ij] * self.network[x_i] * self.network[x_j]
        return(calc)

    def consensus(self):
        k = 0
        while k < self.dim-1:
            self.cost += self.calculate_cost([self.pairs[k], self.pairs[k+1]])
            k += 1

        returning_connection = [self.pairs[self.dim-1], (self.pairs[0][0], self.dim-1)]
        self.cost += self.calculate_cost(returning_connection)

        return(self.cost)

#class Anneal()


distances = np.array([[0, 10, 20, 5, 18],
                      [10, 0, 15, 32, 10],
                      [20, 15, 0, 25, 16],
                      [5, 32, 25, 0, 35],
                      [18, 10, 16, 35, 0]])
x = Network(5, distances)
print(x.network)
print(x.pairs)
print(x.consensus())


