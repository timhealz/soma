"""
Implementation of a Boltzmann Machine with Simulated Annealing to solve the Traveling Salesman Problem (TSP),
consisting of Network and Boltzmann objects.

Author: Tim Healy
"""

import random as rand
import numpy as np
import copy
import seaborn as sb
import matplotlib.pyplot as plt

class Network():
    """
    The Network object initializes a n x n matrix of n random linearly independent unit vectors. This matrix is
    setup in this way because of TSP constraints: cannot visit two cities at the same time, and cannot visit
    cities twice.

    It takes a distance matrix as its only argument. The row and column indices in the matrix represents cities,
    and thus, each entry represents the distance between cities. For example, entry (0,1), represents the distance
    between city A and B, which is 10. This matrix is symmetric, since the distance from A to B is the same as the
    distance from B to A, or (0,1) = (1,0).
    """
    def __init__(self, cities, weight_matrix):
        self.cities = cities
        self.dim = len(weight_matrix)
        self.network = np.array([[0 for i in range(self.dim)]
                    for i in range(self.dim)])

        jar = list(range(self.dim))
        for j in range(self.dim):
            i = jar.pop(rand.randint(0,len(jar)-1))
            self.flip(i, j)

        self.pairs = list(zip(*np.nonzero(self.network)))
        self.route = [self.cities[i[1]] for i in self.pairs]
        self.route.append(self.route[0])
        self.weights = weight_matrix
        self.calculate_energy()

    def flip(self, i, j):
        """
        Helper function to flip states of nodes within the network. Used to initialize the network itself.
        """
        self.network[i][j] = np.abs(self.network[i][j] - 1)

    def calculate_cost(self, connection):
        """
        Helper function to calculate distance from activated nodes. Used to calculate energy
        """
        weight_ij = (connection[0][1], connection[1][1])
        x_i = connection[0]
        x_j = connection[1]
        calc = self.weights[weight_ij] * self.network[x_i] * self.network[x_j]
        return(calc)

    def calculate_energy(self):
        """
        Calculates energy of the network object. Energy is used in the anneal process to optimize the
        route in the TSP. The return trip back to the initial city is also calculated and added to the energy.
        """
        self.energy = 0
        k = 0
        while k < self.dim-1:
            self.energy += self.calculate_cost([self.pairs[k], self.pairs[k+1]])
            k += 1

        returning_connection = [self.pairs[self.dim-1], self.pairs[0]]
        self.energy += self.calculate_cost(returning_connection)

        return(self.energy)

    def shuffle(self):
        """
        Shuffles the network by swapping columns of linearly independent unit vectors.
        """
        seed = rand.sample(list(range(self.dim)), 2)
        self.network[:, [seed[0], seed[1]]] = self.network[:, [seed[1], seed[0]]]
        self.pairs = tuple(zip(*np.nonzero(self.network)))
        self.route = [self.cities[i[1]] for i in self.pairs]
        self.route.append(self.route[0])
        self.calculate_energy()

class Boltzmann():
    """
    Boltzmann object stochastically anneals on the Network object to reach an optimized state, minimizing
    the distance between cities.
    """
    def __init__(self, cities, weight_matrix):
        self.optimized = Network(cities, weight_matrix)
        self.logs = []

    def anneal(self):
        """
        Makes a copy of the initialized network, shuffles it, and tests to see if the energy of the shuffled
        network is lower than the current network.
        """
        self.working = copy.deepcopy(self.optimized)
        self.working.shuffle()

        if self.optimized.energy < self.working.energy:
            self.optimized = self.optimized
        else:
            self.optimized = copy.deepcopy(self.working)

    def train(self, iterations):
        """
        Iterator to anneal the network.
        """
        for i in range(iterations):
            self.anneal()
            self.logs.append(self.optimized.energy)

        print("Optimized Route: " + " -> ".join(self.optimized.route))
        print("Distance: " + str(self.optimized.energy))

        sb.set_style('darkgrid')
        plt.plot(self.logs)
        plt.title("Optimized Distance")
        plt.xlabel("Iterations")
        plt.ylabel("Distance")
        plt.show()