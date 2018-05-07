"""
Implementation of a Boltzmann Machine with Simulated Annealing to solve the Traveling Salesman
Problem (TSP), consisting of Network and Boltzmann objects.

Author: Tim Healy
"""

import random as rand
import numpy as np
import copy
import seaborn as sb
import matplotlib.pyplot as plt

class Network():
    """
    The Network object initializes a n x n matrix of n random linearly independent unit vectors.
    his matrix is setup in this way because of TSP constraints: cannot visit two cities at the
    same time, and cannot visit cities twice. It takes a list of location names and a distance
    matrix as its arguments.
    """
    def __init__(self, cities, distance_matrix):
        # City names
        self.cities = cities
        # Dimension of n x n matrix
        self.dim = len(distance_matrix)
        # Generate a matrix of zeros
        self.network = np.array([[0 for i in range(self.dim)]
                    for i in range(self.dim)])

        # Create "jar" of n possible rows to choose from in matrix
        jar = list(range(self.dim))
        # For each column in matrix...
        for j in range(self.dim):
            # Select random row from "jar"
            i = jar.pop(rand.randint(0,len(jar)-1))
            # Flip the node state from 0 to 1
            self.flip(i, j)
        # This ensures that the unit vectors are lineraly independent, to accommodate the TSP
        # constraints.

        # Useful information is stored in the object
        # List of where the activated nodes are in the matrix
        self.pairs = list(zip(*np.nonzero(self.network)))
        # City route of current network
        self.route = [self.cities[i[1]] for i in self.pairs]
        # Appends the starting city to the aforementioned route
        self.route.append(self.route[0])
        # Distance matrix
        self.weights = distance_matrix
        # Calculates energy of matrix
        self.calculate_energy()

    def flip(self, i, j):
        """
        Helper function to flip states of nodes within the network. Takes row and column as its input.
        Used to initialize the network itself.
        """
        self.network[i][j] = np.abs(self.network[i][j] - 1)

    def calculate_cost(self, connection):
        """
        Helper function to calculate energy of a single City -> City connection. Connection input is
        a list of two activated nodes [(2,0), (3,2)] for example. Used in energy calculation.
        """
        # Index of weight to use from the connection pairs
        weight_ij = (connection[0][1], connection[1][1])
        # Activated node states, will equal 1 in this problem
        x_i = connection[0]
        x_j = connection[1]
        # Calculation of energy for the connection
        calc = self.weights[weight_ij] * self.network[x_i] * self.network[x_j]
        return(calc)

    def calculate_energy(self):
        """
        Calculates energy of the network object. Energy is used in the anneal process to optimize the
        route in the TSP. The return trip back to the initial city is also calculated and added to
        the energy.
        """
        # Initialize the energy value stored in the object
        self.energy = 0
        # While loop to calculate energy for each city-to-city connection, using pairs information
        k = 0
        while k < self.dim-1:
            self.energy += self.calculate_cost([self.pairs[k], self.pairs[k+1]])
            k += 1
        # Return trip must also be added to the energy per the problem outline
        returning_connection = [self.pairs[self.dim-1], self.pairs[0]]
        self.energy += self.calculate_cost(returning_connection)

        return(self.energy)

    def shuffle(self):
        """
        Shuffles the network by swapping columns of linearly independent unit vectors.
        """
        # Determine random columns to swap
        seed = rand.sample(list(range(self.dim)), 2)
        # Swap columns
        self.network[:, [seed[0], seed[1]]] = self.network[:, [seed[1], seed[0]]]
        # Recalculate pairs, route, and energy for shuffled network
        self.pairs = tuple(zip(*np.nonzero(self.network)))
        self.route = [self.cities[i[1]] for i in self.pairs]
        self.route.append(self.route[0])
        self.calculate_energy()

class Boltzmann():
    """
    Boltzmann object stochastically anneals on the Network object to reach an optimized state,
    minimizing the distance between cities.
    """
    def __init__(self, cities, distance_matrix):
        # Initialize an Optimized version of the network
        self.optimized = Network(cities, distance_matrix)
        # Logs track iterations of train function
        self.logs = []

    def anneal(self):
        """
        Makes a copy of the initialized network, shuffles it, and tests to see if the energy
        of the shuffled network is lower than the current network.
        """
        # Create a copy of the Optimized version of the network
        self.working = copy.deepcopy(self.optimized)
        # Shuffle the Working version
        self.working.shuffle()

        # Compare the Working version to the Optimized version. If the energy of the Working version is
        # less than the Optimized version, then the Optimized version takes on the states of the Working version
        if self.optimized.energy < self.working.energy:
            self.optimized = self.optimized
        else:
            self.optimized = copy.deepcopy(self.working)

    def train(self, iterations):
        """
        Iterator to anneal the network based on user inputted iterations.
        """
        for i in range(iterations):
            self.anneal()
            self.logs.append(self.optimized.energy)

        print("Optimized Route: " + " -> ".join(self.optimized.route))
        print("Distance: " + str(self.optimized.energy))

        # Plot of logs
        sb.set_style('darkgrid')
        plt.plot(self.logs)
        plt.title("Optimized Distance")
        plt.xlabel("Iterations")
        plt.ylabel("Distance")
        plt.show()
