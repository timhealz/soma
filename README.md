

# Soma

This repository holds the code for neural networks I've implemented for my Neural Networks course at JHU. `FFBP.py` and `Boltzmann.py` are objects used to implement
Feed Forward Back Propagation Networks and Boltzmann Machines, respectively.

#### Feed Forward Back Propagation (FFBP) Multilayer Perceptron Network

A FFBP perceptron network can be created and trained like so:

```python
from soma import FFBP
# Input data and initial parameters
inputs = [1, 2]
desired_output = 0.7
structure = [2, 1]
# Length of list = numbers of layers
# Items in list = number of perceptrons in each layer
weights = [[[0.3, 0.3], [0.3, 0.3]], [[0.8, 0.8]]]
bias = 0

# Initialize network
x = Network(structure, weights, bias)
# Train with input data for 15 iterations
x.trainer(inputs, desired_output, 15, 1)
```
This example trains the network on one input - output pair for 15 iterations. More input - output pairs can be presented to train the network.

The built-in trainer utilizes a batch method. An online training method can be used if the user sets up a loop that alternates presentation of inputs to the network.


#### Boltzmann Machine with Simulated Annealing for the Traveling Salesman Problem

A Boltzmann Machine can be created and trained as follows. It takes a distance matrix as its only argument. The row and column indices in the matrix represents
cities, and thus, each entry represents the distance between cities. For example, entry (0,1), represents the distance between city A and B, which is 10. This matrix is symmetric, since the distance from A to B is the same as the distance from B to A, or (0,1) = (1,0).

```python
import Boltzmann as boltz
import numpy as np

distances = np.array([[   0, 2451,  713, 1018, 1631, 1374, 2408,  213, 2571,  875, 1420, 2145, 1972], # New York
  [2451,    0, 1745, 1524,  831, 1240,  959, 2596,  403, 1589, 1374,  357,  579], # Los Angeles
  [ 713, 1745,    0,  355,  920,  803, 1737,  851, 1858,  262,  940, 1453, 1260], # Chicago
  [1018, 1524,  355,    0,  700,  862, 1395, 1123, 1584,  466, 1056, 1280,  987], # Minneapolis
  [1631,  831,  920,  700,    0,  663, 1021, 1769,  949,  796,  879,  586,  371], # Denver
  [1374, 1240,  803,  862,  663,    0, 1681, 1551, 1765,  547,  225,  887,  999], # Dallas
  [2408,  959, 1737, 1395, 1021, 1681,    0, 2493,  678, 1724, 1891, 1114,  701], # Seattle
  [ 213, 2596,  851, 1123, 1769, 1551, 2493,    0, 2699, 1038, 1605, 2300, 2099], # Boston
  [2571,  403, 1858, 1584,  949, 1765,  678, 2699,    0, 1744, 1645,  653,  600], # San Francisco
  [ 875, 1589,  262,  466,  796,  547, 1724, 1038, 1744,    0,  679, 1272, 1162], # St. Louis
  [1420, 1374,  940, 1056,  879,  225, 1891, 1605, 1645,  679,    0, 1017, 1200], # Houston
  [2145,  357, 1453, 1280,  586,  887, 1114, 2300,  653, 1272, 1017,    0,  504], # Phoenix
  [1972,  579, 1260,  987,  371,  999,  701, 2099,  600, 1162,  1200,  504,   0]]) # Salt Lake City
cities = ['New York', 'Los Angeles', 'Chicago', 'Minneapolis', 'Denver', 'Dallas', 'Seattle',
          'Boston', 'San Francisco', 'St. Louis', 'Houston', 'Phoenix', 'Salt Lake City' ]

x = boltz.Boltzmann(cities, distances)
x.train(500)
```

With output:
```shell
Optimized Route: Dallas -> St. Louis -> New York -> Boston -> Chicago -> Minneapolis -> Denver
-> Salt Lake City -> Seattle -> San Francisco -> Los Angeles -> Phoenix -> Houston -> Dallas
Distance: 7293
```
![](examples/Boltzmann_example.png)

This example is captured in the `examples/Boltzmann_example_1.py` script
