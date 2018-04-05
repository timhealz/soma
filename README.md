
# Soma

This repository holds the code for neural networks I've implemented for my Neural Networks course at JHU.

Future plans are to implement a framework for Recurrent Neural Networks.

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
