import numpy as np

def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return (y)

def ramp(x):
    y = np.log(1 + np.exp(x))
    return(y)