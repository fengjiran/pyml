# Simple implementation for 1 layer perceptron
# primary learning algorithm
import numpy as np
import numpy.linalg as LA


class Perceptron(object):
    def __init__(self, dim=2, eta=1, max_epoch=None):
        self.dim = dim
        self.eta = eta
        self.W = np.zeros(dim)
        self.b = np.zeros(1)
        self.max_epoch = 1000

    def fit(self, x, y, detailed=False):
        pass
