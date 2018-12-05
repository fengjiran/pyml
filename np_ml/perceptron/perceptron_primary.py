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
        i = 0
        cnt = 0
        epoch = 0
        finished = True

        while cnt != x.shape[0] and (self.max_epoch is None or epoch < self.max_epoch):
            cnt += 1
            if y[i] * (np.sum(self.W * x[i, :], axis=-1) + self.b) <= 0:
                pass
