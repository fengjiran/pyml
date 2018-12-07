# Simple implementation for 1 layer perceptron
# primary learning algorithm
import numpy as np
import numpy.linalg as LA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, dim=2, eta=1, max_epoch=None):
        self.dim = dim
        self.eta = eta
        self.W = np.zeros(dim)
        self.b = np.zeros(1)
        self.max_epoch = 1000

    def plot_process(self, X):
        fig = plt.figure()
        fig.clear()

        # 绘制样本点分布
        plt.scatter(X[0:50, 1], X[0:50, 2], c='r')
        plt.scatter(X[50:100, 1], X[50:100, 2], c='b')

        xx = np.arange(X[:, 1].min(), X[:, 1].max(), 0.1)
        yy = -(self.w[1] * xx + self.w[0]) / self.w[2]
        plt.plot(xx, yy)

        plt.grid()
        plt.pause(1.5)

    def fit(self, x, y, detailed=False):
        i = 0
        cnt = 0
        epoch = 0
        finished = True
        self.w = None


        while cnt != x.shape[0] and (self.max_epoch is None or epoch < self.max_epoch):
            cnt += 1
            if y[i] * (np.sum(self.W * x[i, :], axis=-1) + self.b) <= 0:
                self.W += self.eta * y[i] * x[i, :]
                self.b += self.eta * y[i]
                cnt = 0
                epoch += 1
                if detailed is True:
                    print("Epoch: ", epoch, ", W: ", self.W, ", b: ", self.b)

            i = (i + 1) % x.shape[0]


if __name__ == '__main__':
    # load dataset
    iris_data = load_iris()
    y = np.sign(iris_data.target[0:100] - 0.5)
    X = iris_data.data[0:100]
    print(X.shape)
    X = np.c_[(np.array([1] * X.shape[0])).T, X]
    print(X.shape)
