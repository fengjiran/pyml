# Simple implementation for 1 layer perceptron
# primary learning algorithm
import numpy as np
# import numpy.linalg as LA
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt


class Perceptron(object):
    def __init__(self, n_iter=1, eta=0.01):
        self.n_iter = n_iter
        self.eta = eta
        self.w = None

    def fit(self, X, y, func='batch', detailed=False):
        n_samples, n_features = X.shape
        self.w = np.ones(n_features, dtype=np.float64)

        if detailed is True:
            plt.ion()

        if func == 'batch':
            for t in range(self.n_iter):
                error_cnt = 0
                for i in range(n_samples):
                    if self.predict(X[i])[0] != y[i]:
                        error_cnt += 1
                        self.w += self.eta * y[i] * X[i] / self.n_iter
                if detailed:
                    self.plot_process(X)
                if error_cnt == 0:
                    break
        elif func == 'SGD' or func == 'sgd':
            pass

    def predict(self, X):
        X = np.atleast_2d(X)
        return np.sign(np.dot(X, self.w))


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

    # def fit(self, x, y, detailed=False):
    #     i = 0
    #     cnt = 0
    #     epoch = 0
    #     finished = True
    #     self.w = None
    #
    #
    #     while cnt != x.shape[0] and (self.max_epoch is None or epoch < self.max_epoch):
    #         cnt += 1
    #         if y[i] * (np.sum(self.W * x[i, :], axis=-1) + self.b) <= 0:
    #             self.W += self.eta * y[i] * x[i, :]
    #             self.b += self.eta * y[i]
    #             cnt = 0
    #             epoch += 1
    #             if detailed is True:
    #                 print("Epoch: ", epoch, ", W: ", self.W, ", b: ", self.b)
    #
    #         i = (i + 1) % x.shape[0]


if __name__ == '__main__':
    # load dataset
    iris_data = load_iris()
    y = np.sign(iris_data.target[0:100] - 0.5)
    X = iris_data.data[0:100]
    print(X.shape)
    X = np.c_[(np.array([1] * X.shape[0])).T, X]
    print(X.shape)
