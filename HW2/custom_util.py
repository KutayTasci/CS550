import numpy as np


class BatchGenerator():
    def __init__(self, x, y, batch_size=1, shuffle=False, seed=None):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

        # shuffle data
        if self.shuffle:
            np.random.seed(self.seed)
            ind = np.random.permutation(x.shape[0])
            self.x = self.x[ind]
            self.y = self.y[ind]

        self.batchnum = np.ceil(x.shape[0] / self.batch_size)
        self.lastbatch_size = x.shape[0] % self.batch_size

        self.start = 0

    def getBatch(self):
        end = min(self.start + self.batch_size, self.x.shape[0])
        multiplier = 1

        if (end == self.x.shape[0]):
            if (self.lastbatch_size != 0):
                multiplier = self.lastbatch_size / self.batch_size

            xbatch = self.x[self.start:end, :]
            ybatch = self.y[self.start:end, :]

            # print(self.start, end, multiplier)

            self.start = 0

            if self.shuffle:
                np.random.seed(self.seed)
                ind = np.random.permutation(self.x.shape[0])
                self.x = self.x[ind]
                self.y = self.y[ind]
        else:
            xbatch = self.x[self.start:end, :]
            ybatch = self.y[self.start:end, :]

            # print(self.start, end, multiplier)

            self.start = end

        return xbatch, ybatch, multiplier

def ReLU(x):
    return np.maximum(x,0)

def d_ReLU(x):
    return np.where(x > 0, 1, 0)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - x ** 2

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def d_sigmoid(x):
    return x * (1 - x)

def d_mse(y, y_hat):
    return -2 * (y - y_hat) / len(y)

def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)

def no_activation(x):
    return x

def d_no_activation(x):
    return np.ones(x.shape)

class standard_scaler():
    def __init__(self, array):
        self.mean = np.mean(array)
        self.std = np.std(array)

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_transform(self, x):
        return x * self.std + self.mean