import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from custom_util import *


#print(train_data, train_labels)


train_data = pd.read_csv("train2.txt", sep="\t", header=None)
# train_data = shuffle(train_data)
train_labels = np.array(train_data.drop(0, axis=1))
train_data = np.array(train_data.drop(1, axis=1))
test_data = pd.read_csv("test2.txt", sep="\t", header=None)
test_labels = np.array(test_data.drop(0, axis=1))
test_data = np.array(test_data.drop(1, axis=1))


# print(train_data, train_labels)


class FCLayer:
    def __init__(self, input_size, output_size, activation_function, d_activation_function):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.d_activation_function = d_activation_function
        self.initialize_weights_random((input_size, output_size))
        self.weights_momentum = np.zeros(self.weights.shape)
        self.bias_momentum = np.zeros((1, output_size))
        # print(self.weights)

        # print(self.bias)

    def forward(self, input):
        self.input = input
        self.output = self.activation_function(np.dot(self.input, self.weights) + self.bias)
        return self.output

    def initialize_weights_random(self, shape):
        self.weights = np.random.randn(shape[0], shape[1])
        self.bias = np.zeros((1, shape[1]))

    def initialize_weights_glorot(self, shape):
        self.weights = np.random.normal(0, 2 / shape[0], shape)
        self.bias = np.zeros((1, shape[1]))

    def initialize_weights_xavier(self, shape):
        self.weights = np.random.normal(0, 1 / np.sqrt(shape[0]), shape)
        self.bias = np.zeros((1, shape[1]))


    def initialize_weights_zero(self, shape):
        self.weights = np.zeros((shape[0], shape[1]))
        self.bias = np.zeros((1, shape[1]))


    def backward(self, error):
        self.error = error
        self.delta = self.error * self.d_activation_function(self.output)

        self.weights_grad = np.dot(self.input.T, self.delta)
        self.bias_grad = self.delta.mean(axis=0)
        self.input_grad = np.dot(self.delta, self.weights.T)
        return self.input_grad

    def update(self, learning_rate):
        self.weights -= learning_rate * self.weights_grad
        self.bias -= learning_rate * self.bias_grad

    def update_momentum(self, learning_rate, momentum):
        self.weights_momentum = momentum * self.weights_momentum + learning_rate * self.weights_grad
        self.bias_momentum = momentum * self.bias_momentum + learning_rate * self.bias_grad
        self.weights -= self.weights_momentum
        self.bias -= self.bias_momentum


class Network():
    def __init__(self, input_size, hidden_size, output_size, learning_rate, activation_function, d_activation_function):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.layers = []
        self.layers.append(FCLayer(input_size, hidden_size, activation_function, d_activation_function))
        self.layers.append(FCLayer(hidden_size, output_size, no_activation, d_no_activation))

    def forward(self, input):
        for layer in self.layers:
            input = layer.forward(input)
        return input

    def backward(self, error):
        for layer in reversed(self.layers):
            error = layer.backward(error)
            layer.update_momentum(self.learning_rate,0.9)

    def train(self, input, label):
        output = self.forward(input)
        # print(output)
        error = d_mse(label, output)
        self.backward(error)

    def train_minibatch(self, input, label, batch_size):
        for i in range(0, len(input), batch_size):
            self.train(input[i:i + batch_size], label[i:i + batch_size])

    def train_sgd(self, input, label, batch_size):
        for i in range(len(input)):
            random_index = np.random.randint(0, len(input))
            self.train(input[random_index], label[random_index])

    def train_epoch(self, input, label, epochs, batch_size):
        total_loss = 0
        for i in range(epochs):
            bgen = BatchGenerator(input, label, shuffle=True, batch_size=len(input))
            input, label, _ = bgen.getBatch()
            self.train_sgd(input, label, batch_size=batch_size)
            loss = mse(label, self.predict(input))
            total_loss += loss
            print("epoch:", i, "error:", loss)
        print("average loss:", total_loss / epochs)
    def predict(self, input):
        output = self.forward(input)
        return output


scaler_dat = standard_scaler(train_data)
scaler_lb = standard_scaler(train_labels)

model = Network(input_size=1, hidden_size=8, output_size=1, learning_rate=0.0005, activation_function= tanh, d_activation_function= d_tanh)
train_data_n = scaler_dat.transform(train_data)
train_labels_n = scaler_lb.transform(train_labels)

model.train_epoch(train_data_n, train_labels_n, epochs=5000, batch_size=15) #you can ignore the batch size when using Stochastic training
test_data_n = scaler_dat.transform(test_data)
test_labels_n = scaler_lb.transform(test_labels)


lineer = np.linspace(-1.5, 2.0, 100)
lineer = np.reshape(lineer, (-1, 1))
y_pred = model.predict(lineer)
y_pred_test = model.predict(test_data_n)

er = mse(test_labels_n, y_pred_test)
print("error on test set:", er)
plt.plot(lineer, y_pred, 'b-', label='Function')
#plt.plot(test_data_n, test_labels_n, 'go', label='Test Data')
plt.plot(train_data_n, train_labels_n, 'ro', label='Train Data')

plt.legend()
plt.show()
