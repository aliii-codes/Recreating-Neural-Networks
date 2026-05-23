import numpy as np

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T)


class Activation_ReLU:
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


np.random.seed(0)

layer = Layer_Dense(4, 3)
relu = Activation_ReLU()

X = np.array([[ 1.0,  2.0,  3.0,  2.5],
              [ 2.0,  5.0, -1.0,  2.0],
              [-1.5,  2.7,  3.3, -0.8]])

layer.forward(X)
relu.forward(layer.output)

print("Forward output (ReLU):\n", relu.output)

dvalues = np.ones_like(relu.output)

relu.backward(dvalues)
layer.backward(relu.dinputs)

print("\nGradients (dweights):\n", layer.dweights)
print("Gradients (dbiases):\n", layer.dbiases)

learning_rate = 0.001
layer.weights -= learning_rate * layer.dweights
layer.biases  -= learning_rate * layer.dbiases

print("\nUpdated weights:\n", layer.weights)
print("Updated biases:\n", layer.biases)