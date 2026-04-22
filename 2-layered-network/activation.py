import numpy as np

class Acivations:
    def sigmoid(x):  # x is the input to the activation function
        return 1 / (1 + np.exp(-x))
    
    def relu(x):
        return np.maximum(0, x)
    
    def leaky_relu(x):
        return np.maximum(0.01 * x, x)
    
    def tanh(x):
        return np.tanh(x)
    

    
