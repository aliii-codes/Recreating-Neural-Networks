#!/usr/bin/env python3
"""
Chapter 4: Activation Functions

This script demonstrates activation functions for neural networks,
including ReLU and Softmax, and their implementation with NumPy.
"""

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


def relu_manual():
    """Demonstrate ReLU activation with manual loop."""
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
    outputs = []

    for i in inputs:
        if i > 0:
            outputs.append(i)
        else:
            outputs.append(0)

    print(f"ReLU (manual): {outputs}")
    return outputs


def relu_max():
    """Demonstrate ReLU using max function."""
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
    outputs = []

    for i in inputs:
        outputs.append(max(0, i))

    print(f"ReLU (max): {outputs}")
    return outputs


def relu_numpy():
    """Demonstrate ReLU using NumPy."""
    inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
    outputs = np.maximum(0, inputs)

    print(f"ReLU (NumPy): {outputs}")
    return outputs


def softmax_manual():
    """Demonstrate Softmax activation with manual calculation."""
    layer_outputs = [4.8, 1.21, 2.385]
    import math
    E = math.e

    # Calculate exponential values
    exp_vals = []
    for output in layer_outputs:
        exp_vals.append(E ** output)
    print(f"Exponential values: {exp_vals}")

    # Normalize values
    norm_base = sum(exp_vals)
    norm_vals = []
    for value in exp_vals:
        norm_vals.append(value / norm_base)
    print(f"Normalized values: {norm_vals}")
    print(f"Sum: {sum(norm_vals)}")

    return norm_vals


def softmax_numpy():
    """Demonstrate Softmax using NumPy."""
    layer_outputs = [4.8, 1.21, 2.385]

    exp_values = np.exp(layer_outputs)
    print(f"Exponential values: {exp_values}")

    norm_values = exp_values / np.sum(exp_values)
    print(f"Normalized values: {norm_values}")
    print(f"Sum: {np.sum(norm_values)}")

    return norm_values


def demonstrate_axis():
    """Demonstrate NumPy axis parameter."""
    layer_outputs = [[4.8, 1.21, 2.385],
                      [8.9, -1.81, 0.2],
                      [1.41, 1.051, 0.026]]

    print(f"Sum without axis: {np.sum(layer_outputs)}")
    print(f"Sum with axis=0: {np.sum(layer_outputs, axis=0)}")
    print(f"Sum with axis=0, keepdims: {np.sum(layer_outputs, axis=0, keepdims=True)}")
    print(f"Sum with axis=1: {np.sum(layer_outputs, axis=1)}")


class Activation_ReLU:
    """ReLU activation function."""

    def forward(self, inputs):
        """
        Forward pass through ReLU activation.

        Args:
            inputs: Input data

        Returns:
            Output with ReLU applied
        """
        self.output = np.maximum(0, inputs)
        return self.output


class Activation_Softmax:
    """Softmax activation function."""

    def forward(self, inputs):
        """
        Forward pass through Softmax activation.

        Args:
            inputs: Input data

        Returns:
            Output with Softmax applied (probability distribution)
        """
        # Get unnormalized probabilities
        # Subtract max to prevent overflow
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        self.output = probabilities
        return self.output


class Layer_Dense:
    """Dense (fully-connected) layer for neural networks."""

    def __init__(self, n_inputs, n_neurons):
        """
        Initialize the dense layer.

        Args:
            n_inputs: Number of input features
            n_neurons: Number of neurons in the layer
        """
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        """
        Forward pass through the layer.

        Args:
            inputs: Input data (batch of samples)

        Returns:
            Output of the layer
        """
        self.output = np.dot(inputs, self.weights) + self.biases
        return self.output


def demonstrate_full_network():
    """Demonstrate a full neural network with activations."""
    nnfs.init()

    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create Dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)

    # Create ReLU activation
    activation1 = Activation_ReLU()

    # Create second Dense layer with 3 input features and 3 output values
    dense2 = Layer_Dense(3, 3)

    # Create Softmax activation
    activation2 = Activation_Softmax()

    # Forward pass through first layer
    dense1.forward(X)
    activation1.forward(dense1.output)

    # Forward pass through second layer
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print("First 5 softmax outputs:")
    print(activation2.output[:5])

    return activation2


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 4: Activation Functions")
    print("=" * 60)
    print()

    # ReLU demonstrations
    print("1. ReLU Activation")
    print("-" * 60)
    relu_manual()
    relu_max()
    relu_numpy()
    print()

    # Softmax demonstrations
    print("2. Softmax Activation")
    print("-" * 60)
    softmax_manual()
    print()
    softmax_numpy()
    print()

    # Axis demonstration
    print("3. NumPy Axis Parameter")
    print("-" * 60)
    demonstrate_axis()
    print()

    # Full network demonstration
    print("4. Full Neural Network with Activations")
    print("-" * 60)
    demonstrate_full_network()
    print()

    print("=" * 60)
    print("Note: Softmax subtracts max to prevent overflow errors")
    print("when exponentiating large numbers.")
    print("=" * 60)
