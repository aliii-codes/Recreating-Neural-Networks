#!/usr/bin/env python3
"""
Chapter 3: Adding Layers

This script demonstrates how to build neural networks with multiple layers
using a structured class-based approach, including the Dense Layer class.
"""

import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt


def multi_layer_manual():
    """Demonstrate multiple layers with manual calculation."""
    inputs = [[1, 2, 3, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]

    weights = [[0.2, 0.8, -0.5, 1.0],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]

    biases = [2.0, 3.0, 0.5]

    weights2 = [[0.1, -0.14, 0.5],
                [-0.5, 0.12, -0.33],
                [-0.44, 0.73, -0.13]]

    biases2 = [-1.0, 2.0, -0.5]

    layer_1_output = np.dot(inputs, np.array(weights).T) + biases
    layer_2_output = np.dot(layer_1_output, np.array(weights2).T) + biases2

    print("Layer 1 output:")
    print(layer_1_output)
    print("\nLayer 2 output:")
    print(layer_2_output)
    return layer_1_output, layer_2_output


def generate_spiral_data():
    """Generate and visualize spiral data using nnfs."""
    nnfs.init()

    X, y = spiral_data(samples=100, classes=3)

    # Plot without colors
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(X[:, 0], X[:, 1])
    plt.title("Spiral Data (No Colors)")

    # Plot with colors
    plt.subplot(1, 2, 2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap="brg")
    plt.title("Spiral Data (With Class Colors)")

    plt.tight_layout()
    plt.show()

    return X, y


class Layer_Dense:
    """Dense (fully-connected) layer for neural networks."""

    def __init__(self, n_inputs, n_neurons):
        """
        Initialize the dense layer.

        Args:
            n_inputs: Number of input features
            n_neurons: Number of neurons in the layer
        """
        # Initialize weights with small random values from Gaussian distribution
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
        # Initialize biases as zeros
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


def demonstrate_dense_layer():
    """Demonstrate the Dense Layer class with spiral data."""
    nnfs.init()

    # Create dataset
    X, y = spiral_data(samples=100, classes=3)

    # Create dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)

    # Perform forward pass
    dense1.forward(X)

    print("First 5 samples of layer output:")
    print(dense1.output[:5])

    return dense1


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 3: Adding Layers")
    print("=" * 60)
    print()

    # Multi-layer manual calculation
    print("1. Multi-Layer Manual Calculation")
    print("-" * 60)
    multi_layer_manual()
    print()

    # Generate and visualize spiral data
    print("2. Spiral Data Generation")
    print("-" * 60)
    print("Generating spiral data and visualizing...")
    X, y = generate_spiral_data()
    print(f"Generated {len(X)} samples with {len(set(y))} classes")
    print()

    # Demonstrate Dense Layer class
    print("3. Dense Layer Class")
    print("-" * 60)
    dense1 = demonstrate_dense_layer()
    print()

    print("=" * 60)
    print("Note: nnfs.init() ensures repeatable results by:")
    print("  - Setting random seed to 0")
    print("  - Creating float32 dtype default")
    print("  - Overriding NumPy's dot product")
    print("=" * 60)
