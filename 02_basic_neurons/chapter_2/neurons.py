#!/usr/bin/env python3
"""
Chapter 2: Coding Our First Neurons

This script demonstrates the fundamental concepts of neural network neurons,
including single neuron computation, layers of neurons, and efficient
implementation using NumPy.
"""

import numpy as np


def single_neuron_manual():
    """Demonstrate a single neuron with manual calculation."""
    inputs = [1, 2, 3]
    weights = [0.2, 0.4, 0.6]
    bias = 2

    output = (inputs[0] * weights[0] +
              inputs[1] * weights[1] +
              inputs[2] * weights[2] +
              bias)

    print(f"Single neuron output: {output}")
    return output


def layer_of_neurons_manual():
    """Demonstrate a layer of neurons with manual calculation."""
    inputs = [1, 2, 3, 2.5]
    weights1 = [0.2, 0.4, 0.6, 0.5]
    weights2 = [0.4, -0.5, 0.6, 0.5]
    weights3 = [-0.3, 0.7, 0.1, -0.8]
    bias1 = 2
    bias2 = 3
    bias3 = 0.5

    output = (
        # Neuron 1
        inputs[0] * weights1[0] +
        inputs[1] * weights1[1] +
        inputs[2] * weights1[2] +
        inputs[3] * weights1[3] +
        bias1,
        # Neuron 2
        inputs[0] * weights2[0] +
        inputs[1] * weights2[1] +
        inputs[2] * weights2[2] +
        inputs[3] * weights2[3] +
        bias2,
        # Neuron 3
        inputs[0] * weights3[0] +
        inputs[1] * weights3[1] +
        inputs[2] * weights3[2] +
        inputs[3] * weights3[3] +
        bias3
    )

    print(f"Layer output (manual): {output}")
    return output


def layer_of_neurons_loop():
    """Demonstrate a layer of neurons using loops for better structure."""
    inputs = [1, 2, 3, 2.5]

    # Each inner list represents one neuron's weights (one weight per input)
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]

    # One bias per neuron
    biases = [2, 3, 0.5]

    layer_outputs = []

    # Loop through each neuron (its weights + bias)
    for neuron_weights, neuron_bias in zip(weights, biases):
        neuron_output = 0

        # Multiply each input by its corresponding weight and accumulate
        for n_input, weight in zip(inputs, neuron_weights):
            neuron_output += n_input * weight  # sum(inputs * weights)

        neuron_output += neuron_bias  # Add bias
        layer_outputs.append(neuron_output)  # Store this neuron's output

    print(f"Layer output (loop): {layer_outputs}")
    return layer_outputs


def dot_product_manual():
    """Demonstrate dot product calculation manually."""
    a = [1, 2, 3, 4]
    b = [3, 4, 5, 6]

    dot = sum(x * y for x, y in zip(a, b))
    print(f"Dot product (manual): {dot}")
    return dot


def dot_product_numpy():
    """Demonstrate dot product using NumPy."""
    a = [1, 2, 3, 4]
    b = [3, 4, 5, 6]

    dot = np.dot(a, b)
    print(f"Dot product (NumPy): {dot}")
    return dot


def single_neuron_numpy():
    """Demonstrate a single neuron using NumPy dot product."""
    inputs = [1, 2, 3, 2.5]
    weights = [0.2, 0.8, -0.5, 1]
    bias = 2.0

    output = np.dot(weights, inputs) + bias
    print(f"Single neuron output (NumPy): {output}")
    return output


def layer_of_neurons_numpy():
    """Demonstrate a layer of neurons using NumPy."""
    inputs = [1, 2, 3, 2.5]
    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]

    layer_outputs = np.dot(weights, inputs) + biases
    print(f"Layer output (NumPy): {layer_outputs}")
    return layer_outputs


def batch_processing_numpy():
    """Demonstrate batch processing with NumPy matrix operations."""
    inputs = [[1.0, 2.0, 3.0, 2.5],
              [2.0, 5.0, -1.0, 2.0],
              [-1.5, 2.7, 3.3, -0.8]]

    weights = [[0.2, 0.8, -0.5, 1],
               [0.5, -0.91, 0.26, -0.5],
               [-0.26, -0.27, 0.17, 0.87]]
    biases = [2.0, 3.0, 0.5]

    layer_outputs = np.dot(inputs, np.array(weights).T) + biases
    print(f"Batch processing output:\n{layer_outputs}")
    return layer_outputs


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 2: Coding Our First Neurons")
    print("=" * 60)
    print()

    # Single neuron example
    print("1. Single Neuron (Manual)")
    print("-" * 60)
    single_neuron_manual()
    print()

    # Layer of neurons - manual calculation
    print("2. Layer of Neurons (Manual Calculation)")
    print("-" * 60)
    layer_of_neurons_manual()
    print()

    # Layer of neurons - loop implementation
    print("3. Layer of Neurons (Loop Implementation)")
    print("-" * 60)
    layer_of_neurons_loop()
    print()

    # Dot product - manual
    print("4. Dot Product (Manual)")
    print("-" * 60)
    dot_product_manual()
    print()

    # Dot product - NumPy
    print("5. Dot Product (NumPy)")
    print("-" * 60)
    dot_product_numpy()
    print()

    # Single neuron - NumPy
    print("6. Single Neuron (NumPy)")
    print("-" * 60)
    single_neuron_numpy()
    print()

    # Layer of neurons - NumPy
    print("7. Layer of Neurons (NumPy)")
    print("-" * 60)
    layer_of_neurons_numpy()
    print()

    # Batch processing
    print("8. Batch Processing (NumPy)")
    print("-" * 60)
    batch_processing_numpy()
    print()

    print("=" * 60)
    print("Note: The zip() function lets us iterate over multiple")
    print("iterables (lists in this case) simultaneously.")
    print("=" * 60)
