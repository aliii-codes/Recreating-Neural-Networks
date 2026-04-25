#!/usr/bin/env python3
"""
Chapter 5: Calculating Network Error with Loss

This script demonstrates loss functions for neural networks,
including categorical cross-entropy loss and accuracy calculation.
"""

import numpy as np
import nnfs
from nnfs.datasets import spiral_data


def categorical_cross_entropy_manual():
    """Demonstrate categorical cross-entropy loss manually."""
    import math

    # Example output from the output layer
    softmax_output = [0.7, 0.1, 0.2]
    # Ground truth
    target_output = [1, 0, 0]

    loss = -(math.log(softmax_output[0]) * target_output[0] +
             math.log(softmax_output[1]) * target_output[1] +
             math.log(softmax_output[2]) * target_output[2])

    print(f"Loss: {loss}")
    return loss


def log_explanation():
    """Demonstrate the log function properties."""
    b = 5.2
    print(f"Natural log of {b}: {np.log(b)}")
    print(f"e^({np.log(b)}): {np.exp(np.log(b))}")


def categorical_cross_entropy_batch():
    """Demonstrate categorical cross-entropy for a batch."""
    import math

    softmax_outputs = [[0.7, 0.1, 0.2],
                       [0.1, 0.5, 0.4],
                       [0.02, 0.9, 0.08]]

    class_targets = [0, 0, 1]

    for targ_idx, distribution in zip(class_targets, softmax_outputs):
        loss = -math.log(distribution[targ_idx])
        print(f"Loss for class {targ_idx}: {loss}")


def categorical_cross_entropy_numpy():
    """Demonstrate categorical cross-entropy using NumPy."""
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])
    class_targets = [0, 1, 2]

    print("Target probabilities:")
    print(softmax_outputs[[0, 1, 2], class_targets])

    print("Using range for indexing:")
    print(softmax_outputs[range(len(softmax_outputs)), class_targets])

    print("Negative log likelihoods:")
    print(-np.log(softmax_outputs[range(len(softmax_outputs)), class_targets]))

    neg_log = -np.log(softmax_outputs[range(len(softmax_outputs)), class_targets])
    avg_loss = np.mean(neg_log)

    print(f"Average loss: {avg_loss}")
    return avg_loss


def categorical_cross_entropy_one_hot():
    """Demonstrate categorical cross-entropy with one-hot encoded targets."""
    softmax_outputs = np.array([[0.7, 0.1, 0.2],
                                [0.1, 0.5, 0.4],
                                [0.02, 0.9, 0.08]])

    class_targets = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 1, 0]])

    # Probabilities for target values - only if categorical labels
    if len(class_targets.shape) == 1:
        correct_confidences = softmax_outputs[
            range(len(softmax_outputs)),
            class_targets
        ]

    # Mask values - only for one-hot encoded labels
    elif len(class_targets.shape) == 2:
        correct_confidences = np.sum(softmax_outputs * class_targets, axis=1)

    neg_log = -np.log(correct_confidences)
    avg_loss = np.mean(neg_log)

    print(f"Average loss (one-hot): {avg_loss}")
    return avg_loss


class Loss:
    """Common loss class."""

    def calculate(self, output, y):
        """
        Calculate the data and regularization losses.

        Args:
            output: Model output
            y: Ground truth values

        Returns:
            Mean loss
        """
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss


class Loss_CategoricalCrossentropy(Loss):
    """Categorical cross-entropy loss."""

    def forward(self, y_pred, y_true):
        """
        Forward pass for categorical cross-entropy loss.

        Args:
            y_pred: Predicted values (softmax output)
            y_true: Ground truth values

        Returns:
            Sample losses
        """
        samples = len(y_pred)

        # Clip data to prevent division by 0
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # Probabilities for target values - only if categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        # Mask values - only for one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)

        # Losses
        neg_log = -np.log(correct_confidences)
        return neg_log


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
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        return self.output


def calculate_accuracy():
    """Demonstrate accuracy calculation."""
    softmax_output = np.array([[0.7, 0.1, 0.2],
                               [0.1, 0.5, 0.4],
                               [0.02, 0.9, 0.08]])

    class_targets = np.array([[1, 0, 0]])

    # Calculate values along second axis (axis of index 1)
    predictions = np.argmax(softmax_output, axis=1)

    # If targets are one-hot encoded, turn them into discrete values
    if len(class_targets.shape) == 2:
        class_targets = np.argmax(class_targets, axis=1)

    # True evaluates to 1; False to 0
    accuracy = np.mean(predictions == class_targets)

    print(f'Accuracy: {accuracy}')
    return accuracy


def demonstrate_full_network_with_loss():
    """Demonstrate full neural network with loss and accuracy."""
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

    # Create loss function
    loss_function = Loss_CategoricalCrossentropy()

    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    # Calculate loss
    loss = loss_function.calculate(activation2.output, y)

    # Calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    print("First 5 softmax outputs:")
    print(activation2.output[:5])
    print(f'Loss: {loss}')
    print(f'Accuracy: {accuracy}')

    return loss, accuracy


if __name__ == "__main__":
    print("=" * 60)
    print("Chapter 5: Calculating Network Error with Loss")
    print("=" * 60)
    print()

    # Categorical cross-entropy manual
    print("1. Categorical Cross-Entropy (Manual)")
    print("-" * 60)
    categorical_cross_entropy_manual()
    print()

    # Log explanation
    print("2. Log Function Properties")
    print("-" * 60)
    log_explanation()
    print()

    # Batch processing
    print("3. Categorical Cross-Entropy (Batch)")
    print("-" * 60)
    categorical_cross_entropy_batch()
    print()

    # NumPy implementation
    print("4. Categorical Cross-Entropy (NumPy)")
    print("-" * 60)
    categorical_cross_entropy_numpy()
    print()

    # One-hot encoded targets
    print("5. Categorical Cross-Entropy (One-Hot)")
    print("-" * 60)
    categorical_cross_entropy_one_hot()
    print()

    # Accuracy calculation
    print("6. Accuracy Calculation")
    print("-" * 60)
    calculate_accuracy()
    print()

    # Full network with loss
    print("7. Full Neural Network with Loss and Accuracy")
    print("-" * 60)
    demonstrate_full_network_with_loss()
    print()

    print("=" * 60)
    print("Note: Clipping prevents division by zero in log calculation")
    print("Accuracy measures how often the largest confidence is correct")
    print("=" * 60)
