import numpy as np

# TASK
# Layer class with two methods:
# 1. __init__(n_inputs, n_neurons)
#    - self.weights: shape (n_inputs, n_neurons), small random values
#    - self.biases: shape (n_neurons,), zeros
# 2. forward(inputs)
#    - return inputs @ self.weights + self.biases

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        rng = np.random.default_rng()

        self.weights = rng.random((n_inputs, n_neurons))
        self.biases = np.zeros((n_neurons, ))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs @ self.weights + self.biases)


# TASK
# Activation functions — two of them
#
# 1. relu(x: np.ndarray) -> np.ndarray
#    - anything negative becomes 0, positive stays the same
#
# 2. sigmoid(x: np.ndarray) -> np.ndarray
#    - squishes any value into range (0, 1)
#    - formula: 1 / (1 + e^(-x))
#
# No loops. Pure numpy operations.

def relu(x: np.ndarray) -> np.ndarray:
    return (np.maximum(x, 0))

def sigmoid(x : np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-x)))