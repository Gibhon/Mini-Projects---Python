import numpy as np

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        rng = np.random.default_rng()

        self.weights = rng.random((n_inputs, n_neurons))
        self.biases = np.zeros((n_neurons, ))

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        return (inputs @ self.weights + self.biases)
    

def relu(x: np.ndarray) -> np.ndarray:
    return (np.maximum(x, 0))

def sigmoid(x : np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-x)))

# TASK
# Build a complete forward pass for a 2-layer neural network
#
# Function: forward_pass(inputs: np.ndarray) -> np.ndarray
#
# Architecture:
#   - Layer 1: 2 inputs, 4 neurons, relu activation
#   - Layer 2: 4 inputs, 1 neuron, sigmoid activation
#
# Use your Layer class and activation functions above.
# Return the final output.
#
# Test it with:
# inputs = np.array([[2.0, 6.0],   # student: 2hrs study, 6hrs sleep
#                    [8.0, 7.0]])   # student: 8hrs study, 7hrs sleep
# Output shape should be (2, 1)


# def forward_pass(inputs : np.ndarray) -> np.ndarray:
#     rng = np.random.default_rng()
    
#     #Layer 1
#     weights_1 = rng.random((2, 4))
#     biases_1 = np.zeros((4,))
#     inputs = relu(inputs)
#     layer1_output : np.ndarray = (inputs @ weights_1) + biases_1
    
#     #layer 2
#     weights_2 = rng.random((4, 1))
#     baises_2 = np.zeros(1, )
#     inputs = sigmoid(inputs)
    
#     return (inputs @ weights_2) + baises_2
#I might be retarded


def forward_pass(inputs : np.ndarray) -> np.ndarray:
    layer1 = Layer(2, 4)
    inputs = layer1.forward(inputs)
    inputs = relu(inputs)
    
    layer2 = Layer(4, 1)
    inputs = layer2.forward(inputs)
    inputs = sigmoid(inputs)
    
    return inputs

inputs = np.array([[2.0, 6.0],  
                   [8.0, 7.0]])

print(forward_pass(inputs))