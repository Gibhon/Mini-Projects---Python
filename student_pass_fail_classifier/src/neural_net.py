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

# TASK
# Loss function — Binary Cross Entropy
#
# Function: binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float
#
# Formula:
#   -mean( y_true * log(y_pred) + (1 - y_true) * log(1 - y_pred) )
#
# y_true: actual labels, 0 or 1  e.g. np.array([1, 0])
# y_pred: network output (sigmoid), values between 0 and 1
#
# One problem: log(0) is -infinity. Add a small clip to y_pred
# to keep values in range [1e-7, 1 - 1e-7] before taking log.
#
# Return a single float — the mean loss.
#
# Test it with:
y_true = np.array([1.0, 0.0])
y_pred = np.array([0.9, 0.1])  # good predictions
# Should return a small number close to 0

def binary_cross_entropy(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    y_pred = np.clip(y_pred, a_min = 1e-7 ,a_max = 1 - 1e-7)
    return -np.mean( y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) )

print(binary_cross_entropy(y_true, y_pred))