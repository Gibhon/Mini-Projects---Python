import numpy as np

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        rng = np.random.default_rng()

        self.weights = rng.random((n_inputs, n_neurons))
        self.biases = np.zeros((n_neurons, ))
        self.inputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return (inputs @ self.weights + self.biases)
    
    def backward(self, grad_output : np.ndarray) -> np.ndarray:
        grad_weights = self.inputs.T @ grad_output
        grad_biases = np.sum(grad_output, axis = 0)
        grad_input = grad_output @ self.weights.T
        return grad_input
    

def relu(x: np.ndarray) -> np.ndarray:
    return (np.maximum(x, 0))

def sigmoid(x : np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-x)))

def forward_pass(inputs : np.ndarray) -> np.ndarray:
    layer1 = Layer(2, 4)
    inputs = layer1.forward(inputs)
    inputs = relu(inputs)
    
    layer2 = Layer(4, 1)
    inputs = layer2.forward(inputs)
    inputs = sigmoid(inputs)
    
    return inputs

y_true = np.array([1.0, 0.0])
y_pred = np.array([0.9, 0.1])  # good predictions
# Should return a small number close to 0

def binary_cross_entropy(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    y_pred = np.clip(y_pred, a_min = 1e-7 ,a_max = 1 - 1e-7)
    return -np.mean( y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) )

