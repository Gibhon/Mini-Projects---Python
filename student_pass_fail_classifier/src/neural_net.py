import numpy as np

class Layer:
    def __init__(self, n_inputs: int, n_neurons: int) -> None:
        rng = np.random.default_rng()

        self.weights = rng.random((n_inputs, n_neurons)) * 0.01
        self.biases = np.zeros((n_neurons, ))
        self.inputs = None

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.inputs = inputs
        return (inputs @ self.weights + self.biases)
    
    def backward(self, grad_output : np.ndarray) -> np.ndarray:
        grad_weights = self.inputs.T @ grad_output
        grad_biases = np.sum(grad_output, axis = 0)
        self.grad_weights = grad_weights
        self.grad_biases = grad_biases
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


def binary_cross_entropy(y_true : np.ndarray, y_pred : np.ndarray) -> float:
    y_pred = np.clip(y_pred, a_min = 1e-7 ,a_max = 1 - 1e-7)
    return -np.mean( y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred) )


def train(X : np.ndarray, y : np.ndarray, epochs : int, lr : float) -> None:
    layer1 = Layer(2, 4)
    layer2 = Layer(4, 1)
    for epoch in range(epochs):
        layer1_output = layer1.forward(X)
        input_relu = relu(layer1_output)
        
        layer2_output = layer2.forward(input_relu)
        sigmoid_values = sigmoid(layer2_output)
        loss = binary_cross_entropy(y.reshape(-1, 1), sigmoid_values)
        # print("pred:", sigmoid_values.flatten(), "loss:", loss)
        grad = (sigmoid_values - y.reshape(-1, 1)) / len(y)
        backward_layer2 = layer2.backward(grad)
        grad_relu = backward_layer2 * (layer1_output > 0)
        backward_layer1 = layer1.backward(grad_relu)
        layer1.weights -= lr * layer1.grad_weights
        layer1.biases  -= lr * layer1.grad_biases
        layer2.weights -= lr * layer2.grad_weights
        layer2.biases  -= lr * layer2.grad_biases
        if(epoch % 200 == 0):
            print(loss)
    
        
        
        
        

X = np.array([[2.0, 6.0], [8.0, 7.0], [1.0, 5.0], [9.0, 8.0]])
y = np.array([0.0, 1.0, 0.0, 1.0])
train(X, y, epochs=10000, lr=0.5)