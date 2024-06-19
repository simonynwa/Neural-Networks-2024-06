import numpy as np
from .fullyconnected import FullyConnectedLayer
from .activation import ActivationFunction
from .conv2d import Conv2DLayer
from .pooling import MaxPoolingLayer, AvgPoolingLayer

class FullyConnected:
    def __init__(self, optimizer='adam', init_method='he_normal'):
        self.layers = [
            FullyConnectedLayer(169, 128, optimizer, init_method),
            FullyConnectedLayer(128, 64, optimizer, init_method),
            FullyConnectedLayer(64, 5, optimizer, init_method)
        ]
        self.t = 0

    def forward(self, x):
        x = ActivationFunction.relu(self.layers[0].forward(x))
        x = ActivationFunction.sigmoid(self.layers[1].forward(x))
        x = self.layers[2].forward(x)
        self.output = ActivationFunction.softmax(x)
        return self.output

    def backward(self, y):
        m = y.shape[0]
        y_one_hot = np.eye(5)[y]
        dZ = self.output - y_one_hot
        
        dZ = self.layers[2].backward(dZ)
        dZ = self.layers[1].backward(dZ * ActivationFunction.sigmoid_derivative(self.layers[1].z))
        dZ = self.layers[0].backward(dZ * ActivationFunction.relu_derivative(self.layers[0].z))
    
    def update_params(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        for layer in self.layers:
            layer.update_params(lr, beta1, beta2, epsilon, self.t)


class Conv:
    def __init__(self, optimizer='adam', init_method='he_normal'):
        self.layers = [
            Conv2DLayer(1, 32, 3, 1, 1, optimizer, init_method),
            Conv2DLayer(32, 64, 3, 1, 1, optimizer, init_method),
            FullyConnectedLayer(64 * 13 * 13, 128, optimizer, init_method),  # Adjust the input size
            FullyConnectedLayer(128, 64, optimizer, init_method),
            FullyConnectedLayer(64, 5, optimizer, init_method)
        ]
        self.t = 0

    def forward(self, x):
        x = ActivationFunction.relu(self.layers[0].forward(x))
        x = ActivationFunction.relu(self.layers[1].forward(x))
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = ActivationFunction.relu(self.layers[2].forward(x))
        x = ActivationFunction.relu(self.layers[3].forward(x))
        x = self.layers[4].forward(x)
        self.output = ActivationFunction.softmax(x)
        return self.output

    def backward(self, y):
        m = y.shape[0]
        y_one_hot = np.eye(5)[y]
        dZ = self.output - y_one_hot
        
        dZ = self.layers[4].backward(dZ)
        dZ = self.layers[3].backward(dZ * ActivationFunction.relu_derivative(self.layers[3].z))
        dZ = self.layers[2].backward(dZ * ActivationFunction.relu_derivative(self.layers[2].z))
        dZ = dZ.reshape(m, 13, 13, 64)  # Reshape
        dZ = self.layers[1].backward(dZ * ActivationFunction.relu_derivative(self.layers[1].z))
        dZ = self.layers[0].backward(dZ * ActivationFunction.relu_derivative(self.layers[0].z))
    
    def update_params(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        for layer in self.layers:
            layer.update_params(lr, beta1, beta2, epsilon, self.t)


class Pooling:
    def __init__(self, optimizer='adam', init_method='he_normal'):
        self.layers = [
            Conv2DLayer(1, 32, 3, 1, 1, optimizer, init_method),
            MaxPoolingLayer(2, 2),
            Conv2DLayer(32, 64, 3, 1, 1, optimizer, init_method),
            MaxPoolingLayer(2, 2),
            FullyConnectedLayer(64 * 3 * 3, 128, optimizer, init_method),  # Adjust the input size
            FullyConnectedLayer(128, 64, optimizer, init_method),
            FullyConnectedLayer(64, 5, optimizer, init_method)
        ]
        self.t = 0

    def forward(self, x):
        x = ActivationFunction.relu(self.layers[0].forward(x))
        x = self.layers[1].forward(x)  # Max Pooling
        x = ActivationFunction.relu(self.layers[2].forward(x))
        x = self.layers[3].forward(x)  # Max Pooling
        x = x.reshape(x.shape[0], -1)  # Flatten
        x = ActivationFunction.relu(self.layers[4].forward(x))
        x = ActivationFunction.relu(self.layers[5].forward(x))
        x = self.layers[6].forward(x)
        self.output = ActivationFunction.softmax(x)
        return self.output

    def backward(self, y):
        m = y.shape[0]
        y_one_hot = np.eye(5)[y]
        dZ = self.output - y_one_hot
        
        dZ = self.layers[6].backward(dZ)
        dZ = self.layers[5].backward(dZ * ActivationFunction.relu_derivative(self.layers[5].z))
        dZ = self.layers[4].backward(dZ * ActivationFunction.relu_derivative(self.layers[4].z))
        dZ = dZ.reshape(m, 3, 3, 64)  # Reshape
        dZ = self.layers[3].backward(dZ)
        dZ = self.layers[2].backward(dZ * ActivationFunction.relu_derivative(self.layers[2].z))
        dZ = self.layers[1].backward(dZ)
        dZ = self.layers[0].backward(dZ * ActivationFunction.relu_derivative(self.layers[0].z))
    
    def update_params(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.t += 1
        for layer in self.layers:
            if isinstance(layer, FullyConnectedLayer) or isinstance(layer, Conv2DLayer):
                layer.update_params(lr, beta1, beta2, epsilon, self.t)