import numpy as np
from .activation import ActivationFunction

def he_normal(shape):
    stddev = np.sqrt(2. / shape[0])
    return np.random.randn(*shape) * stddev

class FullyConnectedLayer:
    def __init__(self, input_dim, output_dim, optimizer='adam', init_method='he_normal'):
        self.params = {
            'W': self.initialize_weights((input_dim, output_dim), init_method),
            'b': np.zeros(output_dim)
        }
        self.gradients = {
            'dW': np.zeros_like(self.params['W']),
            'db': np.zeros_like(self.params['b'])
        }
        self.optimizer = optimizer
        if optimizer == 'adam':
            self.m = {key: np.zeros_like(val) for key, val in self.params.items()}
            self.v = {key: np.zeros_like(val) for key, val in self.params.items()}
    
    def initialize_weights(self, shape, init_method):
        if init_method == 'he_normal':
            return he_normal(shape)
        elif init_method == 'zeros':
            return np.zeros(shape)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
    
    def forward(self, x):
        self.x = x
        self.z = np.dot(x, self.params['W']) + self.params['b']
        return self.z
    
    def backward(self, dZ):
        m = self.x.shape[0]
        self.gradients['dW'] = np.dot(self.x.T, dZ) / m
        self.gradients['db'] = np.sum(dZ, axis=0) / m
        dX = np.dot(dZ, self.params['W'].T)
        return dX

    def update_params(self, lr, beta1, beta2, epsilon, t):
        if self.optimizer == 'adam':
            for key in self.params:
                self.m[key] = beta1 * self.m[key] + (1 - beta1) * self.gradients['d' + key]
                self.v[key] = beta2 * self.v[key] + (1 - beta2) * (self.gradients['d' + key] ** 2)
                m_hat = self.m[key] / (1 - beta1 ** t)
                v_hat = self.v[key] / (1 - beta2 ** t)
                self.params[key] -= lr * m_hat / (np.sqrt(v_hat) + epsilon)
        elif self.optimizer == 'sgd':
            for key in self.params:
                self.params[key] -= lr * self.gradients['d' + key]
