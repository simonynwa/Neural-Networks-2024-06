import numpy as np

def he_normal(shape):
    stddev = np.sqrt(2. / np.prod(shape[:-1]))
    return np.random.randn(*shape) * stddev

class Conv2DLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0, optimizer='adam', init_method='he_normal'):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.params = {
            'W': self.initialize_weights((kernel_size, kernel_size, input_channels, output_channels), init_method),
            'b': np.zeros(output_channels)
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
        batch_size, in_height, in_width, _ = x.shape
        out_height = (in_height - self.kernel_size + 2 * self.padding) // self.stride + 1
        out_width = (in_width - self.kernel_size + 2 * self.padding) // self.stride + 1
        
        self.z = np.zeros((batch_size, out_height, out_width, self.output_channels))
        
        if self.padding > 0:
            x = np.pad(x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        
        for i in range(out_height):
            for j in range(out_width):
                x_slice = x[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :]
                for k in range(self.output_channels):
                    self.z[:, i, j, k] = np.sum(x_slice * self.params['W'][:, :, :, k], axis=(1, 2, 3))
        
        self.z += self.params['b']
        return self.z
    
    def backward(self, dZ):
        batch_size, in_height, in_width, _ = self.x.shape
        _, out_height, out_width, _ = dZ.shape
        
        if self.padding > 0:
            x_pad = np.pad(self.x, ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)), mode='constant')
        else:
            x_pad = self.x
        
        self.gradients['dW'] = np.zeros_like(self.params['W'])
        self.gradients['db'] = np.sum(dZ, axis=(0, 1, 2)) / batch_size
        
        dX = np.zeros_like(x_pad)
        
        for i in range(out_height):
            for j in range(out_width):
                x_slice = x_pad[:, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :]
                for k in range(self.output_channels):
                    self.gradients['dW'][:, :, :, k] += np.sum(x_slice * dZ[:, i:i+1, j:j+1, k:k+1], axis=0)
                for n in range(batch_size):
                    dX[n, i*self.stride:i*self.stride+self.kernel_size, j*self.stride:j*self.stride+self.kernel_size, :] += np.sum(
                        self.params['W'][:, :, :, :] * dZ[n, i, j, :], axis=3)
        
        if self.padding > 0:
            dX = dX[:, self.padding:-self.padding, self.padding:-self.padding, :]
        
        self.gradients['dW'] /= batch_size
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
