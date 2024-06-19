import numpy as np

class ActivationFunction:
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(z.dtype)

    @staticmethod
    def linear(z):
        return z
    
    @staticmethod
    def linear_derivative(z):
        return np.ones_like(z)
    
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))
    
    @staticmethod
    def sigmoid_derivative(z):
        sig = ActivationFunction.sigmoid(z)
        return sig * (1 - sig)
    
    @staticmethod
    def softmax(z):
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)
    
    @staticmethod
    def softmax_derivative(z):
        s = ActivationFunction.softmax(z)
        jacobian_m = np.zeros((z.shape[0], z.shape[1], z.shape[1]))
        for i in range(z.shape[0]):
            for j in range(z.shape[1]):
                for k in range(z.shape[1]):
                    if j == k:
                        jacobian_m[i, j, k] = s[i, j] * (1 - s[i, j])
                    else:
                        jacobian_m[i, j, k] = -s[i, j] * s[i, k]
        return jacobian_m
