import numpy as np

class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        (m, h, w, c) = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        out = np.zeros((m, out_h, out_w, c))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                out[:, i, j, :] = np.max(x[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
        
        self.out = out
        return out

    def backward(self, d_out):
        (m, h, w, c) = self.x.shape
        d_x = np.zeros_like(self.x)

        out_h, out_w = d_out.shape[1:3]
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                x_slice = self.x[:, h_start:h_end, w_start:w_end, :]
                mask = (x_slice == np.max(x_slice, axis=(1, 2), keepdims=True))
                d_x[:, h_start:h_end, w_start:w_end, :] += mask * d_out[:, i, j, :][:, np.newaxis, np.newaxis, :]
        
        return d_x

class AvgPoolingLayer:
    def __init__(self, pool_size, stride):
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        self.x = x
        (m, h, w, c) = x.shape
        out_h = (h - self.pool_size) // self.stride + 1
        out_w = (w - self.pool_size) // self.stride + 1
        out = np.zeros((m, out_h, out_w, c))

        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size
                out[:, i, j, :] = np.mean(x[:, h_start:h_end, w_start:w_end, :], axis=(1, 2))
        
        self.out = out
        return out

    def backward(self, d_out):
        (m, h, w, c) = self.x.shape
        d_x = np.zeros_like(self.x)

        out_h, out_w = d_out.shape[1:3]
        
        for i in range(out_h):
            for j in range(out_w):
                h_start = i * self.stride
                h_end = h_start + self.pool_size
                w_start = j * self.stride
                w_end = w_start + self.pool_size

                d_x[:, h_start:h_end, w_start:w_end, :] += d_out[:, i, j, :][:, np.newaxis, np.newaxis, :] / (self.pool_size * self.pool_size)
        
        return d_x