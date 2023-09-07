import numpy as np
from scipy import signal
import helper_functions as hf

class ConvolutionLayer:

    def __init__(self, W, b, num_filters, filter_size, activation="relu", stride=1, padding=0, learning_rate=0.01):
        self.W = W
        self.b = b
        self.n_f = num_filters
        self.f = filter_size
        self.s = stride
        self.p = padding
        self.lr = learning_rate
        self.a = activation


    # n_c: number of channels
    # n_h: height of output
    # n_w: width of output
    def init_params(self, n_c, n_h, n_w):
        filters = np.zeros((self.n_f, self.f, self.f, n_c))
        for i in range(self.n_f):
            filters[i,:,:,:] = np.random.randn(self.f, self.f, n_c) * 0.01
        
        bias = np.zeros((n_h, n_w, self.n_f)) 

        self.W = filters
        self.b = bias

    # pads layer with zeros
    def pad(self, layer):
        pad_width = ((self.p, self.p), (self.p, self.p), (0, 0))
        layer = np.pad(layer, pad_width)
        return layer

    
    # filters: n_f x f x f x n_c
    # prev_layer: n_h_prev x n_w_prev x n_c
    # biases: n_h x n_w x n_f
    # output: m x n_h x n_w x n_f
    def forward(self, prev_layer, W=None, b=None):
        if (W is not None and b is not None):
            self.W = W
            self.b = b
        
        self.input = prev_layer

        (n_h_prev, n_w_prev, n_c) = self.input.shape

        n_h = int((n_h_prev - self.f + 2 * self.p) / self.s) + 1
        n_w = int((n_w_prev - self.f + 2 * self.p) / self.s) + 1

        if (self.W is None and self.b is None):
            self.init_params(n_c, n_h, n_w)

        if (self.p > 0):
            self.input = np.zeros((n_h_prev + 2 * self.p, n_w_prev + 2 * self.p, n_c))
            self.input = self.pad(self, prev_layer)
        
        A = np.zeros((n_h, n_w, self.n_f))
        self.Z = np.zeros((A.shape))

        for i in range(self.n_f):
            for j in range(n_c):
                self.Z[:,:,i] += (signal.correlate2d(self.input[:,:,j], self.W[i,:,:,j], mode='valid') + self.b[:,:,i])

        if (self.a == "relu"):
            A = hf.relu(self.Z)

        A = self.Z

        return A        

    def backprop(self, dA):
        dZ = dA * (self.Z > 0)

        _, _, _, n_c = self.W.shape

        dA_prev = np.zeros((self.input.shape))
        dW = np.zeros((self.W.shape))
        db = dZ

        W_rot = np.rot90(self.W, 2, axes=(1, 2))  

        for i in range(self.n_f):
            for j in range(n_c):
                dW[i,:,:,j] = signal.correlate2d(self.input[:,:,j], dZ[:,:,i], mode='valid')
                dA_prev[:,:,j] = signal.correlate2d(dZ[:,:,i], W_rot[i,:,:,j], mode='full')
        
        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dA_prev, self.W, self.b


