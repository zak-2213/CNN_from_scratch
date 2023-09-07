import numpy as np
from . import helper_functions as hf

# max pool layer

class MaxPool:
    def __init__(self, stride=2, filter_size=2):
        self.s = stride
        self.f = filter_size

    # prev_layer: m x n_h_prev x n_w_prev x n_c
    def forward(self, prev_layer):
        self.input = prev_layer
        
        (n_h_prev, n_w_prev, n_c) = self.input.shape

        n_h = int((n_h_prev - self.f) / self.s) + 1
        n_w = int((n_w_prev - self.f) / self.s) + 1
        self.A = np.zeros((n_h, n_w, n_c))

        for h in range(n_h):
             h_start = h * self.s
             for w in range(n_w):
                w_start = w * self.s
                for c in range(n_c):
                    prev_layer_slice = self.input[h_start:h_start+self.f, w_start:w_start+self.f, c]
                    self.A[h, w, c] = np.max(prev_layer_slice)

        return self.A


    # in order to carry out backprop, we need to filter the max values and set the rest to 0
    def max_pool_mask(window):
        mask = window == np.max(window)
        return mask

    # A: activation of pool layer
    # Same stride and f as used in max_pool forward prop
    def backprop(self, dA):
        if (dA.shape[1] == 1):
            dA = hf.unflatten(dA, self.A.shape)

        n_h, n_w, n_c = self.A.shape
        n_h_prev, n_w_prev, _ = self.input.shape

        dA_prev = np.zeros((n_h_prev, n_w_prev, n_c))

        for h in range(n_h):
            h_start = h * self.s
            for w in range(n_w):
                w_start = w * self.s
                for c in range(n_c):
                    a_slice = self.A[h_start:h_start+self.f, w_start:w_start+self.f, c]

                    # Make sure window is of correct size
                    pad_height = max(0, self.f - a_slice.shape[0])
                    pad_width = max(0, self.f - a_slice.shape[1])          
                    a_slice = np.pad(a_slice, ((0, pad_height), (0, pad_width)), mode='constant')

                    mask = self.max_pool_mask(a_slice)
                    
                    dA_prev[h:h+self.f, w:w+self.f, c] += mask * dA[h, w, c]
        
        return dA_prev