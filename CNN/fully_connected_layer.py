import numpy as np
from . import helper_functions as hf

# fully connected layer
class FullyConnected:
    def __init__(self, W, b, size, activation, learning_rate=0.01):
        self.W = W
        self.b = b
        self.s = size
        self.activation = activation
        self.lr = learning_rate

    def get_W(self):
        return self.W
    
    def get_b(self):
        return self.b

    # n_p: size of previous layer
    def init_params_fc(self, n_p):
            self.W = np.random.randn(self.s, n_p) * 0.01
            self.b = np.zeros((self.s, 1))

    def forward(self, prev_layer):
        self.input = prev_layer

        if (self.input.shape[1] != 1):
                self.input = hf.flattening(self.input)

        if (self.W is None and self.b is None):
            self.init_params_fc(self.input.shape[0])
        
        self.Z = np.zeros((self.s, 1))
        self.Z = np.dot(self.W, self.input) + self.b

        if self.activation == "relu":
            self.A = hf.relu(self.Z)
        elif self.activation == "softmax":
            self.A = hf.softmax_activation(self.Z)
            
        return self.A

    def backprop(self ,dA):
        if (self.activation == "relu"):
            dZ = dA * (self.Z > 0.0)
        elif (self.activation == "softmax"):
            dZ = dA / (1.0 + np.exp(-self.Z)) * (1.0 - (1.0 / (1.0 + np.exp(-self.Z))))

        dW = np.dot(self.input, dZ.T).T

        db = np.sum(dZ, axis=1)
        db = np.reshape(db, (db.shape[0], 1))

        dA_prev = np.dot(dZ.T, self.W).T

        self.W -= self.lr * dW
        self.b -= self.lr * db

        return dA_prev, self.W, self.b

    