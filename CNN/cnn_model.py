
class CNN_Model:
    def __init__(self, layers):
        self.layers = layers

    def cnn_model_forward(self, A):

        for l in self.layers:
            A = l.forward(A)
       
        return A

    def cnn_model_back(self, dA):
        weights = []
        biases = []

        for l in reversed(self.layers):
            
            if isinstance(l, max.MaxPool):
                dA = l.backprop(dA)
            else:
                dA, W, b = l.backprop(dA)
                weights.append(W)
                biases.append(b)
        
        return weights[::-1], biases[::-1]
