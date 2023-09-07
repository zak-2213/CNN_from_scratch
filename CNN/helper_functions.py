import numpy as np

# activation functions
 
def relu(x):
    return np.maximum(0, x)

def softmax_activation(x, epsilon=1e-8):
    t = np.exp(x)
    return t / (np.sum(t) + epsilon)

# cost functions

def cost_function(labels, preds):
    preds = np.clip(preds, 1e-10, 1.0)

    cost = -np.sum(labels * np.log(preds))
    return cost

def cost_backprop(label, pred):
    dZ = pred - label
    return dZ

# flat

def flattening(prev_layer):
    flattened = prev_layer.reshape(-1)
    flattened = np.reshape(flattened, (flattened.shape[0], 1))
    return flattened

def unflatten(prev_layer, new_shape):
    unflattened = np.reshape(prev_layer, new_shape)
    return unflattened

# class error
def compute_error(Y, predictions):
    correct_preds = np.array(list(1.0 * (predictions[i] == np.max(predictions[i])) for i in range(len(predictions))))
    error = np.sum(np.abs(correct_preds - Y)) / len(Y) / 2.0  
    
    return error

# output log

def log(phase, cost, error):
    with open("output.txt", "a") as file:
        file.write(f"{phase} done. Cost: {cost:0.4f} Error: {error:0.4f}\n")
        file.write('\n')