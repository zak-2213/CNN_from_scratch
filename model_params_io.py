import pickle

def store_params(filename, model_params):
    with open(filename, 'wb') as file:
        pickle.dump(model_params, file)

def read_params(filename):
    with open(filename, 'rb') as file:
        loaded_params = pickle.load(file)

        W = loaded_params['weights']
        b = loaded_params['biases']

    return W, b