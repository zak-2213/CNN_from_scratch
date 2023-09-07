from Data import data_preprocessing as data
import training_and_testing as tt
import model_params_io as io
import time
from CNN import convolution_layer as conv
from CNN import max_pool_layer as max
from CNN import fully_connected_layer as fc
from CNN import cnn_model as cnn

def construct_network(W, b, learning_rate):
    if (W is None and b is None):
        W1 = None; W2 = None; W3 = None; W4 = None; W5 = None;
        b1 = None; b2 = None; b3 = None; b4 = None; b5 = None;
    else:
        W1, W2, W3, W4, W5 = W
        b1, b2, b3, b4, b5 = b

    conv1 = conv.ConvolutionLayer(W1, b1, 16, 5, learning_rate=learning_rate)
    max1 = max.MaxPool()
    conv2 = conv.ConvolutionLayer(W2, b2, 32, 5, learning_rate=learning_rate)
    max2 = max.MaxPool()
    fc1 = fc.FullyConnected(W3, b3, 128, "relu", learning_rate=learning_rate)
    fc2 = fc.FullyConnected(W4, b4, 84, "relu", learning_rate=learning_rate)
    softmax = fc.FullyConnected(W5, b5, 10, "softmax", learning_rate=learning_rate)
    
    layers = [conv1, max1, conv2, max2, fc1, fc2, softmax]

    network = cnn.CNN_Model(layers)

    return network

def main():
    filename = "trained_model.pkl"
    start_time = time.time()

    x_train, y_train, x_val, y_val, x_test, y_test = data.load_mnist_database()

    W = None
    b = None
    
    num_epochs = 10
    learning_rate = 0.01
    for i in range(num_epochs):
        train_batches = data.batch_generator(x_train, y_train)
        j = 0
        for x, y in train_batches:
            network = construct_network(W, b, learning_rate)

            W, b = tt.training(x, y, W, b, start_time, x.shape[0], network)   

            j+=1
            if (j%10 == 0):
                print(f'{j} batches done.')

        print(f"Epoch {i+1} done")
        learning_rate *= 0.9
        tt.validation(x_val, y_val, start_time, network)

    print("Training complete.")

    tt.testing(x_test, y_test, start_time, network)

    model_params = {
        'weights': W,
        'biases': b
    }
    
    io.store_params(filename, model_params)
    

if __name__ == '__main__':
    main()