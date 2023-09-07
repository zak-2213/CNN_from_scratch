from CNN import helper_functions as hf
import time

def training(x_train, y_train, W, b, start_time, batch_size, network):
    cost_sum = 0
    predictions = []
    
    for i in range(batch_size):
        image = x_train[i, :, :, :]
        label = y_train[i, :]

        prediction = network.cnn_model_forward(image)
        predictions.append(prediction)
        
        cost = hf.cost_function(label, prediction)
        cost_sum += cost

        t = int(time.time() - start_time); 
        t_str = '%2dh %2dm %2ds' % (int(t / 3600), int((t % 3600) / 60), t % 60)

        print(f'\rTime: {t_str}  Sample: {i+1:5}/{batch_size} Cost: {cost:0.4f} ', end='')

    error = hf.compute_error(y_train, predictions)

    print(f'\n Batch complete. Average cost: {cost_sum/batch_size:0.4f} Error: {error:0.4f}')
    
    dA = hf.cost_backprop(label, prediction)
    W, b = network.cnn_model_back(dA)

    return W, b

def validation(x_val, y_val, start_time, validation_network):
    num_samples = x_val.shape[0]

    cost = 0
    predictions = []

    for i in range(num_samples):
        image = x_val[i,:,:,:]
        label = y_val[i,:]

        prediction = validation_network.cnn_model_forward(image)

        predictions.append(prediction)

        cost += hf.cost_function(label, prediction)
        
        t = int(time.time() - start_time); 
        t_str = '%2dh %2dm %2ds' % (int(t / 3600), int((t % 3600) / 60), t % 60)
        print(f'\rTime: {t_str}  Validating: {i+1:5}/{num_samples} ', end='')

    cost /= num_samples
    error = hf.compute_error(y_val, predictions) 

    hf.log("Validation", cost, error)

def testing(x_test, y_test, start_time, test_network):
    num_samples = x_test.shape[0]

    predictions = []
    cost = 0
    for i in range(num_samples):
        image = x_test[i,:,:,:]
        label = y_test[i,:]

        prediction = test_network.cnn_model_forward(image)
        predictions.append(prediction)

        cost += hf.cost_function(label, prediction)
    
        t = int(time.time() - start_time); 
        t_str = '%2dh %2dm %2ds' % (int(t / 3600), int((t % 3600) / 60), t % 60)
        print(f'\rTime: {t_str}  Testing: {i+1:5}/{num_samples} ', end='')

    cost /= num_samples
    error = hf.compute_error(y_test, predictions)  

    print(f"Testing done. Cost: {cost:0.4f} Error: {error:0.4f}")

    hf.log("Testing", cost, error)