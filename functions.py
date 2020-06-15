import numpy as np
from numba import jit

from sklearn.metrics import accuracy_score
from sklearn.metrics import log_loss

#add input and output layer to layerDims
def adapt_layer(layer_dims, activation, inputSize, numClasses):

    #add input layer
    layer_dims.insert(0, inputSize)

    #add output layers
    layer_dims.append(numClasses)

    #output layer is dependent on number of classes
    if numClasses == 1:
        activation.append("sigmoid")
    else:
        activation.append("softmax")

    return layer_dims

#initalize weight and bias based on the layer of the network
def init_params(layer_dims):

    #weight and bias is stored as a dict
    parameters = {}

    #get the number of layers
    L = len(layer_dims)

    #iterate all layers
    for i in range(L-1):
        idx = i + 1
        parameters['W' + str(idx)] = np.random.randn(layer_dims[idx], layer_dims[idx-1]) * 0.1
        parameters['b' + str(idx)] = np.random.randn(layer_dims[idx], 1) * 0.1

    return parameters

#defines the output of a certain neuron based on type
def activation_func(Z, type, direction, dA=0):

    #prevents an overflow of the values (precision too high)
    Z = np.clip(Z, -500,500)

    if type == "sigmoid":
        if direction == "forward":
            output = 1/(1+np.exp(-Z))
        if direction == "backward":
            sig = activation_func(Z, "sigmoid", "forward")
            output = dA * sig * (1 - sig)

    if type == "relu":
        if direction == "forward":
            output = np.maximum(0,Z)
        if direction == "backward":
            output = np.array(dA, copy = True)
            output[Z <= 0] = 0

    if type == "softmax":
        if direction == "forward":
            shiftz = Z - np.max(Z) # this makes softmax more stable
            temp = np.exp(Z)
            output = temp/np.sum(temp, axis = 0)
            print("TODO CHECK AXIS 0")

        if direction == "backward":
            shiftz = dA - np.max(dA) # this makes softmax more stable
            t_exp = np.exp(shiftz)
            sum = np.sum(t_exp)

            output = -t_exp * t_exp / (sum ** 2)

            for i, elem in enumerate(dA):
                idx = np.squeeze(np.nonzero(elem))
                output[i][idx] = t_exp[i][idx] * (sum - t_exp[i][idx]) / (sum ** 2)

    return output

#execute the forward propagation
def forward_prop(X, params, activation):

    #in this function the value of the nodes are calculated

    cache={}

    #copy first input to A (which is used in the loop to contain the data and is
    #renewed every iteration)
    A_curr = X

    #iterate through the layers from left to right
    for i, layer in enumerate(activation):
        #increase id for 1 as weights and bias are always one step ahead and 0 is not available
        idx = i + 1
        A_prev = A_curr

        #get weights and bias
        W_curr = params["W" + str(idx)]
        b_curr = params["b" + str(idx)]

        #linear forward
        Z_curr = np.dot(W_curr, A_prev) + b_curr

        #apply activation func
        A_curr = activation_func(Z_curr, layer, "forward")

        #save intermedia results in cache
        cache["A" + str(i)] = A_prev  #<- important here not idx but i
        cache["Z" + str(idx)] = Z_curr

    #A_curr is output at the end nodes
    #cache contains all intermediate outputs; A[n] contains the input, Z[n+1] the output of this input
    return A_curr, cache

#execute backwars propagation
def backward_prop(AL, Y, params, activation, cache, numClasses):

    #this dictionary will save the gradients
    grads = {}

    #get number of entries
    m = Y.shape[1]

    #assure that Y has the same shape as AL
    #Y = Y.reshape(AL.shape)

    print(AL)

    #different creation of base gradient based on number of classes
    if numClasses == 1:
        dA_prev = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    else:
        dA_prev = np.zeros((m, numClasses))
        for i in range(Y.shape[0]):
            dA_prev[i, Y[i]] = -1 / AL[i, Y[i]]

    #iterate through the layers from right to left
    for i, layer in reversed(list(enumerate(activation))):

        #increase id for 1 as weights and bias are always one step ahead and 0 is not available
        idx = i + 1

        dA_curr = dA_prev

        #get the values from the cache and params
        A_prev = cache["A" + str(i)] #<- important here not idx but i
        Z_curr = cache["Z" + str(idx)]
        W_curr = params["W" + str(idx)]
        b_curr = params["b" + str(idx)]

        #calculate all gradient
        dZ_curr = activation_func(Z_curr, layer, "backward", dA_curr)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        #save the gradient of input, weight and bias
        grads["dA" + str(i)] = dA_prev
        grads["dW" + str(idx)] = dW_curr
        grads["db" + str(idx)] = db_curr

    return grads

#compute cost for current model
def compute_cost(AL, Y, numClasses, costType):

    #get number of entries
    m = AL.shape[1]
    if costType == "crossEntropy":

        if numClasses == 1:
            cost = -1 / m * (np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T))
        else:
            cost = log_loss(Y.flatten(), AL.T)

    #convert array to single value
    cost = np.squeeze(cost)

    return(cost)

#compute accuracy during training
def compute_accuracy(AL, Y, numClasses, threshold):

    if numClasses == 1:
        #convert prob to classes
        Y_pred = np.zeros(AL.shape)
        Y_pred[AL >= threshold] = 1
    else:
        Y_pred = np.argmax(AL, axis=0)
        Y_pred = np.reshape(Y_pred, (1, -1))

    #calc score
    score = accuracy_score(Y.flatten(), Y_pred.flatten())

    return score

#udpdate weight and bias
def update_params(params, activation, grads, learning_rate):

    for i, layer in enumerate(activation):
        idx = i + 1
        params["W" + str(idx)] -= learning_rate * grads["dW" + str(idx)]
        params["b" + str(idx)] -= learning_rate * grads["db" + str(idx)]
    return params

# reshape data so that it can be used by NN
def reshape(data):
    #(209, 64, 64) -> (4096, 209) for 209 images with 64x64 resolution

    _data = data.reshape(data.shape[0], -1).T

    return _data

#check if a numpy array has NaN values
def hasNaN(data):
    sum = np.sum(data)
    return np.isnan(sum)

#check data and settings for consistency
def checkErrors(layer_activation, layer_dims, X, Y):

    if len(layer_activation) != len(layer_dims):
        raise ValueError("Layer dimensions (layer_dims) and activationtype (layer_activation) must have the same size")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y have a different number of entries")
