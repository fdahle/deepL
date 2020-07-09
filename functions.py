import numpy as np
import pandas as pd
from numba import jit

import time

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
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
def activation_func(Z, type, direction, dA=0, Y=0):

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
        if direction == "backward":
            temp = np.zeros((Y.shape[1],(int(np.amax(Y)) + 1)))
            for i, elem in enumerate(Y.T):
                temp[i, int(elem)] = 1
            output = dA - temp.T
    return output

#execute the forward propagation
def forward_prop(input):
    #in this function the value of the nodes are calculated

    X = input[0]
    params = input[1]
    activation = input[2]

    #save intermediate steps in this dict
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
    return [A_curr, cache]

#execute backwars propagation
def backward_prop(input):

    AL = input[0]
    Y = input[1]
    params = input[2]
    activation = input[3]
    cache = input[4]
    numClasses = input[5]
    costType = input[6]
    costParams = input[7]

    #this dictionary will save the gradients
    grads = {}

    #get number of entries
    m = Y.shape[1]

    #assure that Y has the same shape as AL
    #Y = Y.reshape(AL.shape)

    #different creation of base gradient based on number of classes
    if costType == "crossEntropy":
        if numClasses == 1:
            #use derivative of cross Entroy
            dA_prev = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        else:
            dA_prev = AL
            dA_prev[range(m),Y] -= 1
            dA_prev = dA_prev/1

    if costType == "focalLoss":

        #check if loss params are defined
        if len(costParams) == 0:
            gamma = 2.0
            alpha = 0.25
        else:
            gamma = costParams[0]
            alpha = costParams[1]

        if numClasses == 1:

            pt_0 = np.where(np.equal(Y, 0), AL, np.zeros_like(AL))
            pt_1 = np.where(np.equal(Y, 1), AL, np.ones_like(AL))

            dA_prev = Y * np.power(1. - pt_1, gamma) * (Y * np.log(pt_1) + pt_0 - 1)


    #iterate through the layers from right to left
    for i, layer in reversed(list(enumerate(activation))):

        #increase id for 1 as weights and bias are always one step ahead and 0 is not available
        idx = i + 1

        dA_curr = dA_prev

        #get the values from the cache and params
        A_prev = cache["A" + str(i)]
        Z_curr = cache["Z" + str(idx)]
        W_curr = params["W" + str(idx)]
        b_curr = params["b" + str(idx)]

        #calculate all gradient
        dZ_curr = activation_func(Z_curr, layer, "backward", dA_curr, Y)
        dW_curr = np.dot(dZ_curr, A_prev.T) / m
        db_curr = np.sum(dZ_curr, axis=1, keepdims=True) / m
        dA_prev = np.dot(W_curr.T, dZ_curr)

        #save the gradient of input, weight and bias
        grads["dA" + str(i)] = dA_prev
        grads["dW" + str(idx)] = dW_curr
        grads["db" + str(idx)] = db_curr

    return [grads]

#compute cost for current model
def compute_cost(input):

    AL = input[0]
    Y = input[1]
    numClasses = input[2]
    costType = input[3]
    epsilon=input[4]
    costParams=input[5]

    #get number of entries
    m = AL.shape[1]

    #use a super tiny value to prevent log(0) lead to -inf and consequently NaN for the cost
    if epsilon:
        e = np.nextafter(0, 1)
    else:
        e = 0

    if costType == "crossEntropy": #aka log loss in ml
        #dependent on number of classes cost is calculated different
        if numClasses == 1:
            #binary cross entropy
            cost = -1 / m * (np.dot(Y, np.log(AL + e).T) + np.dot(1 - Y, np.log(1 - AL + e).T))
            cost = cost[0][0] # get single value from cost
        else:
            #cross entropy
            cost = log_loss(Y.flatten(), AL.T)

    elif costType == "focalLoss":

        #check if loss params are defined
        if len(costParams) == 0:
            gamma = 2.0
            alpha = 0.25
        else:
            gamma = costParams[0]
            alpha = costParams[1]

        if numClasses == 1:

            pt_0 = np.where(np.equal(Y, 0), AL, np.zeros_like(AL))
            pt_1 = np.where(np.equal(Y, 1), AL, np.ones_like(AL))


            #cost = -np.sum(alpha * np.power(1. - pt_1, gamma) * np.log(pt_1))-np.sum((1-alpha) * np.power( pt_0, gamma) * np.log(1. - pt_0))

            cost = -np.mean(alpha * np.power(1. - pt_1, gamma) * np.log(pt_1))-np.sum((1-alpha) * np.power( pt_0, gamma) * np.log(1. - pt_0))


    else:
        raise Error("Cost function not supported")

    #convert np float to regular float
    cost = float(cost)

    return [cost]

#compute accuracy during training
def compute_scores(input):

    types = input[0]
    AL = input[1]
    Y = input[2]
    numClasses = input[3]
    threshold = input[4]

    #dependent on number of classes accuracy is calculated different
    if numClasses == 1:
        #convert prob to classes
        Y_pred = np.zeros(AL.shape)
        Y_pred[AL >= threshold] = 1
    else:
        Y_pred = np.argmax(AL, axis=0)
        Y_pred = np.reshape(Y_pred, (1, -1))

    scores = []

    #check which value should be calculated
    for elem in types:
        if elem == "accuracy":
            #calc score
            score = accuracy_score(Y.flatten(), Y_pred.flatten())
        if elem == "pr":
            score = recall_score(Y.flatten(), Y_pred.flatten())

        #convert np float to regular float
        score = float(score)
        scores.append(score)

    return [scores]

#udpdate weight and bias
def update_params(input):

    params = input[0]
    activation = input[1]
    grads = input[2]
    learning_rate = input[3]

    for i, layer in enumerate(activation):
        idx = i + 1
        params["W" + str(idx)] -= learning_rate * grads["dW" + str(idx)]
        params["b" + str(idx)] -= learning_rate * grads["db" + str(idx)]

    return [params]

#create batches
def createBatches(data, batch_size):

    #no batches should be created
    if batch_size == 0:
        return [data]

    #no need to split if batch size exceeds data size
    if batch_size >= data.shape[1]:
        return[data]

    #get number of batches
    num_batches = int(data.shape[1] / batch_size)

    #split
    batches = np.array_split(data, num_batches, axis=1)

    return batches

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

    #give warning if hidden layers are bigger than input layers
    if layer_dims[0] > X.shape[1]:
        print("Warning: Hidden layer is bigger than input layer")

    if len(layer_activation) != len(layer_dims):
        raise ValueError("Layer dimensions (layer_dims) and activationtype (layer_activation) must have the same size")

    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y have a different number of entries")

    #check for NaN
    if pd.isnull(X).any():
        raise ValueError("The training data contains NaN values. Please clean the data")
        #replace Nan with 0 so that the string test can work

    #check for strings
    temp = pd.DataFrame(X.flatten())
    temp = temp.apply(lambda x: pd.to_numeric(x, errors='coerce')) #convert to int, so strings will be nan
    if pd.isnull(temp).values.any():
        raise ValueError("The training data contains string values. Please clean the data")

def debug(debugMode, func, args):

    if debugMode:
        start = time.time()
        output = func(args)
        end = time.time()
        dura = str(round(end - start, 4)) + "s"
        print("     " + func.__name__ + " finished (" + dura + ")")
    else:
        output = func(args)

    return output
