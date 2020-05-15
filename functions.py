import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
from utils import *

class DeepLearner:

    def __init__(self, type='classic', dataType="data"):

        #save basic params
        self.type = type #"classic", "cnn"
        self.dataType = dataType #"data", "images"

        #save params of the model
        self.parameters = None
        self.costs = None

        #save training params
        self.layer_dims = None
        self.learning_rate = None
        self.num_iterations = None
        self.early_stopping = None

        #save input params
        self.num_trainingData = None

        self.imageSizeX = None
        self.imageSizeY = None
        self.imageBands = None

        #save random seed
        self.random_seed = None

    def getParams(self):
        params = {'w': w, 'b': b}
        return params

    def getGrads(self):
        grads = {'dw': dw, 'db': db}
        return grads

    def getCosts(self):
        return costs

    def setTrainingParams(self, layer_dims, num_iterations=None, learning_rate=None, early_stopping=None, loss=None):

        if layer_dims is not None:
            self.layer_dims = layer_dims
        if num_iterations is not None:
            self.num_iterations = num_iterations
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if early_stopping is not None:
            self.early_stopping = early_stopping
        if loss is not None:
            self.loss = loss

    def optimize(self, X_input, Y_input, print_cost=False, print_iter = 100, random_seed=1, silent=True):

        #copy in order not to change the original data
        X = np.asarray(X_input).copy()
        Y = np.asarray(Y_input).copy()

        np.random.seed(random_seed)
        self.random_seed = random_seed
        self.num_trainingData = X.shape[0]

        #check training_params
        assert self.layer_dims is not None, "layer_dims is not initialized"
        assert self.num_iterations is not None, "num_iterations is not initialized"
        assert self.learning_rate is not None, "learning_rate is not initialized"

        if self.dataType =="image":
            self.imageSizeX = X.shape[1]
            self.imageSizeY = X.shape[2]
            self.imageBands = X.shape[3]

        X = reshape(X, self.dataType)
        if (len(Y.shape) == 1):
            Y = reshape(Y, self.dataType)

        if (hasNaN(X)):
            print("X has NaN-values. These are replaced with 0")
            X[np.isnan(X)] = 0

        layer_dims = self.layer_dims
        layer_dims.insert(0, X.shape[0])
        layer_dims.append(1)


        #get number of layers
        self.num_layers = len(layer_dims)

        #init weight and bias
        self.parameters = initialize_params(layer_dims)

        #reset costs
        self.costs = []

        #set params for early stopping
        maxCost = sys.maxsize
        stopCounter = 0
        bestParams = []
        bestCosts = []

        #train
        for i in range(self.num_iterations):

            #forward propagation
            AL, caches = L_model_forward(X, self.parameters)

            #compute cost
            cost = compute_cost(AL, Y, "crossEntropy")

            #if new lowest cost appeared
            if cost < maxCost:
                maxCost = cost
                stopCounter = 0
                bestParams = self.parameters
            stopCounter += 1

            #backwards propagation
            grads = L_model_backward(AL, Y, caches)

            #update parameters
            self.parameters = update_parameters(self.parameters, grads, self.learning_rate)

            #record costs
            self.costs.append(cost)

            #check if the last n rounds didn't improve
            if self.early_stopping is not None:
                if stopCounter == self.early_stopping:
                    self.costs = self.costs[:len(self.costs)-self.early_stopping]
                    self.parameters = bestParams
                    print(len(self.costs))
                    print('Early stopping after iteration %i: %f' % (i-self.early_stopping+1, maxCost))
                    break

            if print_cost and i%print_iter == 0:
                print('Cost after iteration %i: %f' % (i, cost))

        print('Final cost after iteration %i: %f' % (i, cost))

    def predictProba(self, X_input):

        X = np.asarray(X_input).copy()

        ## TODO: check if data is prepared or not and only do if not prepared
        X = reshape(X,"data")

        m = X.shape[1]

        #compute prob vector
        n = self.num_layers // 2
        p = np.zeros((1,m))

        probas, caches = L_model_forward(X, self.parameters)

        if (np.isnan(np.sum(probas))):
            print("NaN-values replaced with 0")
            probas[np.isnan(probas)] = 0

        return(probas[0])

    def predict(self, X_input, threshold=0.5):

        vec =self.predictProba(X_input)
        vec = vec[np.newaxis]
        y_pred = np.zeros((1, vec.shape[1]))

        #TODO: optimze this loop")
        for i in range(vec.shape[1]):
            y_pred[0,i] = 1 if vec[0, i] > threshold else 0

        return y_pred[0]

    def plotCostFunction(self):
        _costs = np.squeeze(self.costs)
        plt.plot(_costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (in 100)')
        plt.title('Learning rate =' + str(self.learning_rate))
        plt.show()

    def saveModel(self, filename, path=os.getcwd(), type="pickle"):

        modelDict = {}

        modelDict["parameters"] = self.parameters
        modelDict["costs"] = self.costs

        modelDict["numLayers"] = self.num_layers
        modelDict["learning_rate"] = self.learning_rate
        modelDict["num_iterations"] = self.num_iterations
        modelDict["num_trainingData"] = self.num_trainingData

        modelDict["imageSizeX"] = self.imageSizeX
        modelDict["imageSizeY"] = self.imageSizeY
        modelDict["imageBands"] = self.imageBands

        modelDict["random_seed"] = self.random_seed

        # TODO: if filename is not valid (.pkl) for example

        if type == "pickle":
            if (filename.endswith(".pkl")):
                filename = filename[:-4]
            f = open(path + "/" + filename + ".pkl", "wb")
            pickle.dump(modelDict,f)
            f.close()

        elif type == "txt":
            # TODO:
            pass
        elif type == "json":
            # TODO:
            pass
        elif type == "csv":
            # TODO:
            pass

    def loadModel(self, path):

        type = path[-3:]

        # TODO: do other types
        if type == "pkl":
            file = open(path, 'rb')
            modelDict = pickle.load(file)
            file.close()

        self.parameters = modelDict["parameters"]
        self.costs = modelDict["costs"]

        self.num_layers = modelDict["numLayers"]
        self.learning_rate = modelDict["learning_rate"]
        self.iterations = modelDict["num_iterations"]
        self.num_trainingData = modelDict["num_trainingData"]

        self.imageSizeX = modelDict["imageSizeX"]
        self.imageSizeY = modelDict["imageSizeY"]
        self.imageBands = modelDict["imageBands"]

        self.random_seed = modelDict["random_seed"]
