import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
from utils import *

class DeepLearner:

    def __init__(self):

        self.parameters = None
        self.costs = None

        self.numLayers = None
        self.learning_rate = None
        self.num_iterations = None
        self.num_trainingData = None

        self.imageSizeX = None
        self.imageSizeY = None
        self.imageBands = None

        self.random_seed = None

    def getParams(self):
        params = {'w': w, 'b': b}
        return params

    def getGrads(self):
        grads = {'dw': dw, 'db': db}
        return grads

    def getCosts(self):
        return costs

    def prepareData(self, data):
        data_fl = data.reshape(data.shape[0], -1).T
        #standardize
        _data = data_fl / 255.
        return _data

    def optimize(self, X_input, Y_input, layer_dims, num_iterations, learning_rate, print_cost=False, print_iter = 100, random_seed=1):

        np.random.seed(random_seed)
        self.random_seed = random_seed

        #save  parameters for later
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.num_trainingData = X_input.shape[0]
        self.imageSizeX = X_input.shape[1]
        self.imageSizeY = X_input.shape[2]
        self.imageBands = X_input.shape[3]

        #define X and y
        X = self.prepareData(X_input)
        Y = Y_input

        layer_dims.insert(0, X.shape[0])
        layer_dims.append(1)

        #get number of layers
        self.num_layers = len(layer_dims)

        #init weight and bias
        self.parameters = initialize_params(layer_dims)

        #reset costs
        self.costs = []

        #train
        for i in range(self.num_iterations):

            #forward propagation
            AL, caches = L_model_forward(X, self.parameters)

            #compute cost
            cost = compute_cost(AL, Y)

            #backwards propagation
            grads = L_model_backward(AL, Y, caches)

            #update parameters
            self.parameters = update_parameters(self.parameters, grads, self.learning_rate)

            #record costs
            if i%print_iter == 0:
                self.costs.append(cost)

            if print_cost and i%print_iter == 0:
                print('Cost after iteration %i: %f' % (i, cost))

    def predict_proba(self, X):

        ## TODO: check if data is prepared or not and only do if not prepared
        X = self.prepareData(X)

        m = X.shape[1]

        #compute prob vector
        n = self.num_layers // 2
        p = np.zeros((1,m))


        probas, caches = L_model_forward(X, self.parameters)

        return(probas)

    def predict(self, X, threshold=0.5):

        vec = self.predict_proba(X)
        y_pred = np.zeros((1, vec.shape[1]))


        print("TODO: optimze this loop")

        for i in range(vec.shape[1]):
            y_pred[0,i] = 1 if vec[0, i] > threshold else 0

        return y_pred

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
