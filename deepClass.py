import numpy as np
import os
import pickle

from functions import *
from conv import *
from visual import *

class DeepLearner():

    #initialize class
    def __init__(self, learning_type="classic", random_seed=123):

        #basic settings for training
        self.learning_type = learning_type
        self.random_seed = random_seed

        #default settings for training
        self.layer_dims = [4,2]
        self.layer_activation = ["relu","relu"]
        self.numberIterations = 10000
        self.learning_rate = 0.01
        self.threshold = 0.5
        self.numClasses = None
        self.early_stopping = None

        #default settings for cnn
        self.filters = None
        self.filterX = 3
        self.filterY = 3
        self.numFilters = 8
        self.filterBias = 0
        self.filterStride = 1
        self.poolSize = 2
        self.poolStride = 2

        #params for history
        self.params = None
        self.costHistory = None
        self.accuracyHistory = None

        #other settings
        self.visual_mode = False
        self.debug_mode = False

    #gives the possibility to change training parameters
    def setTrainingParams(self, layer_dims=None, layer_activation=None, numberIterations=None, learning_rate=None, threshold=None, early_stopping = None):

        if layer_dims is not None:
            self.layer_dims = layer_dims
        if layer_activation is not None:
            self.layer_activation = layer_activation
        if numberIterations is not None:
            self.numberIterations = numberIterations
        if learning_rate is not None:
            self.learning_rate = learning_rate
        if threshold is not None:
            self.threshold = threshold
        if early_stopping is not None:
            self.early_stopping = early_stopping

    #gives the possibility to change cnn parameters
    def setCnnParams(self, filterX = None, filterY = None, numFilters = None, filterBias = None, filterStride=None, poolSize=None, poolStride=None):

        if filterX is not None:
            self.filterX = filterX
        if filterY is not None:
            self.filterY = filterY
        if numFilters is not None:
            self.numFilters = numFilters
        if filterBias is not None:
            self.filterBias = filterBias
        if filterStride is not None:
            self.filterStride = filterStride
        if poolSize is not None:
            self.poolSize = poolSize
        if poolStride is not None:
            self.poolStride = poolStride

    #gives the possibility to change some settings
    def setSettings(self, visual_mode=None, debug_mode=None):
        if visual_mode is not None:
            self.visual_mode = visual_mode

        if debug_mode is not None:
            self.debug_mode = debug_mode

    #train the network
    def optimize(self, X_input, Y_input, verbose=False, verbose_iter=100):

        #check data and settings for consistency
        checkErrors(self.layer_activation, self.layer_dims, X_input, Y_input)

        #set random_seed
        np.random.seed(self.random_seed)

        #copy data in order not to change the original data
        X = np.asarray(X_input).copy()
        Y = np.asarray(Y_input).copy()

        #count number of classes
        self.numClasses = len(np.unique(Y, return_counts=False))
        if self.numClasses == 2: #if number of classes is 2 it can be treated as 1 class (binary )
            self.numClasses = 1

        #check input values have NaN values
        if (hasNaN(X)):
            print("X has NaN-values. These are replaced with 0")
            X[np.isnan(X)] = 0

        #save image params if it is a cnn
        if self.learning_type == "cnn":
            imgSizeX = X.shape[1]
            imgSizeY = X.shape[2]
            imgShape = [imgSizeX, imgSizeY]

        #normalize image if cnn
        if self.learning_type == "cnn":
            X = (X / 255) - 0.5

        #reshape X the structure this NN is working with
        X = reshape(X) #(features, number of entries)
        Y = reshape(Y) #(number of classes, number of entries)

        #copy X for later use in cnn
        if self.learning_type == "cnn":
            X_orig = X.copy()

        #adapt layer (add input and output layer)
        self.layer_dims = adapt_layer(self.layer_dims, self.layer_activation, X.shape[0], self.numClasses)

        #init weight and bias
        self.params = init_params(self.layer_dims)

        #init the visualization mode
        if self.visual_mode:
            visual = Visual()
            visual.drawNetwork(self.layer_dims, self.layer_activation, self.params)
            visual.show()

        #init filters if it is a cnn
        if self.learning_type == "cnn":
            self.filters = init_filters(self.filterX, self.filterY, self.numFilters)

        #create empty list to save history
        self.costHistory = []
        self.accuracyHistory = []

        #set variable for the changing of the input layer if cnn
        if self.learning_type == "cnn":
            inputLayerChanged = False

        #start training
        for i in range(self.numberIterations):

            #execute cnn forward propagation
            if self.learning_type == "cnn":

                if self.debug_mode: print("   Round " + str(i) + ": CNN forward propagation")

                #forward propagation for cnn
                X, filteredCache = cnn_forward_prop(X_orig, self.filters, imgShape,
                                                          self.filterStride, self.filterBias,
                                                          self.poolStride, self.poolSize)

                #when applying the filters and downsample the input size
                #is changing and needs to be adapted only once
                if inputLayerChanged == False:

                    #change input layer
                    self.layer_dims = changeInputLayer(self.layer_dims, X.shape[0])

                    #with new layer_dims new params are needed
                    self.params = init_params(self.layer_dims)

                    #set bool so that the whole if clause is only done once
                    inputLayerChanged = True

            if self.debug_mode: print("   Round " + str(i) + ": forward propagation")

            #execute forward propagation
            AL, cache = forward_prop(X, self.params, self.layer_activation)

            if self.debug_mode: print("   Round " + str(i) + ": compute costs")

            #compute cost and save
            cost = compute_cost(AL, Y, self.numClasses, "crossEntropy",)
            self.costHistory.append(cost)

            if self.debug_mode: print("   Round " + str(i) + ": compute accuracy")

            #compute accuracy and save
            accuracy = compute_accuracy(AL, Y, self.numClasses, self.threshold)
            self.accuracyHistory.append(accuracy)

            if self.debug_mode: print("   Round " + str(i) + ": backward propagation")

            #execute backwards propagation
            grads = backward_prop(AL, Y, self.params, self.layer_activation, cache, self.numClasses)

            if self.learning_type == "cnn":

                if self.debug_mode: print("   Round " + str(i) + ": CNN backward propagation")

                filterGrads = cnn_backward_prop(grads["dA0"], X_orig, imgShape, filteredCache, self.filters)

            if self.debug_mode: print("   Round " + str(i) + ": update parameters")

            #update parameters
            self.params = update_params(self.params, self.layer_activation, grads, self.learning_rate)

            #update filter if necessary
            if self.learning_type == "cnn":

                if self.debug_mode: print("   Round " + str(i) + ": udpdate filters")

                self.filters = update_filters(self.filters, filterGrads, self.learning_rate)

            #print out cost
            if verbose and i%verbose_iter == 0:
                print("Round " + str(i) + " cost: " + str(round(cost, 10)) + ", accuracy: " +str(round(accuracy, 3)))

        self.plotTraining("accuracy")

    #classify input and returns the probability
    def predictProba(self, X_input):

        X = np.asarray(X_input).copy()

        ## TODO: check if data is prepared or not and only do if not prepared
        X = reshape(X)

        m = X.shape[1]

        probas, caches = forward_prop(X, self.params, self.layer_activation)

        return(probas[0])

    #classify input with most likely class
    def predict(self, X_input, threshold=0.5):

        vec = self.predictProba(X_input)
        vec = vec[np.newaxis]
        y_pred = np.zeros((1, vec.shape[1]))

        #TODO: optimze this loop")
        for i in range(vec.shape[1]):
            y_pred[0,i] = 1 if vec[0, i] > threshold else 0

        return y_pred[0]

    #save model to file
    def saveModel(self, filename, path=os.getcwd(), type="pickle"):

        #save all parameters in a dict
        modelDict = {}

        #fill dict with training settings
        modelDict["layer_dimns"] = self.layer_dims
        modelDict["layer_activation"] = self.layer_activation
        modelDict["numberIterations"] = self.numberIterations
        modelDict["learning_rate"] = self.learning_rate
        modelDict["threshold"] = self.threshold

        #fill dict with training results
        modelDict["params"] = self.params
        modelDict["costHistory"] = self.costHistory
        modelDict["accuracyHistory"] = self.accuracyHistory

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

    #load model from dict and save values to class
    def loadModel(self, path):
        type = path[-3:]

        # TODO: do other types
        if type == "pkl":
            file = open(path, 'rb')
            modelDict = pickle.load(file)
            file.close()

        self.layer_dims = modelDict["layer_dimns"]
        self.layer_activation = modelDict["layer_activation"]
        self.numberIterations = modelDict["numberIterations"]
        self.learning_rate = modelDict["learning_rate"]
        self.threshold = modelDict["threshold"]

        self.params = modelDict["params"]
        self.costHistory = modelDict["costHistory"]
        self.accuracyHistory = modelDict["accuracyHistory"]

    #plot training graph
    def plotTraining(self, type):

        if type == "cost":
            costs = np.squeeze(self.costHistory)
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations (in 100)')
            plt.title('Learning rate =' + str(self.learning_rate))
            plt.show()

        if type == "accuracy":
            costs = np.squeeze(self.accuracyHistory)
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations (in 100)')
            plt.title('Learning rate =' + str(self.learning_rate))
            plt.show()
