import numpy as np
import matplotlib.pyplot as plt

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

        def initialize_params(layer_dims):

            #reset
            self.parameters = {}
            L = self.num_layers

            for i in range(1, L):
                self.parameters['W' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i-1]) * 0.01
                self.parameters['b' + str(i)] = np.zeros(shape=(layer_dims[i], 1))

        def prop_forward(X):

            def linear_activation_forward(A_prev, W, b, activation):

                def linear_forward(A, W, b):

                    Z = np.dot(W, A) + b
                    cache = (A, W, b)

                    return Z, cache

                def sigmoid(z):
                    A = 1/(1+np.exp(-z))
                    cache = z

                    return A, cache

                def relu(z):
                    A = np.maximum(0,z)
                    cache = z
                    return A, cache

                if activation == "sigmoid":
                    Z, linear_cache = linear_forward(A_prev, W, b)
                    A, activation_cache = sigmoid(Z)
                elif activation == "relu":
                    Z, linear_cache = linear_forward(A_prev, W, b)
                    A, activation_cache = relu(Z)

                cache = (linear_cache, activation_cache)
                return A, cache

            caches = []

            A = X
            L = len(self.parameters) // 2

            for i in range(1, L):
                A_prev = A
                A, cache = linear_activation_forward(A_prev,
                                self.parameters["W" + str(i)],
                                self.parameters["b" + str(i)],
                                'relu')

                caches.append(cache)

            AL, cache = linear_activation_forward(A,
                            self.parameters["W" + str(L)],
                            self.parameters["b" + str(L)],
                            'sigmoid')
            caches.append(cache)

            return AL, caches

        def prop_backward(AL, Y_orig, caches):

            def sigmoid_backward(dA, cache):
                Z = cache

                s = 1/(1+np.exp(-Z))
                dZ = dA * s * (1-s)

                return dZ


            def linear_backward(dZ, cache):

                A_prev, W, b = cache
                m = A_prev.shape[1]

                dW = (1 / m) * np.dot(dZ, cache[0].T)
                db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
                dA_prev = np.dot(cache[1].T, dZ)

                return dA_prev, dW, db


            grads = {}
            L = len(caches)
            m = AL.shape[1]
            Y = Y_orig.reshape(AL.shape)

            dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

            current_cache = caches[-1]
            grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])

            for i in reversed(range(L-1)):
                current_cache = caches[i]
                dA_prev_temp, dW_temp, db_temp = linear_backward(sigmoid_backward(dAL, current_cache[1]), current_cache[0])
                grads["dA" + str(i + 1)] = dA_prev_temp
                grads["dW" + str(i + 1)] = dW_temp
                grads["db" + str(i + 1)] = db_temp

            return grads

        def compute_cost(AL, Y):
            m = Y.shape[1]
            cost = (-1/m) * np.sum(np.multiply(Y, np.log(AL)) + np.multiply((1-Y), np.log(1-AL)))
            cost = np.squeeze(cost)

            return cost

        def update_parameters(grads):
            L = self.num_layers -1

            for i in range(1, L):
                self.parameters['W' + str(i + 1)] = self.parameters['W' + str(i + 1)] - self.learning_rate * grads['dW' + str(i + 1)]
                self.parameters['b' + str(i + 1)] = self.parameters['b' + str(i + 1)] - self.learning_rate * grads['db' + str(i + 1)]

        # TODO: fix random seed
        np.random_seed=random_seed

        #save  parameters for later
        self.num_iterations = num_iterations
        self.learning_rate = learning_rate
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
        initialize_params(layer_dims)

        #reset costs
        self.costs = []

        #train
        for i in range(self.num_iterations):

            #forward propagation
            AL, caches = prop_forward(X)

            #compute cost
            cost = compute_cost(AL, Y)

            #backwards propagation
            grads = prop_backward(AL, Y, caches)

            #update parameters
            update_parameters(grads)

            #record costs
            if i%print_iter == 0:
                self.costs.append(cost)

            if print_cost and i%print_iter == 0:
                print('Cost after iteration %i: %f' % (i, cost))

    def predict_proba(self, X_input):
        #compute prob vector
        L = self.num_layers


        X = self.prepareData(X_input)

        vec = X
        for i in range(1, L):
            z = np.dot(self.parameters["W" + str(i)], vec) + self.parameters["b" + str(i)]
            vec = np.maximum(0,z) #relu

        return(vec)

    def predict(self, X, threshold=0.5):
        m = X.shape[0]

        Y_pred = np.zeros((1,m))
        vec = self.predict_proba(X)

        print("TODO: optimze this loop")

        for i in range(vec.shape[1]):
            Y_pred[0,i] = 1 if vec[0, i] > threshold else 0

        return Y_pred

    def plotCostFunction(self):
        _costs = np.squeeze(self.costs)
        plt.plot(_costs)
        plt.ylabel('cost')
        plt.xlabel('iterations (in 100)')
        plt.title('Learning rate =' + str(self.learning_rate))
        plt.show()
