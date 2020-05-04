import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from PIL import Image
from load import *
from functions import *

path = "C:/Users/felix/Google Drive/Coding/deepLearn"

num_iterations = 1500
learning_rate = 0.0075
print_cost = True

train_x, train_y, test_x, test_y, classes = load_dataset()

print_info(train_x, train_y, test_x, test_y)

deepL = DeepLearner()

#Gradient descent
deepL.optimize(train_x, train_y, [10,2], num_iterations, learning_rate, print_cost=True, random_seed=1)
deepL.saveModel("test")
deepL.loadModel(path + "/test.pkl")

#Predict test/train set examples
Y_prediction_test = deepL.predict(test_x)
Y_prediction_train = deepL.predict(train_x)

#Print test/train errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))

deepL.plotCostFunction()
