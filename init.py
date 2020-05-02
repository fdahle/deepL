import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from PIL import Image
from load import *
from functions import *

num_iterations = 2500
learning_rate = 0.05
print_cost = True

train_x, train_y, test_x, test_y, classes = load_dataset()

print_info(train_x, train_y, test_x, test_y)

deepL = DeepLearner()

#Gradient descent
deepL.optimize(train_x, train_y, [20, 7, 5], num_iterations, learning_rate, print_cost=True, random_seed=3)

#Predict test/train set examples
Y_prediction_test = deepL.predict(test_x)
Y_prediction_train = deepL.predict(train_x)

#Print test/train errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - train_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - test_y)) * 100))

deepL.plotCostFunction()
