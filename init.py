import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

from load import *
from functions import *
from evaluations import *

modelPath = "models"
modelName = "cats"
num_iterations = 1500
learning_rate = 0.0075
print_cost = True

#train_x, train_y = load_data_change("datasets/trainSet_123.pkl")
#test_x, test_y = load_data_change("datasets/testSet_123.pkl")

train_x, train_y, test_x, test_y, classes = load_data_cats()

#get info from the data
train_x = convert_grayscale(train_x)
test_x = convert_grayscale(test_x)

print_info(train_x, train_y, test_x, test_y)
exit()
deepL = DeepLearner()
deepL.setTrainingParams(layer_dims = [16,8,4], num_iterations=num_iterations,
                        learning_rate=learning_rate, loss="crossEntropy")

#Gradient descent
deepL.optimize(train_x, train_y, print_cost=True, random_seed=1)
deepL.saveModel(modelPath + "/" + modelName)
deepL.loadModel(modelPath + "/" + modelName + ".pkl")

#Predict test/train set examples
pred_y = deepL.predict(test_x)

probs = deepL.predictProba(test_x)

get_cm_score(test_y, pred_y, "cm", silent=False)
plot_curve(test_y, probs, "roc")
plot_curve(test_y, probs, "pr")
