from load import *
from deepClass import *
import mnist

modelPath = "models"
modelName = "test"
train_x, train_y, test_x, test_y, classes = loadExamples("mnist")

train_x = getSubset(train_x, "first", 1000)
train_y = getSubset(train_y, "first", 1000)

#train_x = convert_grayscale(train_x)
#test_x = convert_grayscale(test_x)

#initialize the DeepLearner
deepL = DeepLearner(learning_type="cnn")

deepL.setTrainingParams(layer_dims=[128, 64], layer_activation=["relu", "relu"])

deepL.setSettings(debug_mode=False)

deepL.optimize(train_x, train_y, verbose=True, verboseCounter=1)
#deepL.saveModel(modelPath + "/" + modelName)
#deepL.loadModel(modelPath + "/" + modelName + ".pkl")

output = deepL.predict(test_x)
