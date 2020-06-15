from load import *
from deepClass import *
import mnist

modelPath = "models"
modelName = "test"
train_x, train_y, test_x, test_y, classes = loadExamples("cats")

train_x = convert_grayscale(train_x)
ätest_x = convert_grayscale(test_x)

#initialize the DeepLearner
deepL = DeepLearner(learning_type="cnn")

deepL.setSettings(debug_mode=False)

deepL.optimize(train_x, train_y)
#deepL.saveModel(modelPath + "/" + modelName)
#deepL.loadModel(modelPath + "/" + modelName + ".pkl")

output = deepL.predict(test_x)
