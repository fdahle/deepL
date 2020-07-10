from load import *
from deepClass import *
from evaluations import *

modelPath = "models"
modelName = "test"

train_x, train_y, test_x, test_y, classes = loadExamples("makeMoons")

#train_x = convert_grayscale(train_x)
#test_x = convert_grayscale(test_x)

#train_data = loadCustom("datasets/trainSet_123.pkl", checkData=True)
#train_data = cleanData(train_data, clean=["string", "nan"])

#train_y = train_data[:,28]
#train_x = np.delete(train_data, 28, 1)

#train_x = getSubset(train_x, "first", 1000)
#train_y = getSubset(train_y, "first", 1000)

#test_data = loadCustom("datasets/testSet_123.pkl", checkData=True)
#test_data = cleanData(test_data, clean=["string", "nan"])

#test_y = test_data[:,28]
#test_x = np.delete(test_data, 28, 1)

#initialize the DeepLearner
deepL = DeepLearner(learning_type="classic")

deepL.setTrainingParams(layer_dims=[128, 64, 32], layer_activation=["leakyRelu", "leakyRelu", "leakyRelu"], early_stopping=500)
deepL.setTrainingParams(numberIterations=10000)
deepL.setTrainingParams(costType="crossEntropy", scoreMetrics=["accuracy", "pr"])

#deepL.setSettings(debug_mode=True)

deepL.optimize(train_x, train_y, verbose=True, verbose_iter=1)
deepL.saveModel(modelPath + "/" + modelName, format="json")
deepL.loadModel(modelPath + "/" + modelName + ".json")

output = deepL.predict(test_x)

plot_curve(test_y, output, "roc")
