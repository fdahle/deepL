import h5py
import pickle
import numpy as np
import pandas as pd
import skimage.color

from sklearn.preprocessing import LabelEncoder
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

#convert rgb image to grayscale
def convert_grayscale(data):
    returnData = []
    for i in range(data.shape[0]):
        img = skimage.color.rgb2gray(data[i])
        returnData.append(img)
    return np.asarray(returnData)

#return subset of data based on different criteria
def getSubset(data, type, count, online=False, seed=123):

    np.random.seed(123)

    if type == "first":
        output = data[:count]

    if type == "random":
        output = data[np.random.choice(len(data), size=count, replace=False)]

    return output

#distinguish between different type of example data
def loadExamples(type):

    if type=="cats":
        dataset = loadCats()
    elif type=="sonar":
        dataset = loadSonar()
    elif type=="mnist":
        dataset = loadMnist()
    elif type=="cifar":
        dataset = loadCifar()
    elif type=="makeMoons":
        dataset = loadMoons()
    else:
        raise ValueError('A dataset with this key is not available')

    #data is always a structure with train_x, train_y, test_x, test_y, classes
    return dataset

#get data for cats vs dogs
def loadCats():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")

    train_x = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_y = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_x = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_y = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    return train_x, train_y, test_x, test_y, classes

#get data for sonar
def loadSonar():
    dataframe = pd.read_csv('datasets/sonar.csv', header=None)
    dataset = dataframe.values

    np.random.shuffle(dataset)

    X = dataset[:,0:60].astype(float)
    Y = dataset[:,60]

    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(Y)
    Y = encoder.transform(Y)

    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=42)

    ## TODO: make classes right
    classes = np.asarray([0,1])

    return train_x, train_y, test_x, test_y, classes

#get data for mnist (handwritten digits)
def loadMnist():
    #train_data = np.loadtxt("datasets/mnist_train.csv", delimiter=",")
    #test_data = np.loadtxt("datasets/mnist_test.csv", delimiter=",")

    train_data = np.load("datasets/mnist_train.npy")
    test_data = np.load("datasets/mnist_test.npy")

    def reshape(data):
        output = data.reshape((28,28))
        return(output)

    train_x = train_data[:,1:]
    train_x = np.apply_along_axis(reshape, 1, train_x)
    train_y = train_data[:,0]

    test_x = test_data[:,1:]
    test_x = np.apply_along_axis(reshape, 1, test_x)
    test_y = test_data[:,0]

    classes = np.asarray([0,1,2,3,4,5,6,7,8,9])

    return train_x, train_y, test_x, test_y, classes

def loadCifar():
    with open("datasets/cifar/data_batch_1", 'rb') as file:
        train_data = pickle.load(file, encoding='bytes')
    with open("datasets/cifar/test_batch", 'rb') as file:
        test_data = pickle.load(file, encoding='bytes')

    train_x = np.asarray(train_data.values())
    train_y = np.asarray(train_data.keys())

    test_x = np.asarray(test_data.values())
    test_y = np.asarray(test_data.keys())

    classes = np.unique(np.asarray(train_y))

    return train_x, train_y, test_x, test_y, classes


def loadMoons():
    X, y = make_moons(n_samples = 1000, noise=0.2, random_state=100)
    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=42)

    classes = [0,1]

    return train_x, train_y, test_x, test_y, classes
