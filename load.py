import numpy as np
import h5py
from matplotlib import pyplot as plt


def load_dataset():
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes

    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))

    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

def print_info(train_x, train_y, test_x, test_y):
    m_train = train_y.shape[1]
    m_test = test_y.shape[1]
    x_size = train_x.shape[1]
    y_size = train_x.shape[2]
    num_bands = train_x.shape[3]

    print("Size training-set:", m_train)
    print("Size test-set:", m_test)
    print("Image size: " + str(x_size) + "x" + str(y_size))
    print("Number of bands:", num_bands)
    print("")

def display_image(array):
    plt.imshow(array, interpolation='nearest')
    plt.show()
