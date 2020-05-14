import numpy as np
import pandas as pd
import h5py
from matplotlib import pyplot as plt

def load_data_change(path):
    data = pd.read_pickle(path)

    #remove ids
    data.drop('gml_id_17', 1, inplace=True)
    data.drop('gml_id_18', 1, inplace=True)
    data.drop('polygon_17', 1, inplace=True)
    data.drop('bgt_group_17', 1, inplace=True)

    y = data['change'].copy()

    data.drop('change', 1, inplace=True)

    cols = data.columns
    for i, elem in enumerate(cols):
        if data[elem].dtypes == 'object':
            data[elem] = data[elem].astype('category').cat.codes

    X = data.copy()

    #replace None values with NaN
    X.fillna(value=np.nan, inplace=True)
    y.fillna(value=0, inplace=True)

    return X,y



def load_data_cats():
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

    #check data for consistency I
    assert len(train_x.shape) == len(test_x.shape), "shape for train and test data of X is different"
    assert len(train_y.shape) == len(test_y.shape), "shape for train and test data of Y is different"

    #check data for consistency II
    for i in range(1, len(train_x.shape)):
        assert train_x.shape[i] == test_x.shape[i], "train and test data are different at position " + str(i)

    #try to determine type of data
    if len(train_x.shape) == 2:
        type = "data"
    if len(train_x.shape) == 4:
        type = "image"

    m_train = train_x.shape[0]
    m_test = test_x.shape[0]
    print("Size training-set:", m_train)
    print("Size test-set:", m_test)

    if type == "image":
        x_size = train_x.shape[1]
        y_size = train_x.shape[2]
        num_bands = train_x.shape[3]
        print("Image size: " + str(x_size) + "x" + str(y_size))
        print("Number of bands:", num_bands)

    print("")

def display_image(array):
    plt.imshow(array, interpolation='nearest')
    plt.show()
