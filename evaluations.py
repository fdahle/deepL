import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve, precision_recall_curve

def plotImages(X, indices, plot):
    #todo make it privat:
    num_images = len(indices[0])
    sqrt = np.ceil(np.sqrt(num_images))

    for i in range(num_images):
        index = indices[1][i]
        subset = X[index]
        if plot == "all":
            plt.subplot(sqrt, sqrt, i + 1)
        plt.imshow(X[index])
        plt.axis('off')
        if plot == "single":
            plt.show()
        #plt.title("Prediction: " + classes[int(y_pred[0,index])].decode("utf-8") + " \n Class: " + classes[y_true[0,index]].decode("utf-8"))
    if plot == "all":
        plt.show()

def plot_mislabeled_images(X_input, y_true, y_pred, size=64.0, type='all', plot="all"):

    """
    plot = 'all', 'single'
    """

    #todo check if data needs to be reshaped
    X = X_input.copy()

    if type == 'all':
        a = y_true + y_pred
        indices = np.asarray(np.where(a == 1))
    elif type == 'FP':
        indices = np.asarray(np.where((y_true == 0) & (y_pred == 1)))
    elif type == 'FN':
        indices = np.asarray(np.where((y_true == 1) & (y_pred == 0)))

    plotImages(X_input, indices, plot)

#TODO: CORRECT ALL STUFF
def plot_correct_images(X_input, y_true, y_pred, size=64.0, type='all'):

    #todo check if data needs to be reshaped
    X = X_input.copy()

    if type == 'all':
        a = y_true + y_pred
        indices = np.asarray(np.where((a == 2) | (a == 0)))
    elif type == 'FP':
        indices = np.asarray(np.where((y_true == 0) & (y_pred == 1)))
    elif type == 'FN':
        indices = np.asarray(np.where((y_true == 1) & (y_pred == 0)))

    plotImages(X_input, indices, plot)

def get_cm_score(y_true, y_pred, type, silent=True, limit=3):

    tp = np.count_nonzero(np.where((y_true == 1) & (y_pred==1)))
    tn = np.count_nonzero(np.where((y_true == 0) & (y_pred==0)))
    fp = np.count_nonzero(np.where((y_true == 0) & (y_pred==1)))
    fn = np.count_nonzero(np.where((y_true == 1) & (y_pred==0)))


    if (type == "cm"):
        value = [tp, fp, tn, fn]

    elif (type == "accuracy"):
        value = (tp+tn) / y_true.shape[0]
        value = np.round(accuracy, limit)

    elif (type == "precision"):
        if (tp + fp == 0):
            value = 0
        else:
            value = tp/(tp+fp)
    elif (type == "recall"):
        if (tp + fn == 0):
            value = 0
        else:
            value = tp/(tp+fn)
    elif (type == "f1_score"):
        prec = get_cm_score(y_true, y_pred, "precision", silent=True, limit=limit)
        rec = get_cm_score(y_true, y_pred, "recall", silent=True, limit=limit)
        if (rec*prec == 0):
            value = 0
        else:
            value = 2*(rec*prec)/(rec+prec)

    if silent==False:
        print(type + " =", value)

    return value

def plot_curve(y_true, y_score, pType):

    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    if (pType == "roc"):
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        plt.plot(fpr, tpr,color='darkorange',lw=2)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.ylabel('Precision')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    elif (pType == "pr"):
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        no_skill = len(y_true[y_true==1]) / len(y_true)
        plt.plot(precision, recall,color='darkorange',lw=2)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.plot([0, 1], [no_skill, no_skill], linestyle='--', color='navy', label='No Skill')
    else:
        print("'" + pType + "' is not a valid plot type")
        return
    plt.show()
