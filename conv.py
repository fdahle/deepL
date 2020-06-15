import numpy as np
from numba import jit
from numba.typed import List

from functions import *
import math

#initialize filter with random values
def init_filters(x, y, num_filters):

    #check the size of filter
    assert x % 2 == 1 and y % 2 == 1, "Filter must have an odd shape (3x3,5x5,..)"

    ## TODO: Check xavier init https://www.quora.com/What-is-an-intuitive-explanation-of-the-Xavier-Initialization-for-Deep-Neural-Networks

    # divide by 9 to reduce the variance of our initial values
    filters = np.random.randn(num_filters, x, y) / 9

    return filters

#apply filter on a image
@jit(nopython=True)
def apply_filter(image, filters, stride=1, bias=0):

    assert image.shape[0] >= filters[0].shape[0] or \
     image.shape[1] >= filters[0].shape[1], "Image size must be greater than the filter"

    #get outputSize
    outX = (image.shape[0] - filters[0].shape[0])/stride + 1
    outY = (image.shape[1] - filters[0].shape[1])/stride + 1

    #get half size of filter to substract this value from the image edges
    fhX = int(filters[0].shape[0] / 2) #filterHalfX
    fhY = int(filters[0].shape[1] / 2) #filterHalfY

    ## TODO: find out how to solve this with numba
    #check if filter can be applied with this stride
    #if outX.is_integer() is False or outY.is_integer() is False:
    #    raise ValueError('Filter can be applied to image with this size and stride')

    ## TEMP:
    outX = int(outX)
    outY = int(outY)

    #create empty list to save output
    outputs = []

    for filter in filters:
        #create empty output
        output=np.empty((outX, outY))
        oX, oY = 0,0

        #iterate over image and apply filter
        for i in range(fhX, image.shape[0] - fhX, stride):
            for j in range(fhY, image.shape[1] - fhY, stride):

                #create subset from the image
                subset = image[i-fhX:i+fhX+1, j-fhY:j+fhY+1] #+1 as at slicing np exclude last number

                #get the filter value for this subset
                output[oX, oY] = np.sum(filter*subset) + bias

                oY += 1
            oX += 1
            oY = 0

        #flatten and append to list
        outputs.append(output)

    #to remove the deprecated warning
    typed_outputs = List()
    [typed_outputs.append(x) for x in outputs]

    return typed_outputs

@jit(nopython=True)
def downsample(filtered, kSize=2, stride=2):

    #get size for the output image
    outX = int((filtered[0].shape[0] - kSize)/stride + 1)
    outY = int((filtered[0].shape[1] - kSize)/stride + 1)

    #get half size of pooling to substract this value from the image edges
    phX = int(kSize / 2) #poolingHalfX
    phY = int(kSize / 2) #poolingHalfY

    ## TODO: check if outX and outY have decimals
    # TODO: check if ksize is even
    outX = int(outX)
    outY = int(outY)

    #create empty list to save output
    outputs = []

    #iterate all filtered images
    for image in filtered:
        output = np.empty((outX, outY))
        oX, oY = 0,0

        #iterate one image
        for i in range(phX, image.shape[0], stride):
            for j in range(phY, image.shape[1], stride):

                #get subset of this image
                subset = image[i-phX:i+phX, j-phY:j+phY] #+1 as at slicing np exclude last number

                #only transfer the maxium value of this subset
                output[oX, oY] = np.max(subset)

                oY += 1
            oX += 1
            oY = 0

        outputs.append(output)

    return outputs

#forward propagation for cnn
def cnn_forward_prop(X, filters, imgShape, fStride, fBias, pStride, pSize):

    output = []
    filteredCache = []

    #iterate all images
    for img in X.T:

        #check if image is in grayscale
        try:
            img = img.reshape(imgShape[0], imgShape[1])
        except:
            raise ValueError("The input image has too many pixels for its shape (Can be caused by non grayscale images)")

        #apply filter
        filtered = apply_filter(img, filters, stride=fStride, bias=fBias)
        filteredCache.append(filtered)

        #downsample
        pooled = downsample(filtered, kSize = pSize, stride=pStride)

        #convert list of images into an 1d array
        images = np.asarray(pooled).flatten()

        #append to the outputlist
        output.append(images)

    #convert list to array
    output = np.asarray(output)

    output = output.T

    #ouput = array with each element the flattenend results of all pooling images
    #filteredCache = unflattened filtered images
    return output, filteredCache

@jit(nopython=True)
def deapply_filter(img, cnnGrad, filters, stride=1):

    #get half size of filter to substract this value from the image edges
    fhX = int(filters[0].shape[0] / 2) #filterHalfX
    fhY = int(filters[0].shape[1] / 2) #filterHalfY

    filterGrads = np.zeros(filters.shape)

    for f, filter in enumerate(filters):

        #iterate over image and apply filter
        for i in range(fhX, img.shape[0] - fhX, stride):
            for j in range(fhY, img.shape[1] - fhY, stride):
                filterGrads[f] += cnnGrad[f][i-1,j-1] * img[i-fhX:i+fhX+1, j-fhY:j+fhY+1]

    return filterGrads

#backward propagation for pooling
@jit(nopython=True)
def upsample(grads, cache, kSize=2, stride=2):

    #get size for the output image
    outX = int((cache[0].shape[0] - kSize)/stride + 1)
    outY = int((cache[0].shape[1] - kSize)/stride + 1)

    #get half size of pooling to substract this value from the image edges
    phX = int(kSize / 2) #poolingHalfX
    phY = int(kSize / 2) #poolingHalfY

    #create empty list to save output
    outputs = []

    #iterate all filters
    for k, image in enumerate(cache):

        #make gradient in 2d
        grad = grads[k].copy() #copy needed that reshape can run in numba
        grad = grad.reshape(outX,outY)

        #create np.array where everything is zero
        output = np.zeros((image.shape[0], image.shape[1]))
        oX, oY = 0, 0

        #iterate one image
        for i in range(phX, image.shape[0], stride):
            for j in range(phY, image.shape[1], stride):

                #get subset of this image
                subset = image[i-phX:i+phX, j-phY:j+phY] #+1 as at slicing np exclude last number

                #get max value and replace it with the gradient value
                subset = subset.flatten()
                subset[subset == np.amax(subset)] = grad[oX, oY]
                subset = subset.reshape((kSize,kSize))

                #put the result back to the image
                output[i-phX:i+phX, j-phY:j+phY] = subset

                oY += 1
            oX += 1
            oY = 0

        outputs.append(output)

    typed_outputs = List()
    [typed_outputs.append(x) for x in outputs]

    return typed_outputs

#backward propagation for cnn
def cnn_backward_prop(grads, X, imgShape, cacheF, filters): #cacheF = filteredCache

    cnnGrads = []

    #transpose grads to iterate through it
    grads = grads.T


    #iterate all images
    for i, img in enumerate(X.T):

        #get grad for certain image
        grad = grads[i]
        print(grad)

        grad = np.reshape(grad, (len(cacheF[i]),-1))

        grad = upsample(grad, cacheF[i])
        img = img.reshape(imgShape[0], imgShape[1])
        grad = deapply_filter(img, grad, filters)
        cnnGrads.append(grad)



    cnnGrads = np.asarray(cnnGrads)
    cnnGrads = cnnGrads.mean(axis=(0))
    return cnnGrads

def update_filters(filters, gradient, learning_rate):

    filters -= learning_rate * gradient

    return filters

#change input layer to the new inputsize
def changeInputLayer(layer_dims, inputSize):

    #replace input layer
    layer_dims[0] = inputSize

    return layer_dims
