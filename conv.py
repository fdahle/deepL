import numpy as np

def filter(image, filter, bias, stride=1):

    #bias is a value different for each filter

    #check the sizes of filters and images
    assert filter.shape[0] % 2 == 1 and filter.shape[1] % 2 == 1, "Filter must have an odd shape (3x3,5x5,..)"
    assert image.shape[0] >= filter.shape[0] or image.shape[1] >= filter.shape[1], "image size must be greater than the filter"

    x = int((image.shape[0] - filter.shape[0])/stride + 1)
    y = int((image.shape[1] - filter.shape[1])/stride + 1)
    kX = int(filter.shape[0]/2)
    kY = int(filter.shape[1]/2)

    output = np.empty([x,y])
    oX, oY = 0,0
    for i in range(kX, image.shape[0] - kX, stride):
        for j in range(kY, image.shape[1] - kY, stride):
            subset = image[i-1:i+2, j-1:j+2] #+2 as numpy excludes the last index
            #print(oX, oY)
            output[oX, oY] = np.sum(filter*subset) + bias
            oY += 1
        oX += 1
        oY = 0

    return output

def downsample(image, kSize, stride):
    #kSize = kernelSize

    x = int((image.shape[0] - kSize)/stride + 1)
    y = int((image.shape[1] - kSize)/stride + 1)
    kX = int(kSize/2)
    kY = int(kSize/2)

    output = np.empty([x,y])
    oX, oY = 0,0
    for i in range(kX, image.shape[0], stride):
        for j in range(kY, image.shape[1], stride):
            subset = image[i-1:i+2, j-1:j+2] #+2 as numpy excludes the last index
            #print(oX, oY)
            output[oX, oY] = np.max(subset)
            oY += 1
        oX += 1
        oY = 0
    print(output)
    return output

t = np.asarray([
    [0,0,10,0,0],
    [0,0,10,0,0],
    [0,0,10,0,0],
    [0,0,10,0,0],
    [0,0,10,0,0]
])

t1 = np.asarray([
    [0,1,0],
    [0,1,0],
    [0,1,0]
])
