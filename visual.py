import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import random

class Node:

    def __init__(self, center, radius, id, visual):
        self.center = center
        self.radius = radius
        self.id = id

        self.visual = visual

        self.fig = visual.fig
        self.ax = visual.ax
        self.mypatch = None

    def on_click(self, event):
        if event.artist == self.mypatch:
            for key in self.visual.linesDict:
                startC = key.split("_")[0]
                endC = key.split("_")[1]

                if int(startC) == int(self.id) or int(endC) == int(self.id):
                    r = random.random()
                    b = random.random()
                    g = random.random()
                    color = (r, g, b)
                    self.visual.linesDict[key].set_color(color)
                    self.visual.linesDict[key].set_zorder(0)

                if not int(startC) == int(self.id) and not int(endC) == int(self.id):
                    self.visual.linesDict[key].set_color('lightgray')
                    self.visual.linesDict[key].set_zorder(-0.5)

            plt.pause(0.05)

    # Return a Matplotlib patch of the object
    def mpl_patch(self, diskcolor= 'white', edgecolor='black'):

        self.mypatch =  mpl.patches.Circle( self.center, self.radius,
                                           facecolor = diskcolor, edgecolor = edgecolor,
                                           picker=1 )

        if self.fig != None:
            self.fig.canvas.mpl_connect('pick_event', self.on_click) # Activate the object's method


        return self.mypatch

class Line:

    def __init__(self, start, end, layer, visual):
        self.start = start
        self.end = end
        self.layer = layer

        self.visual = visual

        self.mypatch = None

    def mpl_patch(self):
        self.mypatch =  mpl.patches.ConnectionPatch(self.start, self.end,
                                                    "data", "data",
                                                     color="darkgray", zOrder=0.5)

        return self.mypatch

    def set_color(self, color):
        self.mypatch.set_color(color)

    def set_zorder(self, zOrder):
        self.mypatch.zorder = zOrder

class Visual():

    #init the visual class that will contain the plots and dictionaries
    def __init__(self, showInput = False):

        self.fig = plt.figure()
        self.ax = plt.subplot()
        self.fig.axes.append(self.ax)
        self.nodesDict = {}
        self.linesDict = {}
        self.activationDict = {}
        self.weightDict = {}
        self.classDict = {}

        self.layer_dims = None
        self.activation = None
        self.params = None
        self.classes = None

        self.showInput = showInput

        self.ax.set_aspect('equal')

    #draw the basic neural network
    def drawNetwork(self, layer_dims, activation, params, Y):

        self.layer_dims = layer_dims.copy()
        self.activation = activation.copy()
        self.params = params.copy()

        self.classes = np.unique(Y)

        if self.showInput == False:
            self.layer_dims.pop(0)
            del self.params['W1']
            del self.params['b1']

        #get size for x and y
        maxX = len(self.layer_dims)
        maxY = max(self.layer_dims)

        self.ax.set(xlim=(0, maxX*3), ylim = (0, maxY + 2))

        #draw the nodes
        id = 0
        for i, layer in enumerate(self.layer_dims):

            #for each node in layer
            for n in range(layer):

                #set position for node
                xPos = i * 3 + 1
                yPos = n + (maxY-layer)/2 + 2

                #create node
                node = Node((xPos, yPos), .2, id, self)

                #add node
                self.ax.add_patch(node.mpl_patch())

                #save node
                self.nodesDict[str(i) + "_" + str(n)] = node

                #iterate node id
                id = id + 1

        #draw the lines
        for key in self.nodesDict:

            #get source of node
            startX, startY = self.nodesDict[key].center
            startId = self.nodesDict[key].id


            layer = key.split("_")[0]
            subsetDict = {k:v for k,v in self.nodesDict.items() if str(int(layer) + 1)+"_" in k}

            for _key in subsetDict:
                endX, endY = subsetDict[_key].center
                endId = subsetDict[_key].id

                line = Line((startX, startY), (endX, endY), layer, self)

                self.ax.add_patch(line.mpl_patch())

                self.linesDict[str(startId)+"_"+str(endId)] = line

        #draw the weights
        oldLayerId = 0 #settings I for finding the right layer
        i = -1 #settings II for finding the right layer
        for key in self.linesDict:
            #get line
            line = self.linesDict[key]
            layerId = line.layer

            #get coordinates from line
            start = line.start
            end = line.end

            #get formula for line
            coefficients = np.polyfit([start[0], end[0]], [start[1], end[1]], 1)
            polynomial = np.poly1d(coefficients)

            #determines how far away the weight should be displayed
            posText = 0.2

            #get line X distance
            lineXwidth = end[0] - start[0]

            #get point of line at posText
            posX = start[0] + posText * lineXwidth
            posY = posX*polynomial[1] + polynomial[0]


            #increment if layer stays the same or reset
            if layerId == oldLayerId:
                i = i + 1
            else:
                i = -1

            #find right layer
            layerId = int(layerId) + 2

            #get weight
            text = params["W" + str(layerId)][0][i]

            #round weight
            text = round(text, 3)

            #plot text
            weight = plt.text(posX, posY, text, fontsize=6,
                              horizontalalignment='center',
                              verticalalignment='center')

        #draw the names
        for i, elem in enumerate(self.activation):

            #set position of text
            xPos = i * 3 + 1
            yPos = 0.5

            #set content of text
            text = plt.text(xPos, yPos, elem,
                          horizontalalignment='center',
                          verticalalignment='center')

            #save text to dict
            self.activationDict[str(xPos)] = text

        #draw the classes
        for i, elem in enumerate(reversed(self.classes)):

            #set position for class
            xPos = maxX * 3 - 1
            yPos = i + (maxY-max(self.layer_dims))/2 + 2

            #set content of text
            text = plt.text(xPos, yPos, elem,
                          verticalalignment='center')


            #save text to dict
            self.classDict[str(yPos)] = text

    def show(self):
        plt.ion() #interactive mode so that calculation can continue afer plot
        self.fig.plot()
