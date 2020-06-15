import matplotlib as mpl
import matplotlib.pyplot as plt
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
    def mpl_patch(self, diskcolor= 'blue'):

        self.mypatch =  mpl.patches.Circle( self.center, self.radius, facecolor = diskcolor, picker=1 )

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
        self.mypatch =  mpl.patches.ConnectionPatch(self.start, self.end, "data", "data")

        return self.mypatch

    def set_color(self, color):
        self.mypatch.set_color(color)

    def set_zorder(self, zOrder):
        self.mypatch.zorder = zOrder

class Visual():

    #init the visual class that will contain the plots and dictionaries
    def __init__(self, showInput = False):

        self.fig, self.ax = plt.subplots()
        self.nodesDict = {}
        self.linesDict = {}
        self.textDict = {}
        self.weightDict = {}

        self.layer_dims = None
        self.names = None
        self.params = None

        self.showInput = showInput

        self.ax.set_aspect('equal')

    #draw the basic neural network
    def drawNetwork(self, layer_dims, names, params):

        if self.showInput == False:
            layer_dims.pop(0)
            del params['W1']
            del params['b1']

        self.layer_dims = layer_dims
        self.names = names
        self.params = params

        #get size for x and y
        maxX = len(layer_dims)
        maxY = max(layer_dims)

        self.ax.set(xlim=(0, maxX + 1), ylim = (0, maxY + 1))

        #draw the nodes
        id = 0
        for i, layer in enumerate(layer_dims):

            #for each node in layer
            for n in range(layer):

                #set position for node
                xPos = i + 1
                yPos = n +  maxX / layer

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

        oldLayerId = 0
        i = -1
        #draw the weights
        for key in self.linesDict:

            #get line
            line = self.linesDict[key]
            layerId = line.layer

            #get coordinates from line
            start = line.start
            end = line.end

            #get middle point of line
            xMpos = (start[0] + end[0])/2
            yMpos = (start[1] + end[1])/2

            if layerId == oldLayerId:
                i = i + 1
            else:
                i = -1

            layerId = int(layerId) + 2
            text = params["W" + str(layerId)][0][i]
            text = round(text, 3)
            print(text)
            weight = plt.text(xMpos, yMpos, text)

        #draw the names
        for i, elem in enumerate(names):

            #set position of text
            xPos = i + 1
            yPos = 0.5

            #set content of text
            text = plt.text(xPos, yPos, elem)

            #save text to dict
            self.textDict[str(xPos)] = text

    def show(self):
        plt.show()
