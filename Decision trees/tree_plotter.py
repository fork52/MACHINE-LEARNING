import matplotlib.pyplot as plt
import trees_classifier
import pprint as pr

# CONSTANTS
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
	'''
	Plots a single node with  
	nodeTxt  - Text for the node
	centerPt - Center point co-ordinates of the node
	parentPt - Where the arraow to the point starts
	'''
	createPlot.ax1.annotate(nodeTxt, xy=parentPt,xycoords='axes fraction',xytext=centerPt, textcoords='axes fraction',
	va="center", ha="center", bbox=nodeType, arrowprops=arrow_args) #plot variable

def createTestPlot():
	'''Just to see of it the node definition is right'''
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	createPlot.ax1 = plt.subplot(111, frameon=False)  # plot varable
	plotNode('a decision node', (0.5, 0.1), (0.1, 0.5), decisionNode)
	plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
	plt.show()

def getNumLeafs(myTree):
	numLeafs = 0
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		# if key is a dictionary then it is not a leaf!
		if type(secondDict[key]).__name__=='dict':  
			numLeafs += getNumLeafs(secondDict[key]) #recursion
		else: 
			numLeafs +=1
	return numLeafs

def getTreeDepth(myTree):
	maxDepth = 0
	firstStr = list(myTree.keys())[0]
	secondDict = myTree[firstStr]
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			thisDepth = 1 + getTreeDepth(secondDict[key])
		else: 
			thisDepth = 1
		if thisDepth > maxDepth:
		 maxDepth = thisDepth
	return maxDepth

def plotMidText(cntrPt, parentPt, txtString):
	'''Plots the txtString in the middle of the branches showing transition'''
	xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
	yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
	createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
	'''
	Actual Tree plotter on the graph
	'''
	numLeafs = getNumLeafs(myTree)
	# getTreeDepth(myTree)
	firstStr = list(myTree.keys())[0]
	cntrPt = (plotTree.xOff + (1.0 + float(numLeafs))/2.0/plotTree.totalW,plotTree.yOff)

	plotMidText(cntrPt, parentPt, nodeTxt)
	plotNode(firstStr, cntrPt, parentPt, decisionNode)

	secondDict = myTree[firstStr]
	plotTree.yOff = plotTree.yOff - 1.0/plotTree.totalD
	
	for key in secondDict.keys():
		if type(secondDict[key]).__name__=='dict':
			plotTree(secondDict[key],cntrPt,str(key))
		else:
			plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
			plotNode(secondDict[key], (plotTree.xOff, plotTree.yOff),cntrPt, leafNode)
			plotMidText((plotTree.xOff, plotTree.yOff), cntrPt, str(key))
	plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD


def createPlot(inTree):
	'''
	Takes in a tree and outputs its graphical form
	'''
	fig = plt.figure(1, facecolor='white')
	fig.clf()
	axprops = dict(xticks=[], yticks=[])
	createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
	plotTree.totalW = float(getNumLeafs(inTree))
	plotTree.totalD = float(getTreeDepth(inTree))
	plotTree.xOff = -0.5/plotTree.totalW; plotTree.yOff = 1.0;
	plotTree(inTree, (0.5,1.0), '')
	plt.show()


if __name__ == '__main__':
	dataSet,labels,colNames=trees_classifier.createDataSet()
	root = trees_classifier.createTree(dataSet,labels,colNames)
	pr.pprint(root)
	noOfLeaves = getNumLeafs(root)
	depth  = getTreeDepth(root)
	print(noOfLeaves,depth)
	createPlot(root)