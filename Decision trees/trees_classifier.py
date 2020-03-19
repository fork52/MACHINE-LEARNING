import numpy as np
import pprint as pr
import matplotlib as mpl
import math 
import operator
	
def calcShannonEnt(dataSet,labels):
	'''
	Function which returns the entropy of the dataSet given according to the last
	'''
	currentIndex=0
	numEntries = len(dataSet)
	labelCounts = {}                         # empty dictionary
	#Find the different labels available in the dataSet
	for featVec in dataSet:
		currentLabel = labels[currentIndex]
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel] = 0
		labelCounts[currentLabel] += 1
		currentIndex += 1
	shannonEnt = 0.0
	for key in labelCounts:    #iterating over the dictionary
		prob = float(labelCounts[key])/numEntries
		shannonEnt -= prob * math.log(prob,2)
	return shannonEnt

def splitDataSet(dataSet, axis, value,labels):
	'''
	axis simply means colunm no. in the dataSet
	Function which takes a dataset and returns the subset of data that has 'value' parameter as its 
	value for axis(column) in the dataSet.
	dataSet - Original dataSet
	axis    - Column no. on which split has to be done
	value   - value on which split has to be done
	labels  - list of labels for the Dataset
	'''
	retDataSet = []
	newLabels=[]
	currentIndex=0
	for featVec in dataSet:
		if featVec[axis] == value:
			#Remove the column axis form the data piece before adding it to the retDataSet using slicing
			reducedFeatVec = featVec[:axis]         
			reducedFeatVec.extend(featVec[axis+1:])
			retDataSet.append(reducedFeatVec)
			newLabels.append(labels[currentIndex])
		currentIndex += 1  
	return retDataSet,newLabels

def chooseBestFeatureToSplit(dataSet,labels):
	'''
	CHOOSES THE BEST FEATURE FORM THE DATASET TO SPLIT ON AND RETURNS ITS INTEGER LOCATION
	'''
	numFeatures = len(dataSet[0]) 
	baseEntropy = calcShannonEnt(dataSet,labels)
	bestInfoGain = 0.0 
	bestFeature = -1              #No feature selected yet
	for i in range(numFeatures):
		featList = [example[i] for example in dataSet]  #Create a list of all values of the column
		uniqueVals = set(featList)					    #Find the unqiue values only!
		newEntropy = 0.0
		for value in uniqueVals:
			subDataSet,subLabels = splitDataSet(dataSet, i, value,labels)
			weighFactor = len(subDataSet)/float(len(dataSet))
			newEntropy += weighFactor * calcShannonEnt(subDataSet,subLabels)
		infoGain = baseEntropy - newEntropy
		if (infoGain > bestInfoGain):
			bestInfoGain = infoGain
			bestFeature = i
	return bestFeature

def majorityCnt(classList):
	classCount={}
	for vote in classList:
		if vote not in classCount.keys():
			classCount[vote] = 0
		classCount[vote] += 1
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
	return sortedClassCount[0][0]

def createTree(dataSet,labels,colNames):

	if labels.count(labels[0]) == len(labels):
		return labels[0]

	if len(dataSet[0]) == 0:
		return majorityCnt(labels)

	bestFeat = chooseBestFeatureToSplit(dataSet,labels)

	bestFeatLabel = colNames[bestFeat]
	myTree = {bestFeatLabel:{}}      # Tree is a dictionary!!
	del(colNames[bestFeat])

	featValues = [example[bestFeat] for example in dataSet]
	uniqueVals = set(featValues)

	for value in uniqueVals:
		subcolNames = colNames[:]
		subDataset,subLabels=splitDataSet(dataSet, bestFeat, value,labels)
		myTree[bestFeatLabel][value] = createTree(subDataset,subLabels,subcolNames)
	return myTree


def createDataSet():
	'''
	SAMPLE DATASET FOR TRAINING
	1-True 0 -False
	'''
	dataSet = [
				['Sunny',1,1],
				['Sunny',0,1],
				['Windy',1,1],
				['Rainy',1,0],
				['Rainy',0,1],
				['Rainy',1,0],
				['Windy',0,0],
				['Windy',0,1],
				['Windy',1,1],
				['Sunny',0,1],
	
			]
	labels=['Cinema','Tennis','Cinema','Cinema','Stay in','Cinema','Cinema','Stay in','Cinema','Tennis']
	colNames=['Weather?','Parents?','Money?']
	
	return dataSet, labels,colNames
	


if __name__ == '__main__':
	dataSet,labels,colNames=createDataSet()
	root = createTree(dataSet,labels,colNames)
	pr.pprint(root,indent=3,width=80)