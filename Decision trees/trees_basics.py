import numpy as np
import pprint as pr
import matplotlib as mpl
import matplotlib.pyplot as plt
import math 

def createDataSet():
	'''
	SAMPLE DATASET FOR TESTING
	1-True 0 -False
	'''
	dataSet = [[0, 1, 'no'], [0, 1, 'yes'], [1, 0, 'no'], [0, 1, 'no'], [1, 1, 'no']]
	labels = ['a','b','a','b','c']
	return dataSet, labels
	
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
	Function which takes a dataset and returns the subset of data(rows) that has 'value' parameter as its 
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
			#Remove the column 'axis' form the data piece before adding it to the retDataSet using slicing
			reducedFeatVec = featVec[:axis]         
			reducedFeatVec.extend(featVec[axis+1:])

			# ADD THE ROW TO DATASET TO BE RETURNED AND ADD ITS LABEL AS WELL
			retDataSet.append(reducedFeatVec)
			newLabels.append(labels[currentIndex])
		currentIndex += 1  
	return retDataSet,newLabels



def chooseBestFeatureToSplit(dataSet,labels):
	'''
	CHOOSES THE BEST FEATURE FORM THE DATASET TO SPLIT ON AND RETURNS ITS INTEGER LOCATION
	'''
	numFeatures = len(dataSet[0])  #NO OF COLS IN A ROW
	baseEntropy = calcShannonEnt(dataSet,labels)
	bestInfoGain = 0.0 
	bestFeature = -1             #No feature selected yet
	for i in range(numFeatures):
		#Create a list of all values of the column
		featList = [example[i] for example in dataSet]  
		uniqueVals = set(featList)			#Find the unqiue values int the column!
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

dataSet,labels=createDataSet()
print( 'entropy=',calcShannonEnt(dataSet,labels))

print(splitDataSet(dataSet,0,0,labels))
print(splitDataSet(dataSet,0,1,labels))

print(splitDataSet(dataSet,1,0,labels))
print(splitDataSet(dataSet,1,1,labels))

print(splitDataSet(dataSet,2,'yes',labels))
print(splitDataSet(dataSet,2,'no',labels))

print('Best Feature to Split On = ' ,chooseBestFeatureToSplit(dataSet,labels))

