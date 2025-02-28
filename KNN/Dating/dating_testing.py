import numpy as np
import pprint as pr
import matplotlib as mpl
import matplotlib.pyplot as plt
import operator

def file2matrix(filename):
	'''Function that takes a filename string as input and returns a nx3 matrix and a n-label vector '''

	fr = open(filename)
	numberOfLines = len(fr.readlines())			#find the no. of lines
	returnMat = np.zeros((numberOfLines,3))     
	classLabelVector = []

	fr = open(filename)													
	index = 0						#start from line 0 in the list
	for line in fr.readlines():
		line = line.strip() 			#Remove \n from every line
		listFromLine = line.split('\t')
		returnMat[index,:] = listFromLine[0:3]				#first three elements are 
		classLabelVector.append(int(listFromLine[-1]))      #last element in a line is label
		index += 1
	return returnMat,classLabelVector

def autoNorm(dataSet):
	minVals = dataSet.min(0)
	maxVals = dataSet.max(0)
	ranges = maxVals - minVals
	normDataSet = np.zeros(np.shape(dataSet))
	m = dataSet.shape[0]
	normDataSet = dataSet - np.tile(minVals, (m,1))
	normDataSet = normDataSet/np.tile(ranges, (m,1))
	return normDataSet, ranges, minVals
#REFER TO NORMALIZATION PDF IN FOLDER FOR UNDERSTANDING THE FUNCTION


def classify0(inX, dataSet, labels, k):
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]


def datingClassTest():
	hoRatio = 0.10  # 10% of the data will be used for testing and 90% for KNN comparison
	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
	normMat, ranges, minVals = autoNorm(datingDataMat) #Normalized Data
	m = normMat.shape[0]             #TOTAL NO OF ROWS IN THE DATASET
	numTestVecs = int(m*hoRatio)    #FIRST 10 VECTORS WILL BE USED FOR TESTING
	errorCount = 0.0
	for i in range(numTestVecs):
		classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
		print("the classifier came back with: %d, the real answer is: %d"% (classifierResult, datingLabels[i]))
		if (classifierResult != datingLabels[i]): 
			errorCount += 1.0
	print("the total error rate is: %f" % (errorCount/float(numTestVecs)) )


datingClassTest()