import numpy as np
import pprint as pr
import matplotlib as mpl
import matplotlib.pyplot as plt
import operator

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



def classifyPerson():
	resultList = ['not at all','in small doses', 'in large doses']

	percentTats = float(input("percentage of time spent playing video games?"))
	ffMiles = float(input("frequent flier miles earned per year?"))
	iceCream = float(input("liters of ice cream consumed per year?"))

	datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')

	normMat, ranges, minVals = autoNorm(datingDataMat)
	inArr = np.array([ffMiles, percentTats, iceCream])     # Putting i/p data into array

	#CALLING THE CLASSIFIER
	classifierResult = classify0((inArr-minVals)/ranges,normMat,datingLabels,3)


	print("You will probably like this person: ",resultList[classifierResult - 1])

classifyPerson()