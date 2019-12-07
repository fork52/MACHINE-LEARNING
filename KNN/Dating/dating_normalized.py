import numpy as np
import pprint as pr
import matplotlib as mpl
import matplotlib.pyplot as plt

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

dataSet,labelVector=file2matrix('datingTestSet2.txt')

normMat , dataRanges , minValues = autoNorm(dataSet)



print(normMat[:10],dataRanges,minValues,sep='\n\n')

