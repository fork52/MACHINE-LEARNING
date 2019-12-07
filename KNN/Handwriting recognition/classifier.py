import numpy as np
import pprint as pr
import os
import operator

def classify0(inX, dataSet, labels, k):
	'''
	THIS IS A GENERALIZED KNN CLASSIFIER 
	IT ASSUME THAT THE DATASET IS NORMALIZED DATASET
	'''
	dataSetSize = dataSet.shape[0]
	diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
	sqDiffMat = diffMat**2
	sqDistances = sqDiffMat.sum(axis=1)
	distances = sqDistances**0.5
	sortedDistIndicies = distances.argsort()
	print(sortedDistIndicies)
	classCount={}
	for i in range(k):
		voteIlabel = labels[sortedDistIndicies[i]]
		classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
	sortedClassCount = sorted(classCount.items(),key=operator.itemgetter(1), reverse=True)
	return sortedClassCount[0][0]

def img2vector(filename):
	'''Opens a single txt file having a 32x32 digit and converts it into 1x1024 numpp array'''
	returnVect = np.zeros((1,1024))
	fr = open(filename)
	for i in range(32):
		lineStr = fr.readline()
		for j in range(32):
			returnVect[0,32*i+j] = int(lineStr[j])
	return returnVect

def handwritingClassTest():
	hwLabels = []				# handwriting labels list for each text file

	trainingFileList = os.listdir('trainingDigits') #For getting list of textfiles

	m = len(trainingFileList) # m- no. of training files

	trainingMat = np.zeros((m,1024))  #Training matrix with m examples with labels

	for i in range(m):
		#The filename itself will have the label for the training data piece
		fileNameStr = trainingFileList[i]       #get the filename
		fileStr = fileNameStr.split('.')[0]		
		classNumStr = int(fileStr.split('_')[0]) #the label is in the fileStr
		hwLabels.append(classNumStr)
		trainingMat[i,:] = img2vector('trainingDigits/%s' % fileNameStr)

	testFileList = os.listdir('testDigits')  #Open the testing dataset
	errorCount = 0.0
	mTest = len(testFileList) 

	print(np.shape(trainingMat) , trainingMat.shape)

	for i in range(mTest):
		"Pick up one textFile from the testing list and test it against the exisiting data for k neighbors"
		fileNameStr = testFileList[i]
		fileStr = fileNameStr.split('.')[0]
		classNumStr = int(fileStr.split('_')[0])
		vectorUnderTest = img2vector('testDigits/%s' % fileNameStr)  #current test vector
		classifierResult = classify0(vectorUnderTest,trainingMat, hwLabels, 3)
		print("the classifier came back with: %d, the real answer is: %d"% (classifierResult, classNumStr) )
		if (classifierResult != classNumStr): 
			errorCount += 1.0
			
	print("\nthe total number of errors is: %d" % errorCount)
	print("\nthe total error rate is: %f" % (errorCount/float(mTest)) )

handwritingClassTest()