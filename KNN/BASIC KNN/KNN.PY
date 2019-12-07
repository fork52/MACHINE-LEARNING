import numpy as np
import operator

def createDataSet():
	group = np.array([[1.0,1.1],[1.0,1.0],[1.2,0.9],[0,0],[0,0.1],[-0.01,0]])
	labels = ['A','A','A','B','B','B']
	return group, labels

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

myDataSet , myLabels = createDataSet();

ans1 = classify0( [1.2,1.1] ,myDataSet , myLabels ,3 )
ans2 = classify0( [0.2,0.1] ,myDataSet , myLabels ,3 )

print(ans1 , ans2)

