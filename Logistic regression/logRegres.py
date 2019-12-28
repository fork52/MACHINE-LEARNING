import numpy as np
import pprint as pr

def loadDataSet():
    dataMat = []; labelMat = []
    fr = open('Logistic regression/testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat


def sigmoid(inX):
    return 1.0/(1+ np.exp(-inX))

def gradAscent(dataMatIn, classLabels):
    '''
    Optimization algorithm
    The first input, dataMatIn, is a 2D NumPy array, where the columns are 
    the different features and the rows are the different training examples. 

    Returns the vector matrix of optimized coefficients for the algorithm
    '''
    dataMatrix = np.mat(dataMatIn)
    #for the matrix math to work, you need it to be a column vector, so youtake the transpose of it
    labelMat = np.mat(classLabels).transpose()

    # NumPy can operate on both 2D arrays and matrices, and the results will be different if you assume the wrong data type.
    m,n = np.shape(dataMatrix)     # n - no of features in the dataset
    alpha = 0.001   # learning rate
    maxCycles = 500 # the number of times you’re going to repeat the calculation before stopping

    #INITIALLY ALL WEIGHTS ARE SET TO 1
    weights = np.ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)   
        error = (labelMat - h)
        #The multiplication dataMatrix * weights is not one multiplication but actually n.
        weights = weights + alpha * dataMatrix.transpose()* error
    return weights


if __name__ == '__main__':
    dataSet,labels = loadDataSet()
    optimized_coefficients = gradAscent(dataSet , labels)
    pr.pprint(dataSet[:10])
    pr.pprint(labels[:10])
    pr.pprint(optimized_coefficients)
    pass