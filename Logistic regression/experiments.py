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
    alpha = 0.003   # learning rate
    maxCycles = 1000 # the number of times you’re going to repeat the calculation before stopping

    #INITIALLY ALL WEIGHTS ARE SET TO 1
    weights = np.ones((n,1))
    allOnes = np.mat(np.ones( np.shape(labelMat)))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix*weights)   
        error = (labelMat - h)
        '''
        Qualitatively you can see we’re calculating the error between the actual class
        and the predicted class and then moving in the direction of that error
        '''
        oneMinusFunc = allOnes -  h 
        temp = np.mat(np.array(oneMinusFunc)*np.array(h))        
        temp2 = np.mat(np.array(error)*np.array(temp))        

        #The multiplication dataMatrix * weights is not one multiplication but actually no of multiplications.
        #This is the line where u actually use gradient ascent formula
        weights = weights + alpha * dataMatrix.transpose()* temp2
 
    return weights


def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei.getA() 
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)

    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')

    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)

    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()


if __name__ == '__main__':
    dataSet,labels = loadDataSet()
    optimized_coefficients = gradAscent(dataSet , labels)

    pr.pprint(dataSet[:10])
    pr.pprint(labels[:10])
    pr.pprint(optimized_coefficients)

    plotBestFit(optimized_coefficients)

   