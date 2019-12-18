import numpy as np
import pprint as pr

def loadDataSet():
    postingList=[
        ['my', 'dog', 'has', 'flea','problems', 'help', 'please'],
        ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
        ['my', 'dalmation', 'is', 'so', 'cute','I', 'love', 'him'],
        ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
        ['mr', 'licks', 'ate', 'my', 'steak', 'how','to', 'stop', 'him'],
        ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
    ]
    classVec = [0,1,0,1,0,1] #1 is abusive, 0 not
    return postingList,classVec


def createVocabList(dataSet):
    ''' RETURNS A LIST OF WORDS IN THE VOCABULARY'''
    vocabSet = set([]) # EMPTY SET
    for document in dataSet:
        # 'UNION' SET OPERATION - |
        vocabSet = vocabSet | set(document)   
    return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    ''' 
    vocabList is the list of unique words in vocabulary 
    inputSet is the list of words to be tested
    '''
    returnVec = [0] * len(vocabList)  # [0,0,0,.......n]
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word) ] = 1  # WORD PRESENT
        else: 
            print("the word: %s is not in my Vocabulary!" %word )
    return returnVec

def trainNB0(trainMatrix,trainCategory):
    '''
    Function train Naive Bayes 
    INPUTS:
     1) trainMatrix   - dataSet which is a list of lists.
        Innerlist is a vocabulary vector which has 0 or 1 for corresponding 
        words in the vocabulary list
     2) trainCategory - labels for the sentences . (abusive - 1 or not abusive -0)
    OUTPUTS:
        probability vectors p0Vect and p1Vect and value pAbusive

    '''
    numTrainDocs = len(trainMatrix)   #no of sentences
    numWords     = len(trainMatrix[0]) # no of words in vocabulary list

    # sum(trainCategory) will return the no of ones in the labels i.e the no of
    # abusive documents/lines in the matrix.
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    p0Num = np.zeros(numWords)  #THESE ARE VECTORS!!
    p1Num = np.zeros(numWords)

    p0Denom = 0.0
    p1Denom = 0.0

    for i in range(numTrainDocs):
        if trainCategory[i] == 1: #line or doc is abusive

            # add the entire corresponding vocabulary list vector to p1Num
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i]) # add the no of words present in line to the den

        else:            #line or doc is non-abusive

            # add the entire corresponding vocabulary list to p0Num
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])

    '''
    at the end of the loop , 
    p0Denom will be the no. of words belonging to data pieces in the training set having non-abusive label
    p1Denom will be the no. of words belonging to data pieces in the training set having abusive label
    '''
    p1Vect = p1Num/p1Denom     #change to log()
    p0Vect = p0Num/p0Denom     #change to log()
    return p0Vect,p1Vect,pAbusive


if __name__ == '__main__':
    dataSet , labels = loadDataSet()
    myVocabList = createVocabList(dataSet)

    vocVector = setOfWords2Vec( myVocabList , dataSet[0] )

    print(myVocabList)
    # print(vocVector)
    trainMat = []
    for line in dataSet:
        trainMat.append( setOfWords2Vec(myVocabList , line) )
    
    p0Vect,p1Vect,probAbusive =  trainNB0(trainMat , labels)
    print( p0Vect , p1Vect ,probAbusive,sep='\n\n')

