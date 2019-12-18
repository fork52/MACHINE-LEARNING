import numpy as np
import pprint as pr
import math 

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
    ''' RETURNS A UNIQUE LIST OF WORDS IN THE VOCABULARY'''
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
    p0Vect is a vector of probability of occurences of word given that the sentence is non-abusive
    p1Vect is a vector of probability of occurences of word given that sentence is non-abusive

    '''
    numTrainDocs = len(trainMatrix)   #no of sentences
    numWords     = len(trainMatrix[0]) # no of words in vocabulary list

    # sum(trainCategory) will return the no of ones in the labels i.e the no of
    # abusive documents/lines in the matrix.
    pAbusive = sum(trainCategory)/float(numTrainDocs)

    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords) #THESE ARE VECTORS!!

    '''
    We set intial count to all 1's and not 0's bcoz if we keep all 0's then if the probability 
    for even a single word is 0 then when we multiply the 
    probability the answer will become 0 even if there is one 0 in probability vector
    '''
    p0Denom = 2.0
    p1Denom = 2.0


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
    p1Vect = np.log(p1Num/p1Denom)
    p0Vect = np.log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    THE FUNCTION WHICH DOES THE ACTUAL CLASSIFICATION BASED ON THE probability 
    vectors genereted by trainNB0

    vec2Classify is a vocabulary list format having 0 or 1 indicating presence of words and
    importantly it shud be a np.array.
    pClass1 is the probability of the vector being abusive
    '''

    #WE ARE USING ADDITION/SUM BCOZ WE ARE USING log to store probabilities
    #The multiplication is element–wise
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def testingNB():
    '''
    JUST A FUNCTION TO TEST WHETHER THE TRAINED MATRIX IS OF ANY USE :) 
    '''
    listOPosts,listClasses = loadDataSet()
    myVocabList = createVocabList(listOPosts)
    trainMat=[]

    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))

    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) )

    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb) )

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
    # print( p0Vect , p1Vect ,probAbusive,sep='\n\n')
    testingNB()