import re
import random
import numpy as np

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
     2) trainCategory - labels for the sentences . (spam - 1 or not spam -0)

    OUTPUTS:
    probability vectors p0Vect and p1Vect and value pSpam
    p0Vect is a vector of probability of occurences of word given that the sentence is non-spam
    p1Vect is a vector of probability of occurences of word given that sentence is spam

    '''
    numTrainDocs = len(trainMatrix)   #no of sentences
    numWords     = len(trainMatrix[0]) # no of words in vocabulary list

    # sum(trainCategory) will return the no of ones in the labels i.e the no of
    # spam documents/lines in the matrix.
    pSpam = sum(trainCategory)/float(numTrainDocs)
    print(pSpam)

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
    return p0Vect,p1Vect,pSpam

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    '''
    THE FUNCTION WHICH DOES THE ACTUAL CLASSIFICATION BASED ON THE probability 
    vectors genereted by trainNB0

    vec2Classify is a vocabulary list format having 0 or 1 indicating presence of words and
    importantly it shud be a np.array.
    pClass1 is the probability of the vector being spam
    '''
    #WE ARE USING ADDITION/SUM BCOZ WE ARE USING log to store probabilities
    #The multiplication is element–wise
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1 # spam
    else:
        return 0

def textParse(bigString):
    listOfTokens = re.split('[\\W*]', bigString)
    # print( [tok.lower() for tok in listOfTokens if len(tok) > 2])
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]        # CONTENT LIST
    classList = []    # LABELS LIST 
    fullText =[]

    for i in range(1,26):
        '''READING THE FILES IN A LOOP'''
        wordList = textParse(open('Naive Bayes/email/spam/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)

        wordList = textParse(open('Naive Bayes/email/ham/%d.txt' % i).read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)

    vocabList = createVocabList(docList)
    trainingSet = list(range(50))
    testSet=[]

    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)) )
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])

    print('No. of training data pieces:',len(trainingSet))
    print('No. of training data pieces:',len(testSet))



    trainMat=[]
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])

    p0V,p1V,pSpam = trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0

    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1
    print("errorCount =" ,errorCount )
    print('the error rate is: ',float(errorCount)/len(testSet)  )


if __name__ == '__main__':
    spamTest()