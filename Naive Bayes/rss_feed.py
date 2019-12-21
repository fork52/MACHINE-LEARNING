import feedparser
import email_classify
import random
import numpy as np
import pprint as pr


def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def calcMostFreq(vocabList,fullText):
    '''
    RETURNS A LIST HAVING TOP 30 MOST OCCURING WORDS IN fullText
     index 0 - WORD OCCURING IN THE VOCABLIST
     index 1 -  NO. OF OCCURENCES OF WORD IN fullText
    '''
    import operator
    freqDict = {}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq = sorted(freqDict.items(), key=operator.itemgetter(1), reverse=True) 
    return sortedFreq[:30]       

def localWords(feed1,feed0):
    import feedparser
    docList=[]
    classList = []
    fullText =[]

    minLen = min(len(feed1['entries']),len(feed0['entries']))


    for i in range(minLen):
        "Load the words"
        wordList = email_classify.textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1) #NY is class 1
        wordList = email_classify.textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    
    vocabList = email_classify.createVocabList(docList)#create vocabulary
    top30Words = calcMostFreq(vocabList,fullText)   #remove top 30 words

    for pairW in top30Words:
        if pairW[0] in vocabList: 
            vocabList.remove(pairW[0])

    trainingSet = list(range(2*minLen))
    testSet=[]           #create test set
    for i in range(20):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])  

    trainMat=[]; trainClasses = []
    for docIndex in trainingSet:#train the classifier (get probs) trainNB0
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = email_classify.trainNB0(np.array(trainMat),np.array(trainClasses))
    errorCount = 0

    for docIndex in testSet:        #classify the remaining items
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if email_classify.classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount += 1

    print('the error rate is: ',float(errorCount)/len(testSet) )
    return vocabList,p0V,p1V

if __name__ == '__main__':
    feed1= feedparser.parse('https://www.feedspot.com/infiniterss.php?followfeedid=117&q=site:http%3A%2F%2Frss.cnn.com%2Frss%2Fcnn_topstories.rss')
    feed2 = feedparser.parse('http://www.independent.co.uk/news/world/rss')
    
    # text_file = open("sample3.txt", "w")
    # n = text_file.write(pr.pformat(str(feed1)) )
    # text_file.close()
    # print( len(feed1))
    # print(feed1.keys())
    vocabList,pSF,pNY=localWords(feed1,feed2)