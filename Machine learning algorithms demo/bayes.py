from numpy import *

def loadDataSet():
    #词条切分后的文档集合
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    #类别标签的集合，标注信息用于训练程序一遍自动监测侮辱性语言
    classVec = [0,1,0,1,0,1]    #1 代表侮辱性词语, 0 代表正常言论
    return postingList,classVec

#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])  #create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document) #求并集
    #最后返回的是一个大列表
    return list(vocabSet)


#词集模型
def setOfWords2Vec(vocabList, inputSet):#输入的是词汇表以及某个文档
    returnVec = [0]*len(vocabList)    #创建一个所含元素都为0的向量
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else: print("the word:{0} is not in my Vocabulary!".format(word))
    return returnVec


#朴素贝叶斯分类器训练函数,这是啥意思，其实就是给一堆文章，这些文章的所属类别已知，然后学习出各个词语在不同类中出现的频率
#第一个参数是由很多文档组成的文档矩阵，矩阵中的每一行元素都是一个包含0,1的列表，第二个参数是这些文档所属类别标签组成的向量
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)      #计算有多少片文档
    numWords = len(trainMatrix[0])       #计算第一篇文档里有多少个词
    pAbusive = sum(trainCategory)/float(numTrainDocs)     #计算侮辱性文档的概率
    p0Num = zeros(numWords); p1Num = zeros(numWords)      #初始化，P(w|c)中的分母变量是一个元素个数等于词汇表大小的NumPy数组
    p0Denom = 0.0; p1Denom = 0.0                        #
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:   #如果这篇文章是侮辱性文档
            p1Num += trainMatrix[i]   #所有侮辱性文章中各个词语出现的频数
            p1Denom += sum(trainMatrix[i])  #所有侮辱性文章中的词语个数求和
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #相当于求出来了已知所属类别c时，各个特征出现的频率
    p1Vect = p1Num/p1Denom        #侮辱性文章中各个词语出现的频率，利用NumPy实现了数组除以浮点数
    p0Vect = p0Num/p0Denom         #正常文章中各个词语出现的频率
    return p0Vect,p1Vect,pAbusive    #返回的是正常和侮辱性文章中各个词（特征）出现的频率，以及侮辱性文档的概率，
                                      #问题是咋判断一篇文章属于侮辱性文章？

"""
import bayes
>>> listOPosts,listClasses=bayes.loadDataSet()
>>> myVocabList=bayes.createVocabList(listOPosts)
>>> trainMat=[]
>>> for postinDoc in listOPosts:
	trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))

>>> poV,p1V,pAb=bayes.trainNB0(trainMat,listClasses)
>>> p1V
array([0.        , 0.        , 0.05263158, 0.        , 0.        ,
       0.05263158, 0.05263158, 0.05263158, 0.        , 0.10526316,
       0.        , 0.05263158, 0.10526316, 0.05263158, 0.        ,
       0.        , 0.        , 0.15789474, 0.        , 0.05263158,
       0.        , 0.05263158, 0.        , 0.        , 0.05263158,
       0.05263158, 0.        , 0.        , 0.        , 0.        ,
       0.05263158, 0.05263158])
>>> myVocabList
['help', 'love', 'food', 'ate', 'I', 'to', 'stop', 'buying', 'cute', 'dog', 'flea', 'posting', 'worthless',
'him', 'licks', 'steak', 'dalmation', 'stupid', 'is', 'park', 'my', 'quit', 'please', 'problems', 'take', 'garbage', 'so', 'mr', 'how', 'has', 'not', 'maybe']

"""

#修改后的代码
#第一个参数是由很多文档组成的文档矩阵，矩阵中的每一行元素都是一个包含0,1的列表，第二个参数是这些文档所属类别标签组成的向量
def trainNB1(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)      #计算有多少片文档
    numWords = len(trainMatrix[0])       #计算第一篇文档里有多少个词
    pAbusive = sum(trainCategory)/float(numTrainDocs)     #计算侮辱性文档的概率
    p0Num = ones(numWords); p1Num = ones(numWords)      #初始化，P(w|c)中的分母变量是一个元素个数等于词汇表大小的NumPy数组
    p0Denom = 2.0; p1Denom = 2.0                        #
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:   #如果这篇文章是侮辱性文档
            p1Num += trainMatrix[i]   #所有侮辱性文章中各个词语出现的频数
            p1Denom += sum(trainMatrix[i])  #所有侮辱性文章中的词语个数求和
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #相当于求出来了已知所属类别c时，各个特征出现的频率
    p1Vect = log(p1Num/p1Denom)        #侮辱性文章中各个词语出现的频率，利用NumPy实现了数组除以浮点数
    p0Vect = log(p0Num/p0Denom)         #正常文章中各个词语出现的频率
    return p0Vect,p1Vect,pAbusive    #返回的是正常和侮辱性文章中各个词（特征）出现的频率，以及侮辱性文档的概率，
                                      #问题是咋判断一篇文章属于侮辱性文章？


#朴素贝叶斯分类函数
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
    listOPosts,listClasses = loadDataSet()
    #词语集合
    myVocabList = createVocabList(listOPosts)
    trainMat=[]
    #集合中的词语在各篇文章中的出现情况
    for postinDoc in listOPosts:
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #返回的是正常和侮辱性文章中各个词（特征）出现的频率，以及侮辱性文档的概率
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    testEntry = ['love', 'my', 'dalmation']
    #集合中的词语在testEntry中的出现情况,只记录其是否在测试样本中出现，而不记录出现的次数
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    testEntry = ['stupid', 'garbage']
    thisDoc = array(setOfWords2Vec(myVocabList, testEntry))
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))


#将之前的词集模型改为词袋模型（朴素贝叶斯词袋模型）
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


def textParse(bigString):    #input is big string, #output is word list
    import re
    listOfTokens = re.split(r'\W+', bigString,re.A)
    #去掉长度少于两个字符的字符串，并都转换为小写
    return [tok.lower() for tok in listOfTokens if len(tok)] 
    
def spamTest():
    #重复进行10次试验
    errorCount = list(range(10))
    errorrate=0.0
    for index in range(10):
        errorCount[index] = 0
        docList=[]; classList = []; fullText =[]
        for i in range(1,26):           
        #一篇好的，一篇坏的
            wordList = textParse(open('email/spam/{0}.txt'.format(i)).read())
            docList.append(wordList)
            fullText.extend(wordList)
            classList.append(1)
            wordList = textParse(open('email/ham/{0}.txt'.format(i)).read())
            docList.append(wordList)
            fullText.extend(wordList) 
            classList.append(0)
        #创建了一个大列表
        vocabList = createVocabList(docList)#create vocabulary
        trainingSet = list(range(50)); testSet=[]
        ################留存交叉验证###############
        #创建训练集
        for i in range(10):
            #返回的是一个随机的浮点数，在0——50内，取整
            randIndex = int(random.uniform(0,len(trainingSet)))
            #把索引添加到测试集中
            testSet.append(trainingSet[randIndex])
            #将索引从训练集中删除
            del(trainingSet[randIndex])
            #将训练集中每一篇文章中词语出现的次数进行统计，返回实际用于训练的训练集（构建词向量）
            trainMat=[]; trainClasses = []
        for docIndex in trainingSet:#train the classifier (get probs) trainNB0
            trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
            trainClasses.append(classList[docIndex])
        p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))
    #对测试集中的每封邮件进行分类，统计分类错误数
        for docIndex in testSet:
            #构建用于测试的邮件的词向量
            wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
            if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
                errorCount[index] += 1
                print("classification error",docList[docIndex])
        errorrate+=float(errorCount[index])/len(testSet)
        print ('the error rate is: ',float(errorCount[index])/len(testSet))
        #return vocabList,fullText
    #输出10次试验的平均错误率
    print('average error rate is :',errorrate/10 )
