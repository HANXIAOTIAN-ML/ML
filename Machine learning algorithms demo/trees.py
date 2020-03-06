from math import log
import operator

def calcShannonEnt(dataSet):
    numEntries=len(dataSet)
    labelCounts={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel]=0
        labelCounts[currentLabel]+=1
    shannonEnt=0.0
    for key in labelCounts:
        prob=float(labelCounts[key])/numEntries
        shannonEnt-=prob*log(prob,2)
    return shannonEnt


def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels


def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            #其实就是把axis那一列去掉
            reducedFeatVec = featVec[:axis]     #chop out axis used for splitting
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures=len(dataSet[0])-1
    #整个数据集的原始香农熵
    baseEntropy=calcShannonEnt(dataSet)
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):
        #列表推导创建新列表
        featList = [example[i] for example in dataSet]
        #转换成集合，从列表中创建集合是python语言得到列表中唯一元素值最快的方法
        uniqueVals = set(featList)       
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)
        #信息增益，信息增益是熵的减少或者是数据无序度的减少，将熵用于度量数据无序度的减少更容易理解
        infoGain = baseEntropy - newEntropy    
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain         
            bestFeature = i
    return bestFeature                      

def majorityCnt(classList):
    #字典存储了classList中每个类标签出现的频率
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    #根据键值对，按第二域进行排序，逆序
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    #返回出现次数最多的分类名称
    return sortedClassCount[0][0]


def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #递归函数的第一个停止条件：如果所有的类标签完全相同，则直接返回数据集的类标签
    if classList.count(classList[0]) == len(classList): 
        return classList[0]
    #第二个停止条件：使用完了所有特征，仍然不能把数据集划分成仅包含唯一类别的分组
    if len(dataSet[0]) == 1: 
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    #一定要注意这里把labels中的一项给删掉了，列表传递的是引用，所以相当于把原值改变了
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree 


def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel


#将分类器存储在硬盘上，而不用每次对数据分类时重新学习一遍
def storeTree(inputTree,filename):
    import pickle
    #pickle存储方式默认是二进制方式
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
    
def grabTree(filename):
    import pickle
    #序列化操作时，文件模式不正确，改为“rb+”，即可
    fr = open(filename,'rb+')
    return pickle.load(fr)
    






#操作
#>>> import trees
#>>> fr=open('lenses.txt')
#>>> lenses=[inst.strip().split('\t') for inst in fr.readlines()]
#>>> lenseLabels=['age','prescript','astigmatic','tearRate']
#>>> labels=lenseLabels
#>>> lensesTree=trees.createTree(lenses,lenseLabels)
#>>> trees.storeTree(lensesTree,'classifierStorage.txt')
#>>> trees.grabTree('classifierStorage.txt')
#>>> trees.classify(trees.grabTree('classifierStorage.txt'),['age','prescript','astigmatic','tearRate'],['pre','myope','n0','normal'])
#'soft'

