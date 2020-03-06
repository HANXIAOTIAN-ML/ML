from numpy import *
import operator
from os import listdir

def createDataSet():
    group=array([[1.0,1.1],[1,1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    #tile就是复制数组的意思，把inX这个数组复制dataSetSize行，1列
    diffMat=tile(inX,(dataSetSize,1))-dataSet
    #对数组中的每个元素求平方
    sqDiffMat=diffMat**2
    #sum中的axis=0 就是普通的相加，axis=1以后就是将一个矩阵的每一行向量相加
    sqDistances=sqDiffMat.sum(axis=1)
    distances=sqDistances**0.5
    #argsort()函数是将x中的元素从小到大排列，提取其对应的index(索引)，然后输出到y。
    sortedDistIndicies=distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    sortedClassCount=sorted(classCount.items(),
                            key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]

#将文本记录转换到NumPy的解析程序
def file2matrix(filename):
    fr=open(filename)
    arrayOLines=fr.readlines()
    #得到文件行数
    numberOfLines=len(arrayOLines)
    #创建以0填充的矩阵NumPy
    returnMat=zeros((numberOfLines,3))
    classLabelVector=[]
    index=0
    for line in arrayOLines:
        #截取掉所有的回车字符
        line=line.strip()
        #按制表符进行分割
        listFormLine=line.split('\t')
        returnMat[index,:]=listFormLine[0:3]
        classLabelVector.append(int(listFormLine[-1]))
        index+=1
    return returnMat,classLabelVector

#归一化特征值
def autoNorm(dataSet):
    #参数0使得函数可以从列中选取最小值，而不是选取当前行的最小值
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    ranges=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    #y是一个两行三列的二维数组，y.shape[0]代表行数，y.shape[1]代表列数
    m=dataSet.shape[0]
    #minVals和range的值都为1*3，我们使用Numpy库中title()函数将变量复制成输入矩阵相同大小的矩阵
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals


#分类器针对约会网站的测试代码
def datingClassTest():
    hoRatio=0.1
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)
    errorCount=0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with:{0:d},the real answer is:{1:d}'.format(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount+=1
    print('the total error rate is:{0:f}'.format(errorCount/float(numTestVecs)))


#约会网站预测函数
def classifyPerson():
    resultList=['not at all','in small doses','in large doses']
    #原文中的raw_input是在python2中的，在python3中应用input()
    #还有必须使用float进行强制转换，否则和后面的inArr类型不一致
    percentTats=float(input("percentage of time spent playing video games?"))
    ffMiles=float(input("frequent flier miles earned per year?"))
    iceCream=float(input("liters of ice cream consumed per year?"))
    datingDataMat,datingLabels=file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    print(ranges.dtype)
    #python中的list是python的内置数据类型，list中的数据类不必相同的，而array的中的类型必须全部相同。
    #在list中的数据类型保存的是数据的存放的地址，简单的说就是指针，并非数据，这样保存一个list就太麻烦了，例如list1=[1,2,3,'a']需要4个指针和四个数据，增加了存储和消耗cpu。
    #numpy中封装的array有很强大的功能，里面存放的都是相同的数据类型
    #list和array都可以根据索引来取其中的元素。
    #list是列表，list中的元素的数据类型可以不一样。array是数组，数组中的元素的数据类型必须一样。
    #list不可以进行四则运算，array可以进行四则运算。
    #python 中的 list 是 python 的内置数据类型，list 中的数据类型不必相同，
    #在 list 中保存的是数据的存放的地址，即指针，并非数据。
    #array() 是 numpy 包中的一个函数，array 里的元素都是同一类型。ndarray 是一个多维的数组对象，
    #具有矢量算术运算能力和复杂的广播能力，并具有执行速度快和节省空间的特点。ndarray 的一个特点是同构：即其中所有元素的类型必须相同。
    #NumPy 提供的 array() 函数可以将 Python 的任何序列类型转换为 ndarray 数组。例如将一个列表转换成 ndarray 数组
    inArr=array([percentTats,ffMiles,iceCream])
    print(inArr.dtype)
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('the result is {0}'.format(resultList[classifierResult-1]))






#以下是手写数字识别的代码
#将一个32*32的二进制图像矩阵转换为1*1024的向量
def img2vector(filename):
    returnVect=zeros((1,1024))
    fr=open(filename)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect


def handwritingClassTest():
    hwLabels=[]
    #将目录中的文件名存在列表中
    trainingFileList=listdir('trainingDigits')
    m=len(trainingFileList)
    #创建m行1024列的训练矩阵，矩阵每一行存储一个图像的信息
    trainingMat=zeros((m,1024))
    for i in range(m):
        #将训练数据的标签存储
        fileNameStr=trainingFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)
        trainingMat[i,:]=img2vector('trainingDigits/%s'%fileNameStr)
    testFileList=listdir('testDigits')
    errorCount=0.0
    mTest=len(testFileList)
    for i in range(mTest):
        fileNameStr=testFileList[i]
        fileStr=fileNameStr.split('.')[0]
        classNumStr=int(fileStr.split('_')[0])
        vectorUnderTest=img2vector('testDigits/%s'%fileNameStr)
        classifierResult=classify0(vectorUnderTest,trainingMat,hwLabels,3)
        print('the classifier came back with:{0:d},the real answer is:{1:d}'.format(classifierResult,classNumStr))
        if(classifierResult!=classNumStr):
            errorCount+=1
    print("the total number of errors is:{0:f}".format(errorCount))
    print("the total error rate is :{0}".format(errorCount/float(mTest)))

