from numpy import *
import string


def loadDataSet():
    dataMat=[]
    labelMat=[]
    fr=open('testSet.txt')
    for line in fr.readlines():
        lineArr=line.strip().split('\t')   #数据里使用制表符分割的
        dataMat.append([1,float(lineArr[0]),float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inx):
    if inx>=0:      #对sigmoid函数的优化，避免了出现极大的数据溢出
        return 1.0/(1+exp(-inx))
    else:
        return exp(inx)/(1+exp(inx))

def grandAscent(dataMatIn,classLabels):
    dataMatrix=mat(dataMatIn)
    labelMat=mat(classLabels).transpose()
    m,n=shape(dataMatrix)   #获得行列数
    alpha=0.001
    maxCycles=500
    weights=ones((n,1))   #weights是一个数组
    for k in range(maxCycles):
        h=sigmoid(dataMatrix*weights)   #矩阵与数组相乘默认转换为矩阵相乘
        error=(labelMat-h)    #所求的函数
         #梯度算法的迭代公式，搞不清的是为什么用dataMatrix.transpose()* error表示该函数的梯度
        weights=weights+alpha*dataMatrix.transpose()*error
    return weights
    
def plotBestFit(weights,w1):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0] 
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig1 = plt.figure()    #开图窗
    ax1 = fig1.add_subplot(111)
    ax1.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax1.scatter(xcord2, ycord2, s=30, c='green')
    #最佳拟合直线
    x1 = arange(-3.0, 3.0, 0.1)
    y1 = (-weights[0]-weights[1]*x1)/weights[2]
    ax1.plot(x1, y1)
    plt.xlabel('X1'); plt.ylabel('X2');
    plt.show()
    fig2 = plt.figure()  
    ax2 = fig2.add_subplot(111)
    ax2.plot(range(len(w1)),w1)
    plt.show()


#随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels): 
    dataMatrix=array(dataMatrix)     ##注意dataMatrix是一个列表，必须先把它转为numpy数组
    m,n = shape(dataMatrix)   #获得行列  
    alpha = 0.01
    weights = ones(n)   #创建了一个一维数组
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))  #数组与数组相乘是对应相乘，h是一个数值
        error = classLabels[i] - h   #error也是一个数值
        weights = weights + alpha * error * dataMatrix[i]
    return weights

#改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = shape(dataMatrix)
    dataMatrix=array(dataMatrix)    ##注意dataMatrix是一个列表，必须先把它转为numpy数组
    weights = ones(n)
    w1=[]
    w1.append(weights[1])
    times=0
    for j in range(numIter):
        dataIndex = list(range(m))   #range不允许删除其内容，所以先得转换为列表
        for i in range(m):
            #alpha每次迭代时都需要调整
            alpha = 4/(1.0+j+i)+0.0001
            #随机选取样本来更新回归系数，将减少周期性的波动
            randIndex = int(random.uniform(0,len(dataIndex)))#go to 0 because of the constant
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            times=times+21
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            w1.append(weights[1])
            del(dataIndex[randIndex])
    return weights





#预测病马的死亡率
def classify0(inX,weights):
    prob=sigmoid(sum(inX*weights))
    if prob>0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt'); frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(trainingSet, trainingLabels,1000)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr =[]
        for i in range(21):
            lineArr.append(float(currLine[i]))
        print(lineArr)
        if int(classify0(array(lineArr), trainWeights))!= int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: {0:f}".format(errorRate))
    return errorRate

def multiTest():
    numTests = 10; errorSum=0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after {0:d} iterations the average error rate is:{1:f}".format(numTests, errorSum/float(numTests)))
