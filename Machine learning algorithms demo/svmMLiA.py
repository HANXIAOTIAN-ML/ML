'''
Created on Nov 4, 2010
Chapter 5 source file for Machine Learing in Action
@author: Peter
'''
from numpy import *
from time import sleep

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

def selectJrand(i,m):
    j=i #函数随机选择一个不等于i的值返回
    while (j==i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):   #用于调整aj
    if aj > H: 
        aj = H   #当aj太大时，把它降下来，当aj太小时，把它升上去
    if L > aj:
        aj = L
    return aj

def smoSimple(dataMatIn, classLabels, C, toler, maxIter):   #常数C、容错率、最大循环次数
    dataMatrix = mat(dataMatIn); labelMat = mat(classLabels).transpose()
    b = 0; m,n = shape(dataMatrix)
    alphas = mat(zeros((m,1)))    #alpha是一个m行1列的矩阵
    iter = 0    #遍历数据集的次数，达到maxIter时函数结束并退出
    while (iter < maxIter):
        alphaPairsChanged = 0    #用于记录alpha是否已经优化
        for i in range(m):
            fXi = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[i,:].T)) + b     #.T和transpose()一样都是求矩阵转置   [i,:]取矩阵的那一行
            Ei = fXi - float(labelMat[i])   #计算误差
            if ((labelMat[i]*Ei < -toler) and (alphas[i] < C)) or ((labelMat[i]*Ei > toler) and (alphas[i] > 0)):   #如果误差很大，可以对该数据实例所对应的alpha值进行优化
                j = selectJrand(i,m)   #随机选择第二个alpha
                fXj = float(multiply(alphas,labelMat).T*(dataMatrix*dataMatrix[j,:].T)) + b
                Ej = fXj - float(labelMat[j])  #计算这个alpha的误差
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();#之后可以将新的alpha和老的alpha进行比较
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])    #L/H用于将alpha[j]调整到0和C之间
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H:
                    print("L==H")
                    continue
                #eta是最优修改量
                eta = 2.0 * dataMatrix[i,:]*dataMatrix[j,:].T - dataMatrix[i,:]*dataMatrix[i,:].T - dataMatrix[j,:]*dataMatrix[j,:].T
                if eta >= 0:   #简化处理
                    print("eta>=0")
                    continue
                #同时优化这两个向量
                alphas[j] -= labelMat[j]*(Ei - Ej)/eta   #修改后的alphas[j]
                alphas[j] = clipAlpha(alphas[j],H,L)     #对alphas[j]进行调整
                if (abs(alphas[j] - alphaJold) < 0.00001):   #检查alphas[j]是否有轻微改变，改变很小，开始下次循环
                    print("j not moving enough")
                    continue
                alphas[i] += labelMat[j]*labelMat[i]*(alphaJold - alphas[j])#alphas[i]同样进行改变，改变的与alphas[j]大小相同，方向相反
                #给两个alpha设置一个常数b
                b1 = b - Ei- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[i,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[i,:]*dataMatrix[j,:].T
                b2 = b - Ej- labelMat[i]*(alphas[i]-alphaIold)*dataMatrix[i,:]*dataMatrix[j,:].T - labelMat[j]*(alphas[j]-alphaJold)*dataMatrix[j,:]*dataMatrix[j,:].T
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2)/2.0        
                alphaPairsChanged += 1
                print("iter: %d i:{0:d}, pairs changed {1:d}".format(iter,i,alphaPairsChanged))
        if (alphaPairsChanged == 0):
            iter += 1
        else:
            iter = 0
        print("iteration number: {0:d}".format(iter))
    return b,alphas


class optStruct:
    def __init__(self,dataMatIn, classLabels, C, toler, kTup):  
        self.X = dataMatIn
        self.labelMat = classLabels
        self.C = C
        self.tol = toler
        self.m = shape(dataMatIn)[0]
        self.alphas = mat(zeros((self.m,1)))
        self.b = 0
        #误差缓存
        self.eCache = mat(zeros((self.m,2))) #第一列给出的是ecache是否有效的标志位,第二列给出的是实际的e值
        self.K = mat(zeros((self.m,self.m)))

#计算E值并返回
def calcEk(oS, k):
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.X[:,k]+ oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
        
def selectJ(i, oS, Ei):         #选择合适的第二个alpha，以保证在每次优化中采用最大步长
    maxK = -1; maxDeltaE = 0; Ej = 0
    oS.eCache[i] = [1,Ei]  #set valid #choose the alpha that gives the maximum delta E
    validEcacheList = nonzero(oS.eCache[:,0].A)[0]
    if (len(validEcacheList)) > 1:
        for k in validEcacheList:   #loop through valid Ecache values and find the one that maximizes delta E
            if k == i: continue #don't calc for i, waste of time
            Ek = calcEk(oS, k)
            deltaE = abs(Ei - Ek)
            if (deltaE > maxDeltaE):
                maxK = k; maxDeltaE = deltaE; Ej = Ek
        return maxK, Ej
    else:   #in this case (first time around) we don't have any valid eCache values
        j = selectJrand(i, oS.m)
        Ej = calcEk(oS, j)
    return j, Ej

def updateEk(oS, k):#after any alpha has changed update the new value in the cache
    Ek = calcEk(oS, k)
    oS.eCache[k] = [1,Ek]
        
def innerL(i, oS):
    Ei = calcEk(oS, i)#计算误差
    if ((oS.labelMat[i]*Ei < -oS.tol) and (oS.alphas[i] < oS.C)) or ((oS.labelMat[i]*Ei > oS.tol) and (oS.alphas[i] > 0)):#如果误差很大，可以对该数据实例所对应的alpha值进行优化
        j,Ej = selectJ(i, oS, Ei)       #选择第二个alpha
        alphaIold = oS.alphas[i].copy(); alphaJold = oS.alphas[j].copy();
        if (oS.labelMat[i] != oS.labelMat[j]):  #L/H用于将alpha[j]调整到0和C之间
            L = max(0, oS.alphas[j] - oS.alphas[i])
            H = min(oS.C, oS.C + oS.alphas[j] - oS.alphas[i])
        else:
            L = max(0, oS.alphas[j] + oS.alphas[i] - oS.C)
            H = min(oS.C, oS.alphas[j] + oS.alphas[i])
        if L==H:
            print("L==H")
            return 0
        #eta是最优修改量
        eta = 2.0 * oS.K[i,j] - oS.K[i,i] - oS.K[j,j]
        if eta >= 0:
            print("eta>=0")
            return 0
        oS.alphas[j] -= oS.labelMat[j]*(Ei - Ej)/eta     #修改后的alphas[j]
        oS.alphas[j] = clipAlpha(oS.alphas[j],H,L)       #对alphas[j]进行调整
        updateEk(oS, j)   #更新误差缓存
        if (abs(oS.alphas[j] - alphaJold) < 0.00001):      #检查alphas[j]是否有轻微改变，改变很小，开始下次循环
            print("j not moving enough")
            return 0
        oS.alphas[i] += oS.labelMat[j]*oS.labelMat[i]*(alphaJold - oS.alphas[j])    #alphas[i]同样进行改变，改变的与alphas[j]大小相同，方向相反
        updateEk(oS, i) #更新误差缓存                 #the update is in the oppostie direction
        b1 = oS.b - Ei- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,i] - oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[i,j]
        b2 = oS.b - Ej- oS.labelMat[i]*(oS.alphas[i]-alphaIold)*oS.K[i,j]- oS.labelMat[j]*(oS.alphas[j]-alphaJold)*oS.K[j,j]
        if (0 < oS.alphas[i]) and (oS.C > oS.alphas[i]):
            oS.b = b1
        elif (0 < oS.alphas[j]) and (oS.C > oS.alphas[j]):
            oS.b = b2
        else:
            oS.b = (b1 + b2)/2.0
        return 1
    else:
        return 0


#完整版Platt SMO外循环
def smoP(dataMatIn, classLabels, C, toler, maxIter,kTup=('lin', 0)):    #full Platt SMO
    oS = optStruct(mat(dataMatIn),mat(classLabels).transpose(),C,toler, kTup)
    iter = 0
    entireSet = True; alphaPairsChanged = 0     #对控制函数退出的一些变量进行初始化  
    while (iter < maxIter) and ((alphaPairsChanged > 0) or (entireSet)):   #当迭代次数超过最大值，或者遍历整个集合都未对任意alpha进行修改时就退出循环，跟之前的不一样
        alphaPairsChanged = 0
        #外循环选择第一个alpha，并且其选择过程在两种方式之间交替进行
        if entireSet:   #遍历所有数据集
            for i in range(oS.m):        
                alphaPairsChanged += innerL(i,oS)   #选择第二个,alpha值,并在可能时对其进行优化处理，如果有任意一对alpha值发生改变，那么会返回1
                print("fullSet, iter: {0:d} i:{1:d}, pairs changed {2:d}".format(iter,i,alphaPairsChanged))
            iter += 1
        else:#遍历所有的非边界alpha值，也就是不在边界0或C上的值
            nonBoundIs = nonzero((oS.alphas.A > 0) * (oS.alphas.A < C))[0]
            for i in nonBoundIs:
                alphaPairsChanged += innerL(i,oS)
                print("non-bound, iter: {0:d}i:{1:d}, pairs changed {2:d}" .format(iter,i,alphaPairsChanged))
            iter += 1
        if entireSet:
            entireSet = False 
        elif (alphaPairsChanged == 0):
            entireSet = True  
        print("iteration number:{0:d}".format(iter))
    return oS.b,oS.alphas

def calcWs(alphas,dataArr,classLabels):
    X = mat(dataArr); labelMat = mat(classLabels).transpose()
    m,n = shape(X)
    w = zeros((n,1))
    for i in range(m):
        w += multiply(alphas[i]*labelMat[i],X[i,:].T)
    return w

def testRbf(k1=1.3):
    dataArr,labelArr = loadDataSet('testSetRBF.txt')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, ('rbf', k1)) #C=200 important
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]
    sVs=datMat[svInd] #get matrix of only support vectors
    labelSV = labelMat[svInd];
    print("there are %d Support Vectors".format(shape(sVs)[0]))
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1
    print("the training error rate is: %f".format(float(errorCount)/m))
    dataArr,labelArr = loadDataSet('testSetRBF2.txt')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],('rbf', k1))
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]): errorCount += 1    
    print("the test error rate is: %f".format(float(errorCount)/m))




def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def loadImages(dirName):
    from os import listdir
    hwLabels = []
    trainingFileList = listdir(dirName)           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        if classNumStr == 9:
            hwLabels.append(-1)
        else:
            hwLabels.append(1)
        trainingMat[i,:] = img2vector('{0:s}/{1:s}'.format(dirName, fileNameStr))
    return trainingMat, hwLabels    

def testDigits(kTup=('rbf', 10)):
    dataArr,labelArr = loadImages('trainingDigits')
    b,alphas = smoP(dataArr, labelArr, 200, 0.0001, 10000, kTup)
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    svInd=nonzero(alphas.A>0)[0]   
    sVs=datMat[svInd]              #把不是支持向量的数据都剔除了
    labelSV = labelMat[svInd];
    print("there are {0:d} Support Vectors".format(shape(sVs)[0]))
    m,n = shape(datMat)
    errorCount = 0
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1
    print("the training error rate is: {0:f}".format(float(errorCount)/m))
    dataArr,labelArr = loadImages('testDigits')
    errorCount = 0
    datMat=mat(dataArr); labelMat = mat(labelArr).transpose()
    m,n = shape(datMat)
    for i in range(m):
        kernelEval = kernelTrans(sVs,datMat[i,:],kTup)
        predict=kernelEval.T * multiply(labelSV,alphas[svInd]) + b
        if sign(predict)!=sign(labelArr[i]):
            errorCount += 1    
    print("the test error rate is: {0:f}".format(float(errorCount)/m))
