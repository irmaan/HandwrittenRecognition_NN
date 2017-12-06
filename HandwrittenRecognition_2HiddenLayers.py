import mnist
import random
from scipy.special import expit
from pathlib import Path
import csv

trainImages=[]
trainLabels =[]
testImages =[]
testLabels =[]

v=[] # first layer weights
w=[] # second layer weights
u=[] # third layer weights

firstHiddenLayerBias=[]
secHiddenLayerBias=[]
outputLayerBias=[]


y_in=[]
z_in=[]
z2_in=[]

DeltaV=[]
DeltaW=[]
DeltaU=[]

DeltaB3=[] # bis third layer
DeltaB2=[] # bias second layer
DeltaB1=[] # bias first layer

alpha=0.01

n=28*28 # number of inputs
p=100  # number of first hidden layer neurons
q=100  # number of second hidden layer neurons
Beta=0.7 * pow(p,1/n)

target = [[0,0,0,0,0,0,0,0,0,1], # 0
          [1,0,0,0,0,0,0,0,0,0], #1
          [0,1,0,0,0,0,0,0,0,0], #2
          [0,0,1,0,0,0,0,0,0,0], #3
          [0,0,0,1,0,0,0,0,0,0], #4
          [0,0,0,0,1,0,0,0,0,0], #5
          [0,0,0,0,0,1,0,0,0,0], #6
          [0,0,0,0,0,0,1,0,0,0], #7
          [0,0,0,0,0,0,0,1,0,0], #8
          [0,0,0,0,0,0,0,0,1,0]] #9

'''
target = [[-1,-1,-1,-1,-1,-1,-1,-1,-1,1], # 0
          [1,-1,-1,-1,-1,-1,-1,-1,-1,-1], #1
          [-1,1,-1,-1,-1,-1,-1,-1,-1,-1], #2
          [-1,-1,1,-1,-1,-1,-1,-1,-1,-1], #3
          [-1,-1,-1,1,-1,-1,-1,-1,-1,-1], #4
          [-1,-1,-1,-1,1,-1,-1,-1,-1,-1], #5
          [-1,-1,-1,-1,-1,1,-1,-1,-1,-1], #6
          [-1,-1,-1,-1,-1,-1,1,-1,-1,-1], #7
          [-1,-1,-1,-1,-1,-1,-1,1,-1,-1], #8
          [-1,-1,-1,-1,-1,-1,-1,-1,1,-1]] #9
'''
'''
target = [[-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,0.8], # 0
          [0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8], #0.8
          [-0.8,0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8], #2
          [-0.8,-0.8,0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8], #3
          [-0.8,-0.8,-0.8,0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8], #4
          [-0.8,-0.8,-0.8,-0.8,0.8,-0.8,-0.8,-0.8,-0.8,-0.8], #5
          [-0.8,-0.8,-0.8,-0.8,-0.8,0.8,-0.8,-0.8,-0.8,-0.8], #6
          [-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,0.8,-0.8,-0.8,-0.8], #7
          [-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,0.8,-0.8,-0.8], #8
          [-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,-0.8,0.8,-0.8]] #9
'''


lenOfTrainClass=10
lenOfTestClass=2
lenOfValidation=1


def loadDataFromMNIST():
    global trainLabels,trainImages,testImages,testLabels
    mndata = mnist

    trainImages = mndata.train_images()
    trainLabels = mndata.train_labels()

    testImages = mndata.test_images()
    testLabels = mndata.test_labels()


def pickSamples(labels,lenOfClass,lenOfValidation):
    classIndexes=[]
    validDataIndexes = []
    for i in range(10):
        count=0
        classRow=[]
        for j in range(len(labels)) :
            if count<lenOfClass:
                if labels[j]== i :
                    classRow.append(j)
                    count+=1
            else:
                break
        classIndexes.append(classRow)

        if lenOfValidation>0  :# start picking validation data
            count=0
            validDataRow=[]
            k=j # index of last picked training sample kept in j of above loop used as start index for validation data picking
            for k in range(j,len(labels)):
               if count<lenOfValidation:
                    if labels[k] == i:
                        validDataRow.append(k)
                        count += 1
               else:
                    break
            validDataIndexes.append(validDataRow)

    if lenOfValidation>0:
        return classIndexes,validDataIndexes
    else:
        return classIndexes


def normalizeData(classIndexes,data):
    normalizedData=[]
    for i in range(len(classIndexes)):
        for d in classIndexes[i]:
            vectorSample=[]
            for k in range(28):
                for l in range(28):
                    #normalizedValue=2*(data[d][k][l]/255)-1
                    normalizedValue= data[d][k][l]/255
                    vectorSample.append(normalizedValue)
            normalizedData.append(vectorSample)

    return normalizedData


def readWeights(weightFileContent):
    weights=[]
    for weight in weightFileContent:
        rowW=[]
        for i in range(len(weight)):
            if i==0:
                weight[i]=str(weight[i]).replace("[","")
            if i == len(weight) - 1:
                weight[i]=str(weight[i]).replace("]","")
            rowW.append(float(weight[i]))
        weights.append(rowW)

    return weights


def readWeightFiles(weight):
    weightFile = Path(weight +".txt")
    weightList = []
    if weightFile.is_file():
        weightFile = open(weight +".txt", 'r', newline="\n")
        reader = csv.reader(weightFile, delimiter=',')
        weightList = list(reader)
    return weightList


def initWeights():
    global  v,w,u
    firstLayerWeights=[]
    secondLayerWeights=[]
    thirdLayerWeights=[]

    vFile = Path("v.txt")
    wFile = Path("w.txt")
    uFile = Path("u.txt")

    if vFile.is_file() and wFile.is_file() and uFile.is_file():
        vList=readWeightFiles("v")
        wList = readWeightFiles("w")
        uList = readWeightFiles("u")

        w = readWeights(wList)
        v = readWeights(vList)
        u = readWeights(uList)

    else:
        for i in range(n):
            xRowV=[]
            for j in range(p):
                xRowV.append(random.uniform(-0.2,0.2))
            firstLayerWeights.append(xRowV)

        for k in range(p):
            xRowW = []
            for l in range(q):
                xRowW.append(random.uniform(-0.2, 0.2))
            secondLayerWeights.append(xRowW)

        for j in range(q):
            xRowU = []
            for l in range(len(target)):
                xRowU.append(random.uniform(-0.2, 0.2))
            thirdLayerWeights.append(xRowU)

        v=firstLayerWeights
        u=secondLayerWeights
        w=thirdLayerWeights


def initBias():
    global firstHiddenLayerBias,secHiddenLayerBias,outputLayerBias
    bH1File = Path("bH1.txt")
    bH2File = Path("bH2.txt")
    bOFile = Path("bO.txt")

    if bH1File.is_file() and bOFile.is_file():
        bH1List = readWeightFiles("bH1")
        bH2List = readWeightFiles("bH2")
        bOList = readWeightFiles("bO")

        firstHiddenLayerBias = readWeights(bH1List)
        secHiddenLayerBias = readWeights(bH2List)
        outputLayerBias = readWeights(bOList)


    else:
        for j in range(len(target)):
            outputLayerBias.append(random.uniform(-0.2, 0.2))
        for i in range(p):
            firstHiddenLayerBias.append(random.uniform(-0.2, 0.2))
        for k in range(q):
            secHiddenLayerBias.append(random.uniform(-0.2, 0.2))


def activationFunction(input): # Bipolar Sigmoid
    #return (2/(1+math.exp(-1*input)))-1
    #return (2*(expit(input)))-1
    return expit(input)


def derivationOfActivationFunction(input):
    #return 0.5 *(1+activationFunction(input))*(1-activationFunction(input))
    return activationFunction(input) *(1-activationFunction(input))



def feedForward(X):
    global Z,Y,y_in,z_in,z2_in
    tempZ=[]
    tempZ2=[]
    tempY=[]
    y_in=[]
    z_in=[]
    z2_in=[]
    for j in range(p):
        sumZ=0
        for i in range(n):
            sumZ+=X[i]* v[i][j]
        z_in.append(firstHiddenLayerBias[j] + sumZ)
        tempZ.append(activationFunction(z_in[j]))
    Z=tempZ

    for h in range(q):
        sumZ2 = 0
        for i in range(p):
            sumZ2 += Z[i] * u[i][h]
        z2_in.append(secHiddenLayerBias[h] + sumZ2)
        tempZ2.append(activationFunction(z2_in[h]))
    Z2 = tempZ2

    for k in range(len(target)):
        sumY=0
        for j in range(q):
            sumY+=Z2[j]* w[j][k]
        y_in.append(outputLayerBias[k]+sumY)
        tempY.append(activationFunction(y_in[k]))
    Y=tempY

    return Y,Z,Z2


def backPropagation(X,Y,Z,Z2,target):
    #step 6
    deltaOut=[]
    for k in range(len(target)):
        deltaOut.append((target[k]-Y[k])* derivationOfActivationFunction(y_in[k]))

    for j in range(q):
        deltaRow=[]
        for k in range(len(target)):
            deltaRow.append(alpha* deltaOut[k]*Z2[j])
        DeltaW.append(deltaRow)

    for k in range(len(target)):
        DeltaB2.append(alpha * deltaOut[k])

    #step 7
    deltaHidden2=[]
    delta_in2 = []
    for j in range(q):
        sumDelta=0
        for k in range(len(target)):
            sumDelta+=deltaOut[k]*w[j][k]
        delta_in2.append(sumDelta)
        deltaHidden2.append(delta_in2[j] * derivationOfActivationFunction(z2_in[j]))

    for i in range(p):
        deltaRow = []
        for j in range(q):
            deltaRow.append(alpha * deltaHidden2[j] * Z[i])
        DeltaU.append(deltaRow)

    for j in range(q):
        DeltaB2.append(alpha * deltaHidden2[j])

    deltaHidden1 = []
    delta_in1 = []
    for j in range(p):
        sumDelta = 0
        for k in range(q):
            sumDelta += deltaHidden2[k] * u[j][k]
        delta_in1.append(sumDelta)
        deltaHidden1.append(delta_in1[j] * derivationOfActivationFunction(z_in[j]))

    for i in range(n):
        deltaRow = []
        for j in range(p):
            deltaRow.append(alpha * deltaHidden1[j] * X[i])
        DeltaV.append(deltaRow)

    for j in range(p):
        DeltaB1.append(alpha * deltaHidden1[j])



def updateWeights():
    #step 8
    for j in range(q):
        for k in range(len(target)):
            w[j][k]=w[j][k]+DeltaW[j][k]

    for k in range(len(target)) :
            outputLayerBias[k]=outputLayerBias[k]+DeltaB2[k]

    for i in range(p):
        for j in range(q):
            u[i][j]=u[i][j]+DeltaU[i][j]

    for j in range(p):
        secHiddenLayerBias[j] = secHiddenLayerBias[j] + DeltaB2[j]

    for i in range(n):
        for j in range(p):
            v[i][j]=v[i][j]+DeltaV[i][j]

    for j in range(q):
        firstHiddenLayerBias[j] = firstHiddenLayerBias[j] + DeltaB1[j]

validTSE=[]
validMSE=[]  # mean squared errors
trainTSE=[]  # mean squared errors
trainMSE=[]  # mean squared errors


def calculateError(Y,target,TSE):
    sumSquaredError = 0
    for i in range(len(Y)):
        sumSquaredError += pow(target[i] - Y[i], 2)
    TSE.append(sumSquaredError)


def calculateMSE(MSE,TSE):
    if len(TSE)>0:
        sumTSEs = 0
        for value in TSE:
            sumTSEs += value
        mse = sumTSEs/len(TSE)
        MSE.append(mse)


def stoppingCondition():
    testNetwork(normalizedValidData,lenOfValidation,True)
    print("Valid MSE :%s" % str(validMSE[-1]))
    print("Train MSE :%s" % str(trainMSE[-1]))
    if len(validMSE)>10:
        last3MSEs=validMSE[len(validMSE)-2:]
        if sorted(last3MSEs)==last3MSEs:
                return True
        else :
                return False
    else:
        return  False


def all_same(items):
    return all(x == items[0] for x in items)

def saveWeights(epoch):
    wFile = open('w.txt', 'w')
    wFile.truncate()
    for wRow in w:
        wFile.write("%s\n" % wRow)

    vFile = open('v.txt', 'w')
    vFile.truncate()
    for vRow in v:
        vFile.write("%s\n" % vRow)

    bHFile = open('bH.txt', 'w')
    bHFile.truncate()
    bHFile.write("%s" % firstHiddenLayerBias)

    bOFile = open('bO.txt', 'w')
    bOFile.truncate()
    bOFile.write("%s" % outputLayerBias)

    epochFile = open('e.txt', 'w')
    epochFile.write("%s\n" % str(epoch))

    trainMSEFile = open('trainMSE.txt', 'w')
    for row in trainMSE:
        trainMSEFile.write("%s\n" % row )

    validMSEFile = open('validMSE.txt', 'w')
    for row in validMSE:
        validMSEFile.write("%s\n" % row)



def clearSampleVariables():
    y_in.clear()
    z_in.clear()
    DeltaW.clear()
    DeltaV.clear()
    DeltaB1.clear()
    DeltaB2.clear()

def modifyTargets(Y):
    max=0
    iMax=0
    for i in range(len(Y)):
        if Y[i]>=max:
            iMax=i
            max=Y[i]
    Y[iMax]=1
    for i in range(len(Y)):
        if i!=iMax:
            Y[i]=0


def testNetwork(testData,lenOfTest,validation):
    correct=0
    false=0
    validTSE.clear()
    for i in range(len(testData)):
        X=testData[i]
        Y,Z,Z2=feedForward(X)
        if validation==False:
            modifyTargets(Y)
        if Y==target[int(i/lenOfTest)]:
            correct+=1
        else:
            calculateError(Y, target[int(i / lenOfTest)], validTSE)
    calculateMSE(validMSE, validTSE)
    return (correct/len(testData)) *100


def trainNetwork():
    global alpha
    epoch=0
    while True:
        #if epoch%10==0 and alpha >=0.01:
           # alpha=alpha/2
        trainTSE.clear()
        for i in range(len(normalizedTrainData)):
            clearSampleVariables()
            X = normalizedTrainData[i]
            Y, Z,Z2 = feedForward(X)
            backPropagation(X,Y, Z,Z2, target[int(i / lenOfTrainClass)])
            updateWeights()
            calculateError(Y, target[int(i / lenOfTrainClass)],trainTSE)
        calculateMSE(trainMSE,trainTSE)
        epoch += 1
        saveWeights(epoch)
        if stoppingCondition():
            break

    return epoch


loadDataFromMNIST()
trainClassIndexes,validDataIndexes = pickSamples(trainLabels,lenOfTrainClass,lenOfValidation)
testClassIndexes = pickSamples(testLabels,lenOfTestClass,0)

normalizedTrainData=normalizeData(trainClassIndexes,trainImages)
normalizedValidData=normalizeData(validDataIndexes,trainImages)
normalizedTestData=normalizeData(testClassIndexes,testImages)

initWeights()
initBias()

errorRates=[]
finalEpoch=trainNetwork()
print(finalEpoch)
finalPrec=testNetwork(normalizedTestData,lenOfTestClass,False)
print("FINAL TEST PRECISION :")
print(finalPrec)



