import numpy as np
import math
import random

inputData = []
header = []
inputsCount = 0
neuronesCount = 0

learningConstant = 0.2

inputsWeights = []
hiddenLayerWeights = []
hiddenLayerBiasWeights = []
finalBiasWeight = 0

inputsDividers = []

def parseDataFile(pFile):
    global header
    global inputData

    inputData = []

    f = open(pFile, 'r')
    header = f.readline().split(';')

    for line in f.readlines():
        inputData.append(map(float, line.split(';')))
    f.close()

    inputData = np.array(inputData)
    for i in range(len(inputData[0])):
        col = inputData[:,i]
        maxCol = max(1.0,max(col))
        col /= float(maxCol)
        inputsDividers.append(maxCol)

    print inputData

    random.shuffle(inputData)

def addDataToCSV(pFile, data):
    f = open(pFile, 'a')
    f.write('\n' + ';'.join(map(str, data)))

def sigmoid(x):
    #print x
    return 1/(1 + math.exp(-x))

def derivate(x):
    return sigmoid(x) * (1 - sigmoid(x))

def learnOne(data):
    tmpPredictions = []
    tmpErrors = []
    tmpXs = []
    y = data[-1]

    for i in range(len(inputsWeights)):
        weights = inputsWeights[i]
        biasWeight = hiddenLayerBiasWeights[i]
        tmpX = sum(np.array(data[:-1]) * weights) + biasWeight
        prediction = sigmoid(tmpX)
        tmpXs.append(tmpX)
        tmpPredictions.append(prediction)

    tmpFinalX = sum(np.array(tmpXs) * hiddenLayerWeights) + finalBiasWeight
    tmpFinalPrediction = sigmoid(tmpFinalX)
    tmpFinalError = predictionError(tmpFinalPrediction, y)

    for i in range(len(inputsWeights)):
        weights = inputsWeights[i]
        error = tmpFinalError * hiddenLayerWeights[i]
        tmpErrors.append(error)

    return tmpFinalPrediction, tmpFinalError, tmpFinalX, tmpPredictions, tmpErrors, tmpXs

def predictionError(prediction, dataTarget):
    return dataTarget - prediction

def retropropagation(errors, finalError, data):
    global finalBiasWeight


    tmpFinalSum = sum(np.array(errors) * hiddenLayerWeights) + finalBiasWeight
    tmpFinalCorrection = learningConstant * finalError * derivate(tmpFinalSum)
    for i in range(neuronesCount):
        tmpSum = sum(np.array(data[:-1]) * inputsWeights[i]) + hiddenLayerBiasWeights[i]
        tmpCorrection = learningConstant * finalError * hiddenLayerWeights[i] * derivate(tmpSum)
        for j in range(inputsCount):
            inputsWeights[i][j] += tmpCorrection * data[j]

        hiddenLayerBiasWeights[i] += tmpCorrection

        hiddenLayerWeights[i] += tmpCorrection * errors[i]
    finalBiasWeight += tmpFinalCorrection


def init(pFile):
    global inputsWeights
    global inputsCount
    global neuronesCount
    global hiddenLayerWeights
    global hiddenLayerBiasWeights
    global finalBiasWeight

    parseDataFile(pFile)

    inputsCount = len(header) - 1
    neuronesCount = int(inputsCount * 0.66)
    print "inputs count : ", inputsCount
    print "hiden layer neurones count : ", neuronesCount
    inputsWeights = np.ones([neuronesCount, inputsCount]) * 2 * np.random.rand(neuronesCount, inputsCount) - np.ones([neuronesCount, inputsCount])
    hiddenLayerWeights = np.ones(neuronesCount) * 2 * np.random.rand(neuronesCount) - np.ones(neuronesCount)
    hiddenLayerBiasWeights = np.ones(neuronesCount) * 2 * np.random.rand(neuronesCount) - np.ones(neuronesCount)
    finalBiasWeight = random.random()


def main():
    init("test.txt")

    tmpFinalPrediction, tmpFinalError, tmpFinalX, tmpPredictions, tmpErrors, tmpFinalX = learnOne(inputData[0])
    print "prediction : ", tmpFinalPrediction
    print "error : ", tmpFinalError

    print inputsWeights

    retropropagation(tmpErrors, tmpFinalError, inputData[0])

    print inputsWeights

if __name__ == '__main__':
    main()