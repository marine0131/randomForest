#!/usr/bin/env python

from RandomForest import RandomForest
from Data import TrainingData, sample
import copy
import sys
import numpy as np
import json
import cPickle

# An array of all the files containing data and an array of the labels for each file
labels = ["negative", "positive"]
sub1 = ["../feature_data/neg_20180306.log", "../feature_data/pos_20180306.log"]


class confusionMatrix:
    '''
    This is just a helper class for creating a confusion matrix
    It contains a map of each label where the first key is the true
    label and each value is another map where the second key is the
    predicted label. From there we can increment a value for that
    prediction.
    '''

    def __init__(self, labelList):

        self.matrix = {}
        for label in labelList:
            self.matrix[label] = dict((key, 0) for key in labelList)

    def update(self, actual, predicted):
        self.matrix[actual][predicted] += 1

    def printMatrix(self):
        for label in self.matrix.keys():
            for nextLabel in self.matrix[label].keys():
                print self.matrix[label][nextLabel],
            print('\n')


def generateTrainData():
    '''
    This creates a data set to be used to train a classifier.
    The data is from participant 1 and uses the calculated
    joint angles as features.
    '''
    trainingData = TrainingData("train data")
    index = 0

    for filename in sub1:
        # Open the file
        fIn = open(filename, 'r')
        # For each line of the file calculate the
        # angles inbetween joints and use the resulting
        # array as the feature vector. Add that to the trainingData.
        for line in fIn:
            features = np.array(json.loads(line)['feature'])
            trainingData.addSampleFromFeatures(features, labels[index])
        fIn.close()
        index += 1

    return trainingData


def generateTestData():
    '''
    This creates a list of data samples to be tested.
    Theses samples are from participants 2 and 3 and
    the calculated angles are used as features.
    '''
    testSamples = []
    index = 0

    for filename in sub1:
        # Open the file
        fIn = open(filename, 'r')
        # For each line of the file calculate the
        # angles inbetween joints and use the resulting
        # array as the feature vector. Add that to the list.
        for line in fIn:
            features = np.array(json.loads(line)['feature'])
            testSamples.append(sample(features, labels[index]))
        fIn.close()
        index += 1

    return testSamples


def generateDetectData(filename):
    '''
    This creates a list of data samples to be tested.
    Theses samples are from participants 2 and 3 and
    the calculated angles are used as features.
    '''
    detectSamples = []

    # Open the file
    fIn = open(filename, 'r')
    for line in fIn:
        features = np.array(json.loads(line)['feature'])
        detectSamples.append(sample(features))
    fIn.close()

    return detectSamples


def train(rf):
    '''
    Trains a random forest on the data from all data
    '''
    theData = generateTrainData()
    testForest = RandomForest(theData)
    print("Training")
    testForest.train()
    print("Done!")

    with open(rf, 'wb') as f:
        cPickle.dump(testForest, f)
        print('randomForest model saved to: ' + rf)


def test(rf):
    '''
    test a random forest
    '''
    with open(rf, 'rb') as f:
        testForest = cPickle.load(f)

    testList = generateTestData()

    results = confusionMatrix(labels)

    print(results.matrix)
    for samp in testList:
        resultLabel = testForest.classify(samp)
        trueLabel = samp.getLabel()

        results.update(trueLabel, resultLabel)

    results.printMatrix()


def detect(rf, log):
    '''
    detect a set of data
    '''
    with open(rf, 'rb') as f:
        testForest = cPickle.load(f)

    detectList = generateDetectData(log)

    for samp in detectList:
        resultLabel = testForest.classify(samp)
        print(resultLabel)


'''
1st train/test/detect
2nd model_name in model folder
3rd if --detect log_filename in data folder
'''
if __name__ == '__main__':
    arg = sys.argv[1]

    if arg == '--train':
        rf = '../model/' + sys.argv[2]
        train(rf)
    elif arg == '--test':
        rf = '../model/' + sys.argv[2]
        test(rf)
    elif arg == '--detect':
        rf = '../model/' + sys.argv[2]
        log = '../data/' + sys.argv[3]
        detect(rf, log)
    else:
        print('please input correct args')
