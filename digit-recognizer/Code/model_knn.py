# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import operator
import time

plt.rcParams['font.sans-serif']=['simhei'] #用于正常显示中文
plt.rcParams['axes.unicode_minus']=False #用于正常显示负号

testDataSet = pd.read_csv('../input/test.csv')
trainDataSet = pd.read_csv('../input/train.csv')
trainLabel = trainDataSet['label']
trainData = trainDataSet.iloc[:, 1:785]
m_t,n_t = np.shape(testDataSet)
m_tr,n_tr = np.shape(trainData)

testDataMat = np.multiply(testDataSet != np.zeros((m_t, n_t)), np.ones((m_t, 1)))
trainDataMat = np.multiply(trainData != np.zeros((m_tr, n_tr)), np.ones((m_tr, 1)))
np.shape(testDataMat)[0]


def classify(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

Label = []
ImageId = []
np.shape(testDataMat)[0]
i=1
time1 = time.time()
for i in range(np.shape(testDataMat)[0]):
    classdifys = classify(testDataMat.iloc[i],trainDataMat,trainLabel,k=10)
    ImageId.append(i+1)
    Label.append(classdifys)
    #print "This is " + str(i+1) +"testdata,classdify is:" + str(classdifys)
time2 = time.time()
train_time = time2-time1
data = pd.DataFrame(Label,ImageId)
data.to_csv('../sub/knn.csv',encoding='utf-8')