def loadDataSet():
    # 切分的词条
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    # 类别标签向量，1代表侮辱性词汇，0代表不是
    classVec = [0, 1, 0, 1, 0, 1]
    # 返回实验样本切分的词条、类别标签向量
    return postingList, classVec

# 把所有文本按照词汇set()
def creatVocabList(dataSet):
    vocabSet = set()
    for exam in dataSet:
        vocabSet = vocabSet | set(exam)
    return vocabSet

# 查看input在词汇表中的出现情况
def judge(vocabSet, input):
    ans = [0] * len(vocabSet)

    for word in input:
        for i, dicts in enumerate(vocabSet):
            if word == dicts:
                ans[i] = 1
                break

    return ans

# 朴素贝叶斯

import numpy as np
def trainBN(trainMatrix, trainCategory):
    numtrain = len(trainMat)
    numwords = len(trainMat[0])
    pAbusive = sum(trainCategory) / numtrain # p(侮辱性)概率

    p0Num = np.ones(numwords)
    p1Num = np.ones(numwords)
    p0Denom, p1Denom = 2, 2

    for i in range(numtrain):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = p1Num / p1Denom        # p(wi | c1)    在是侮辱型语句的条件下， 是这个wi单词的概率
    p0Vect = p0Num / p0Denom


    # 这样，我们就可以通过条件概率公式:
    # p(ci | w) = p(w*ci) / p(w) = p(w|ci) * p(ci) / p(w)   求得是这个文本下的，某个种类的概率
    p1Vect = log(p1Num / p1Denom)
    p0Vect = log(p0Vect / p0Denom)

    return p0Vect, p1Vect, pAbusive

from math import log
def classfiyBN(testvec, p0Vect, p1Vect, pClass1):
    # 基于朴素贝叶斯的假设, 各个特征之间独立
    # p(W | c) = p (w1, w2, w3,...,wn | c) = p(w1|c)*p(w2|c)...p(wn|c)
    # 然后，概率p 都很小，这样乘起来，就到0，我们知道ln(a*b) = ln(a) + ln(b)，可以使用加法
    p1 = sum(testvec * p1Vect) + log(pClass1)
    p0 = sum(testvec * p0Vect) + log(1.0 -pClass1)
    if p1 > p0:
        return 1
    else :
        return 0


if __name__ == '__main__':
    data, label = loadDataSet()
    dataSet = creatVocabList(data)
    print(dataSet)
    print(data[0])
    trainMat = []       # 每个文本在词汇表中的出现情况
    for e in data:
        trainMat.append(judge(dataSet, e))
    print('-'* 50)
    for i in trainMat:
        print(i)
    print('-'*30)
    p0Vect, p1Vect, pAbusive = trainBN(trainMat, label)
    print(p0Vect, p1Vect)
    print(p0Vect.shape, p1Vect.shape)
    print('*'*30)
    testExample = ['love', 'my', 'dalmation']
    thisDoc = np.array(judge(dataSet, testExample))

    predictlabel= classfiyBN(thisDoc, p0Vect, p1Vect, pAbusive)
    print(predictlabel)

    testExample = ['stupid']
    print(classfiyBN(judge(dataSet, testExample), p0Vect, p1Vect, pAbusive))