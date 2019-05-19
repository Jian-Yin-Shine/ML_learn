# 构造数据集

def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    label = ['no surfacing', 'flippers']
    return dataSet, label

# 计算香农熵
from math import log
def calcShannonEnt(dataset):
    numEntries = len(dataset)  # 总数据量
    mp = {}
    for data in dataset:        # 计算 p(xi),即每种类别的概率
        data_lable = data[-1]
        if data_lable not in mp.keys():
            mp[data_lable] = 0
        mp[data_lable] +=1

    ans = 0
    for key in mp:              # H =  - sum p(xi) * log2(p(xi))
        ans -= mp[key] / numEntries * log(mp[key] / numEntries, 2)
    return ans


# 划分数据集
# axis 第几个特征
# value 第几个特征的值
def splitDataSet(dataSet, axis, value):
    ret = []
    for data in dataSet:
        if data[axis] == value:
            ret.append(data[:axis]+data[axis+1:]) # list [] + [] = extend
    return ret

# 选择最好的划分方式
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1       # 特征个数
    baseEntropy = calcShannonEnt(dataSet)   # 信息熵
    infoGain = 0
    bestinfoGain = -1

    for i in range(numFeatures):            # 遍历每个特征
        val = [ e[i] for e in dataSet]      # 挑出每个样本的特征的取值
        uniqueval = set(val)
        newEntropy = 0
        for value in uniqueval:
            subDataSet = splitDataSet(dataSet, i , value)
            newEntropy += len(subDataSet) / len(subDataSet) * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy     # 划分后，不在混乱，熵减小 tips
        if infoGain > bestinfoGain:
            bestinfoGain = infoGain
            bestFeature = i
    return bestFeature


import operator
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 0
        classCount[vote]+=1
    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    return classList[sorted(classCount, key=lambda x:classCount[x], reverse=True)[0]]

# 递归构建决策树
def creatTree(dataSet, labels):
    classList = [ e[-1] for e in dataSet]
    if len(set(classList)) == 1:      # 集合只有一个类别
        return classList[0]
    if len(dataSet[0]) == 1:        # 没有特征可以分了，采取投票的方法
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = label[bestFeat]
    myTree = {bestFeatLabel: {}}
    del(labels[bestFeat])

    featValues = [ e[bestFeat] for e in dataSet]
    uniqueVals = set(featValues)

    for val in uniqueVals:
        subLabels = labels[:]
        myTree[bestFeatLabel][val] = creatTree(splitDataSet(dataSet, bestFeat, val), labels=subLabels)
    return myTree


def classify(inputTree, featLabels, testVec):
    # firstStr = inputTree.keys()
    for i in inputTree:
        firstStr = i
        break
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] == key:
            if type(secondDict[key]).__name__ == 'dict':
                classLabel = classify(secondDict[key], featLabels, testVec)
            else:
                classLabel = secondDict[key]
    return classLabel


if __name__ == '__main__':
    data, label = createDataSet()
    print(label)
    for i in data:
        print (i)
    # print(calcShannonEnt(dataset=data))
    # print(splitDataSet(dataSet=data, axis=0, value=0))
    # print(chooseBestFeatureToSplit(data))
    tree = creatTree(data, label)
    _, label = createDataSet()
    print(classify(tree, label, [1, 0]))
    # tree = creatTree(data, label)