## 决策树

- 决策树构造
- 信息增益
- 划分数据集
- 递归构建决策树
- 使用决策树
- 案例

### 信息增益

划分数据集后，信息发生的变化即为信息增益，当选择一个特征后，计算信息增益，如果获得最高的信息增益，那这个特征就是最好的选择。



熵：信息的期望。

如果待分类的事务可能有多种分类，则 $x_i$ 的信息定义为： 其中p(xi) 是该分类的概率
$$
l(x_i) = -log_{2}p(x_i)
$$
整个数据集的信息期望为：
$$
H = - \sum_{i=1}^Np(x_i)log_2p(x_i)
$$
当混合的数据越多，熵越大。



### 构建决策树

递归构建，结构是 {featrue : {value : label } }每次选择最好的特征，特征有哪些值，对应的标签（或者特征）

当划分到一定程度，某个标签取值下，数据全是一样的，则退出。

或者，这个数据集下，没有特征可以选择了，则返回多数的标签

```python
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
  
# {'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}}
# {特征：{取值：标签，取值：{}}}
```

同理，使用决策树时，递归遍历决策树

```python
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
```



以上就是《机器学习实战》中的决策树的基本思路，思路还算清晰，里面有部分不适合py3的，例如dict.keys()[0]，以及用pickle存储时，open未使用 'rb', 'wb'的方式，然后打算重新写一下这部分的代码。

[https://github.com/Jian-Yin-Shine/ML_learn/blob/master/%E5%86%B3%E7%AD%96%E6%A0%91/de_classify.py](https://github.com/Jian-Yin-Shine/ML_learn/blob/master/决策树/de_classify.py)

例如：在测试部分，其实只需要example的特征向量，和决策树。