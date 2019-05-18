## 决策树

- 决策树构造
- 信息增益
- 划分数据集
- 递归构建决策树
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

