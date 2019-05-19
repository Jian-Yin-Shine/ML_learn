from math import log

def dataSet():
    data = [[1, 1, 'yes'],
            [1, 1, 'yes'],
            [1, 0, 'no'],
            [0, 1, 'no'],
            [0, 1, 'no']]
    vec= ['no surfacing', 'flippers']
    return data, vec

def calcShannon(data):
    nums = len(data)
    dicts = {}
    for example in data:
        if example[-1] not in dicts:
            dicts[example[-1]] = 0
        dicts[example[-1]] +=1
    ans = 0
    for i in dicts:
        ans -= dicts[i] / nums * log(dicts[i] / nums, 2)
    return ans

def splitDataSet(dataSet, axis, value):
    subDataSet = []
    for e in dataSet:
        if e[axis] == value:
            subDataSet.append(e[:axis] + e[axis+1:])
    return subDataSet

def choosvec(data):
    baseShannoe = calcShannon(data)
    infoGain = 0
    for i in range(len(data[0])-1):         # 遍历每一个特征
        valuelist = [ e[i] for e in data]
        valuelist = set(valuelist)
        shannon = 0
        for j in valuelist:
            subdataSet = splitDataSet(data, i, j)
            shannon += len(subdataSet) / len(data) * calcShannon(subdataSet)
        # print(shannon)
        if infoGain < baseShannoe - shannon:
            res = i
            infoGain = baseShannoe - shannon
    return res



def creatTree(data, vec):
    labels = [ i[-1] for i in data]
    # print(labels)
    if len(set(labels)) == 1:
        return labels[0]
    if len(data[0]) == 1:
        dicts = {}
        for label in labels:
            if label not in dicts.keys():
                dicts[label] = 0
            dicts[label]+=1
        return sorted(dicts, key=lambda x:dicts[x], reverse=True)[0]

    # 挑选最好的特征
    vecindex = choosvec(data)
    vecname = vec[vecindex]

    tree = {vecname: {}}
    value = [ e[vecindex] for e in data]
    uniquevalue = set(value)

    for i in uniquevalue:
        subDataSet = splitDataSet(data, vecindex, i)
        tree[vecname][i] = creatTree(subDataSet, vec[:vecindex]+vec[vecindex+1:])
    return tree

def test(tree, example):
    vecname = list(tree.keys())[0]
    tree = tree[vecname]
    for i in tree:
        if i == example[0]:
            if type(tree[i]).__name__ == 'dict':
                label = test(tree[i], example[1:])
            else:
                label  = tree[i]
    return label

import pickle

if __name__ == '__main__':
    data, vec = dataSet()
    print(calcShannon(data))
    # print(creatTree(data, vec))
    tree = creatTree(data, vec)
    # print(tree)

    fw = open('decision_tree.txt', 'wb')
    pickle.dump(tree,fw)
    fw.close()
    '''
    file = open('decision_tree.txt', 'rb')
    tree = pickle.load(file)'''

    example = [1, 1]
    print(test(tree, example))
    # print()

