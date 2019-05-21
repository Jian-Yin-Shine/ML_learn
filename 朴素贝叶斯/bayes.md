## 朴素贝叶斯



- 条件概率
- 使用朴素贝叶斯进行文档分类



条件概率公式:
$$
P(B|A) = \frac{P{(AB)}}{P(A)} = \frac{P(A|B)*P(B)}{P(A)}
$$
朴素贝叶斯基于两点假设： 

1. 每个特征相互独立（例如：在文本中，特征bocan(培根) 出现在 特征unhealthy 与 特征delicious 后面的概率相同，当然，在现实中，这是不太可能的）
2. 每个特征同等重要（例如：在文本中，在判断留言板中的留言是否得当，不需要看完所有的单词，而是只需要看10～20个单词即可）

尽管上述假设不太成立，或者有些小瑕疵，但是朴素贝叶斯的效果还是可以的。



在案例中：

p(B) 就是p(ci) 每个类别的概率

p(A|B) 就是 vec0 ，这个类别的情况下，每个单词（特征）出现的概率

那么我们所求的p(W|ci) 根据独立假设，就是各个特征出现的概率相乘。我们这里取对数防止下溢。

p(w1|ci)p(w2|ci)…p(wn|c) 有一个为0，就全部为零，初始化为1

```python
def trainBN(trainMatrix, trainCategory):
    numtrain = len(trainMat)
    numwords = len(trainMat[0])
    pAbusive = sum(trainCategory) / numtrain # p(侮辱性)概率

    p0Num = np.ones(numwords)								# p0Num 每个侮辱性词汇出现的次数
    p1Num = np.ones(numwords)
    p0Denom, p1Denom = 2, 2									# p0Denom 所有侮辱性词汇的个数

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
```

