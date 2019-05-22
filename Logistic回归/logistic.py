def loadDataSet():
    data , label = [], []
    file = open('testSet.txt')
    for line in file:
        line = line.split()
        # print(line)
        data.append([1.0, float(line[0]), float(line[1])])
        label.append(int(line[2]))
    return data, label

import math
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# z = w0*x0 + w1*x1 + b
def grad(data, label):
    # np.mat() 将输入转成矩阵，如果输入是array，则不会复制，会共享内存
    # np.mat().transpose()矩阵的转置
    dataMatrix = np.mat(data)         # 转成矩阵 n*3
    labelMatrix = np.mat(label).transpose() # n*1
    m, n = dataMatrix.shape
    alpha = 0.001
    epoch = 500
    weight = np.ones((n,1))     # 3*1

    # 此处weight的更新
    for k in range(epoch):
        h = sigmoid(dataMatrix*weight)
        error = labelMatrix - h
        weight = weight + alpha * dataMatrix.transpose()*error
    return weight



import matplotlib.pyplot as plt

if __name__ == '__main__':
    data, lable  = loadDataSet()
    weight = grad(data, lable)
    # print(weight)
    # print(data, lable)
    data0, data1 = [], []
    for i, e in enumerate(lable):
        if e == 1:
            data1.append([data[i][1], data[i][2]])
        else:
            data0.append([data[i][1], data[i][2]])

    data0 = np.array(data0)
    data1 = np.array(data1)
    plt.plot(data0[:,0], data0[:,1],'*r')
    plt.plot(data1[:,0], data1[:,1], '*g')
    x = np.arange(-3, 3, 0.1)
    # 0 = w0*1 + w1*x1 + w2*x2
    # x2 = (-w0 - w1*x1) / w2
    y = (-weight[0] -weight[1]*x) / weight[2]
    plt.plot(x, y.transpose())
    plt.show()