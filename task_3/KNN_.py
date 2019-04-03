'''
k     精度
1     0.69536
3     0.746053
5     0.736341
10    0.743985
20    0.749749

所以k选取3到10为佳
'''




import numpy as np
import pandas as pd
import random


def get_data():
    '''获取数据集'''
    path = r"diabetes.csv"
    data = pd.read_csv(path)  # 读取文件
    data = np.array(data)  # 转成数组

    X = data[:, :-1]  # 特征
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))  # 特征归一化
    y = data[:, -1].reshape((X.shape[0],))  # 标签

    return X, y


def compute_dis(sample, train_X, k=20):
    '''计算距离，并按大小顺序排序
    sample为用于预测的样本， X为训练集的特征
    返回距离最小的二十个样本的索引'''
    distance = np.ones((train_X.shape[0],))
    for i in range(train_X.shape[0]):
        # 每一次循环记录一个距离
        for j in range(train_X.shape[1]):
            distance[i] += (sample[j] - train_X[i][j]) ** 2
        distance[i] = distance[i] ** 0.5
    index = np.argsort(distance)  # 获取距离最小的二十个样本的索引

    return index[:k]


def classify(sample, train_X, train_y):
    '''进行分类'''
    index = compute_dis(sample, train_X)  # 获取距离最小的二十个样本的索引

    dic = {}
    for example in train_y[index]:
        '''计算各标签的出现次数'''
        if example not in dic.keys():
            '''如果之前没有出现过该标签'''
            dic[example] = 1
        else:
            '''如果已经出现过，直接加1'''
            dic[example] += 1

    return max(dic, key=dic.get)



def accuracy(train_X, train_y, test_X, test_y):
    '''计算精度'''
    prediction = np.ones(test_X.shape[0])
    for i in range(test_X.shape[0]):
        prediction[i] = classify(test_X[i], train_X, train_y)

    if_correct = [1 if prediction[i] == test_y[i] else 0 for i in range(test_X.shape[0])]

    return sum(if_correct) / len(if_correct)  # 返回精度



def kk_split(X, y, kk=10):
    '''交叉验证法划分数据集'''

    data = np.hstack((X, y.reshape(X.shape[0], 1)))  # 合并特征与标签
    n = int(data.shape[0] / kk)  # 每一份子集的样本数

    index = np.array(range(data.shape[0]))
    tem_index = index

    all_accuracy = []

    for i in range(kk):
        '''每一次随机划分出一个子集'''
        # 随机选出索引
        if i == kk - 1:
            test_index = tem_index
        else:
            test_index = np.random.choice(tem_index, n, replace=False)

        print("1： %d" % len(tem_index))
        tem_index = np.setdiff1d(tem_index, test_index)    # 更新后的供选择的索引池
        train_index = np.setdiff1d(index, test_index)    # 训练集的索引
        test_data = data[test_index]    # 测试集
        train_data = data[train_index]  # 训练集


        print("2： %d" % test_data.shape[0])
        print("3： %d\n\n" % train_data.shape[0])
        # 获取X与y
        train_X = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_X = test_data[:, :-1]
        test_y = test_data[:, -1]

        '''训练模型并计算精度'''
        all_accuracy.append(accuracy(train_X, train_y, test_X, test_y))

    mean_accuracy = sum(all_accuracy) / len(all_accuracy)  # 求平均精度

    return mean_accuracy


X, y = get_data()  # 获取数据

# print(X.shape)  (768, 8)
# print(y.shape)  (768, 1)


mean_accuracy = kk_split(X, y)
print("精度： %f" % mean_accuracy)