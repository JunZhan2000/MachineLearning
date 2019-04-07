'''
logister回归模型

参数：
alpha: 0.01
迭代次数： 10000

初始的精度：
 0.3489583333333333

优化后的精度：
0.6238406165790417

'''


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def get_data():
    '''获取数据集'''
    path = r'/home/gokej/learngit/AI-group-tasks/task_3/diabetes.csv'
    data = pd.read_csv(path)  # 读取文件
    data = np.array(data)  # 转成数组

    X = data[:, :-1]  # 特征
    X = (X - np.min(X, axis=0)) / (np.max(X, axis=0) - np.min(X, axis=0))  # 特征归一化
    y = data[:, -1].reshape((X.shape[0],))  # 标签

    return X, y


def sigmoid(z):
    '''sigmoid函数'''
    return 1 / (1 + np.exp(-z))    #返回函数值


# 无用
# def compute_cost(theta, X, y):
#     '''计算代价的函数'''
#
#     h_x = sigmoid(np.dot(X, theta))    # 计算由假设函数得到的预测值，exp函数为以e为底的指数函数
#     Cost_x = y * np.log(h_x) + (1 - y) * np.log(1 - h_x)    # 实现Cost函数
#
#     return -np.mean(Cost_x)    # 返回代价值
#


def gradient_descent(train_X, train_y, alpha, epoch):
    '''梯度下降算法'''

    theta = np.ones(train_X.shape[1])  # 初始化数据
    m = train_X.shape[0]
    for i in range(epoch):
        theta = theta - (alpha / m) * (sigmoid(np.dot(train_X, theta)) - train_y) @ train_X

    return theta



def compute_accuracy(theta, test_X, test_y):
    '''计算模型的预测精度'''
    probility = np.dot(test_X, theta)    # 计算得到的可能性
    predictions = [1 if example >= 0.5 else 0 for example in probility]    # 预测结果

    if_yes = [1 if pre_y == tru_y else 0 for pre_y in predictions for tru_y in test_y]
    accuracy = sum(if_yes) / len(if_yes)    # 精度

    return accuracy



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

        tem_index = np.setdiff1d(tem_index, test_index)    # 更新后的供选择的索引池
        train_index = np.setdiff1d(index, test_index)    # 训练集的索引
        test_data = data[test_index]    # 测试集
        train_data = data[train_index]  # 训练集

        # 获取X与y
        train_X = train_data[:, :-1]
        train_y = train_data[:, -1]
        test_X = test_data[:, :-1]
        test_y = test_data[:, -1]

        tem_final_theta = gradient_descent(train_X, train_y, alpha=0.01, epoch=100000)    #　一次参数
        '''训练模型并计算精度'''
        all_accuracy.append(compute_accuracy(tem_final_theta, test_X, test_y))    #　一次精度

    mean_accuracy = sum(all_accuracy) / len(all_accuracy)  # 求平均精度

    return mean_accuracy





X, y = get_data()  # 获取数据
inited_theta = np.ones(X.shape[1])    # 初始化数据
print("初始的精度：\n", compute_accuracy(inited_theta, X, y), '\n')



'''进行优化'''

# alpha = 0.01
# epoch = 10000
# final_theta = gradient_descent(X, y, inited_theta, alpha, epoch)
print("优化后的精度：\n", kk_split(X, y))
