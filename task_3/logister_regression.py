'''
logister回归模型

参数：
alpha: 0.01
迭代次数： 10000

初始的代价：
 1.6237314892936103
初始的精度：
 0.3489583333333333

优化后的代价：
 0.6202681830688881
优化后的精度：
 0.6419949001736112

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


def compute_cost(theta, X, y):
    '''计算代价的函数'''

    h_x = sigmoid(np.dot(X, theta))    # 计算由假设函数得到的预测值，exp函数为以e为底的指数函数
    Cost_x = y * np.log(h_x) + (1 - y) * np.log(1 - h_x)    # 实现Cost函数

    return -np.mean(Cost_x)    # 返回代价值



def gradient_descent(X, y, theta, alpha, epoch):
    '''梯度下降算法'''

    m = X.shape[0]
    for i in range(epoch):
        theta = theta - (alpha / m) * (sigmoid(np.dot(X, theta)) - y) @ X

    return theta



def compute_accuracy(theta, X, y):
    '''计算模型的预测精度'''
    probility = np.dot(X, theta)    # 计算得到的可能性
    predictions = [1 if example >= 0.5 else 0 for example in probility]    # 预测结果

    if_yes = [1 if pre_y == tru_y else 0 for pre_y in predictions for tru_y in y]
    accuracy = sum(if_yes) / len(if_yes)    # 精度

    return accuracy



X, y = get_data()  # 获取数据
inited_theta = np.ones(X.shape[1])    # 初始化数据
print("初始的代价：\n", compute_cost(inited_theta, X, y))
print("初始的精度：\n", compute_accuracy(inited_theta, X, y), '\n')



'''进行优化'''
alpha = 0.01
epoch = 10000
final_theta = gradient_descent(X, y, inited_theta, alpha, epoch)
print("优化后的代价：\n", compute_cost(final_theta, X, y))
print("优化后的精度：\n", compute_accuracy(final_theta, X, y))
