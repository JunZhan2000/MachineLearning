import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


def load_data():
    '''获取数据集'''
    data = load_iris()
    X = data['data']
    y = data['target']
    X = pd.DataFrame(X)
    X.insert(0, 'ones', 1)    # 向量化
    X = np.array(X)

    return X, y



def sigmoid(z):
    '''sigmoid函数'''
    return 1 / (1 + np.exp(-z))    #返回函数值



def compute_cost(theta, X, y):
    '''计算代价的函数'''

    h_x = sigmoid(np.dot(X, theta))    # 计算由假设函数得到的预测值，exp函数为以e为底的指数函数
    Cost_x = y * np.log(h_x) + (1 - y) * np.log(1 - h_x)    # 实现Cost函数

    return -np.mean(Cost_x)    # 返回代价值



def compute_all_cost(all_theta, X, y):
    '''计算所有标签的代价和'''

    all_cost = 0    # 总代价

    for i in range(all_theta.shape[0]):
        '''一次循环计算一组代价'''
        tem_theta = all_theta[i, :]    # 获取第i个标签对应的theta
        tem_y = np.array([0 if label == i else 1 for label in y])    # 获取是i与非i的标签数组
        tem_cost = compute_cost(tem_theta, X, tem_y)    #第i个标签的代价
        all_cost += tem_cost

    return all_cost


def gradient_descent(X, y, theta, alpha, epoch):
    '''梯度下降算法'''

    m = X.shape[0]
    for i in range(epoch):
        # theta = theta - (alpha / m) * (X * theta.T - y).T * X
        # 上面一行是在线性回归中使用的公式，只需替换这一行代码便可以适用于逻辑回归
        theta = theta - (alpha / m) * (sigmoid(np.dot(X, theta)) - y) @ X

    return theta



def get_all_theta(X, y, K):
    '''获取优化后的theta矩阵'''
    '''X是特征，y是标签，K是标签种类数'''

    all_theta = np.zeros((K, X.shape[1]))

    for i in range(0, K):
        '''对每一种标签进行一次循环'''
        theta = np.zeros(X.shape[1])
        tem_y = np.array([0 if label == i else 1for label in y ])    # 获取是i与非i的标签数组
        tem_final_theta = gradient_descent(X, tem_y, theta, alpha = 0.03, epoch = 40000)    # 获取优化后的一组theta
        all_theta[i, :] = tem_final_theta    # 更新第i组theta的值

    return all_theta



def predict(theta, X):
    '''获得预测结果'''

    probability = sigmoid(np.dot(X, theta))
    return [1 if x >= 0.5 else 0 for x in probability]  # return a list



# 得到特征，标签，标签种类数
X, y = load_data()
K = len(np.unique(y))
final_theta = get_all_theta(X, y, K)


print("初始代价："),
print(compute_all_cost(np.zeros((K, X.shape[1])), X, y))
print("\n运行梯度下降后：")
print("theta：")
print(final_theta)
print("最终代价："),
print(compute_all_cost(final_theta, X, y))
# '''计算预测正确率'''
# predictions = predict(final_theta, X)
# correct = [1 if a==b else 0 for (a, b) in zip(predictions, y)]
# accuracy = sum(correct) / len(X)
# print(accuracy)