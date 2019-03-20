'''
注：
在这个logister回归问题中，我实现了梯度下降算法
在NG的视频中以及博客里面提到了一些高级的算法
只需要调用SciPy库中现成的函数即可使用
我尝试使用TNC算法，但是两种方法都出现了 'Linear search failed'的错误
最终我放弃了。。（反正也用不上，傻瓜式用法
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
    '''sigmoid函数'''
    return 1 / (1 + np.exp(-z))    #返回函数值


def compute_cost(theta, X, y):
    '''计算代价的函数'''

    h_x =  sigmoid(np.dot(X, theta))    # 计算由假设函数得到的预测值，exp函数为以e为底的指数函数
    # 因为theta是数组对象，所以无需转置
    Cost_x = y * np.log(h_x) + (1 - y) * np.log(1 - h_x)    # 实现Cost函数

    return -np.mean(Cost_x)    # 返回代价值


def gradient_descent(X, y, theta, alpha, epoch):
    '''梯度下降算法'''

    m = X.shape[0]
    for i in range(epoch):
        # theta = theta - (alpha / m) * (X * theta.T - y).T * X
        # 上面一行是在线性回归中使用的公式，只需替换这一行代码便可以适用于逻辑回归
        theta = theta - (alpha / m) * (sigmoid(np.dot(X, theta)) - y) @ X

    return theta


def gradient(theta, X, y):
    '''计算梯度，即代价函数的偏导数'''

    return (X.T @ (sigmoid(X @ theta) - y))/len(X)


def predict(theta, X):
    '''获得预测结果'''

    probability = sigmoid(np.dot(X, theta))
    return [1 if x >= 0.5 else 0 for x in probability]  # return a list




'''得到数据'''
path = 'ex2data1.txt'    # 数据集的路径
data = pd.read_csv(path, header=None, names = ['grade1', 'grade2', 'if_pass'])    # 以csv文件模式打开文件
print(data.describe())           # 描述数据特征


'''数据可视化'''
positive = data[data.if_pass == 1]
nagetive = data[data.if_pass == 0]
fig, ax = plt.subplots(figsize=(6,5))    # fig是画板，ax是画纸
# 这里其实形成了两个图，用不同的形状标识，展示在同一张画纸上
ax.scatter(positive['grade1'], positive['grade2'], c='b', label='Admitted')
ax.scatter(nagetive['grade1'], nagetive['grade2'], s=50, c='r', marker='x', label='Not Admitted')
# 设置图例显示在图的上方
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width , box.height* 0.8])
ax.legend(loc='center left', bbox_to_anchor=(0.2, 1.12),ncol=3)
# 设置横纵坐标名
ax.set_xlabel('grade1')
ax.set_ylabel('grade2')
plt.show()    #展示数据


# 得到自变量，因变量，参数的数组
data.insert(0, 'ones', 1)    # 在自变量矩阵首列位置插入一列1
X = data.iloc[:, 0:-1].values    # 获得自变量，并将dataframe对象转化为adarray对象
y = data.iloc[:, -1].values    # 获得自变量，并将dataframe对象转化为adarray对象
theta = np.zeros(X.shape[1])


'''梯度下降版本'''
alpha = 0.0025    # 初始化学习率的值
epoch = 200000     # 初始化迭代次数
print("初始：")
print(theta)
print(compute_cost(theta, X, y))


final_theta = gradient_descent(X, y, theta, alpha, epoch)
print("运行梯度下降后：")
print(final_theta)
print(compute_cost(final_theta, X, y))
'''计算预测正确率'''
predictions = predict(final_theta, X)
correct = [1 if a==b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(X)
print(accuracy)


'''绘制决策边界'''
fig, ax = plt.subplots(figsize = (8, 5))
ax.scatter(positive['grade1'], positive['grade2'], c='b', label='Admitted')
ax.scatter(nagetive['grade1'], nagetive['grade2'], s=50, c='r', marker='x', label='Not Admitted')
a = np.linspace(data.grade1.min(), data.grade1.max(), 100)  # 横坐标
f_a = - (final_theta[1] * a + final_theta[0]) / final_theta[2]
ax.plot(a, f_a, 'r', label='decision boundary')
plt.show()


'''调用高级算法函数的失败尝试'''
# '''高级优化算法一'''
# result = opt.fmin_tnc(func=compute_cost, x0=theta, fprime=gradient, args=(X, y))
# print(result[0])
# '''高级优化算法二'''
# result2 = opt.minimize(fun=compute_cost, x0=theta, args=(X, y), method='TNC', jac=gradient)
# final_theta = result2['jac']
# print(result2)
