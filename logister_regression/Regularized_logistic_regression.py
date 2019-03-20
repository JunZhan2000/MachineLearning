import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def feature_mapping(x1, x2, power):
    '''特征映射函数，即获取很多高阶的特征量'''

    data = {}    # 创建一个新的字典
    for i in np.arange(power + 1):    # power指x1与x2的系数和的最大值
        for p in np.arange(i + 1):    # 两个嵌套循环产生了1+7+6+5+4+3+2+1个特征
            data["f{0:d}{1:d}".format(i - p, p)] = np.power(x1, i - p) * np.power(x2, p)    # 每次执行这行代码，添加一列

    return pd.DataFrame(data)


def sigmoid(z):
    '''sigmoid函数'''
    return 1 / (1 + np.exp(-z))    #返回函数值


def compute_cost(X, y, theta, l):
    '''计算代价的函数'''

    h_x = sigmoid(np.dot(X, theta))    # 计算由假设函数得到的预测值，exp函数为以e为底的指数函数
    # 因为theta是数组对象，所以无需转置
    Cost_x = y * np.log(h_x) + (1 - y) * np.log(1 - h_x)    # 实现Cost函数
    _theta = theta[1:]    # 不惩罚第一项
    reg = ( l / (2 * X.shape[0]) ) * (np.dot(_theta, _theta.T))    # 惩罚项

    return -np.mean(Cost_x) + reg # 返回代价值


def gradient_descent(X, y, theta, alpha, epoch, l):
    '''梯度下降算法'''

    m = X.shape[0]    # 样本数量
    theta0 = theta[0]
    _theta = theta[1:]    # 除了theta[0]以外的所有theta参数
    X0 = X[:, 0:1]    # X0特征量
    _X = X[:, 1:]    # 除了X0以外的所有特征量
    for i in range(epoch):
        theta0 = theta0 - (alpha / m) * np.sum(sigmoid(np.dot(X0, theta0)) - y)    # 不惩罚第一项
        _theta = _theta - (alpha / m) * ( (sigmoid(np.dot(_X, _theta)) - y) @ _X + l * _theta)    # 惩罚其他theta参数

    return np.hstack((theta0, _theta))


def predict(theta, X):
    '''计算假设函数与数据集拟合程度'''

    probability = sigmoid(X@theta)
    return [1 if x >= 0.5 else 0 for x in probability]  # return a list


'''获取数据'''
path = r'ex2data2.txt'    # 数据集的路径
data = pd.read_csv(path, header=None, names=['Test1', 'Test2', 'if_accepted'])


'''数据可视化，数据集的散点图'''
positive = data[data.if_accepted == 1]    # 合格的芯片
nagetive = data[data.if_accepted == 0]    # 不合格的芯片
fig, ax = plt.subplots(figsize = (8, 5))    # 画板和画纸
ax.scatter(positive['Test1'], positive['Test2'], c = 'r')    # 用红色的圆圈绘制出合格的点
ax.scatter(nagetive['Test1'], nagetive['Test2'], s = 30, c = 'b', marker = 'x')    # 用蓝色的三角形绘制出不合格的点
# 设置图例
ax.legend()
# 设置横纵坐标轴名字
ax.set_xlabel('Test1')
ax.set_ylabel('Test2')
plt.show()


'''特征缩放，获取X, y, theta'''
x1 = data['Test1'].values
x2 = data['Test2'].values
mapped_data = feature_mapping(x1, x2, 6)
X = mapped_data.values    # 缩放后的特征量
y = data['if_accepted'].values    # 实际值
theta = np.zeros(X.shape[1])    # theta系数



'''进行梯度下降'''
alpha = 0.03
epoch = 100000
l = 0.01
final_theta = gradient_descent(X, y, theta, alpha, epoch, l)


'''比较代价'''
print("初始代价：")
print(compute_cost(X, y, theta, l))
print("最终代价：")
print(compute_cost(X, y, final_theta, l))

'''计算拟合程度'''
predictions = predict(final_theta, X)
correct = [1 if a==b else 0 for (a, b) in zip(predictions, y)]
accuracy = sum(correct) / len(correct)
print("精度：")
print(accuracy)


'''一下这一块是我直接抄的代码，通过画等高线图的方式把决策边界画出来
但我目前无法理解这段代码，暂时先放着'''
'''绘制决策边界'''
x = np.linspace(-1, 1.5, 250)
xx, yy = np.meshgrid(x, x)
z = np.matrix(feature_mapping(xx.ravel(), yy.ravel(), 6))
z = z @ final_theta
z = z.reshape(xx.shape)

def plot_data():
    positive = data[data['if_accepted'].isin([1])]
    negative = data[data['if_accepted'].isin([0])]

    fig, ax = plt.subplots(figsize=(8,5))
    ax.scatter(positive['Test1'], positive['Test2'], s=50, c='b', marker='o', label='Accepted')
    ax.scatter(negative['Test1'], negative['Test2'], s=50, c='r', marker='x', label='Rejected')
    ax.legend()
    ax.set_xlabel('Test 1 Score')
    ax.set_ylabel('Test 2 Score')

plot_data()
plt.contour(xx, yy, z, 0, colors = 'yellow')
plt.ylim(-0.8, 1.2)
plt.show()