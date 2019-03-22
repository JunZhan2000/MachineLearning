'''
逻辑回归解决多分类问题
数据集划分方法：留出法
精度： 0.973333
查全率： [1.0, 0.96, 0.96]
查准率： [1.0, 0.96, 0.96]
F1 score: [1.0, 0.96, 0.96]
Micro-F1：0.973333
Micro-F2：0.973333
AUC：[1.000000, 0.828400, 0.997600]
'''


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
        tem_y = np.array([1 if label == i else 0 for label in y])    # 获取是i与非i的标签数组
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
        tem_y = np.array([1 if label == i else 0 for label in y ])    # 获取是i与非i的标签数组
        tem_final_theta = gradient_descent(X, tem_y, theta, alpha = 0.03, epoch = 40000)    # 获取优化后的一组theta
        all_theta[i, :] = tem_final_theta    # 更新第i组theta的值

    return all_theta



def predict(all_theta, X):
    '''获得预测结果'''

    h_x = sigmoid(np.dot(X, all_theta.T))
    predictions = np.argmax(h_x, axis=1)    # 获取最大值对应的索引，即label

    return predictions    # 返回预测值



def evaluate(final_theta, X, y, K):
    '''对模型进行性能评估'''

    '''计算精度'''
    predictions = predict(final_theta, X)
    correct = [1 if a == b else 0 for (a, b) in zip(predictions, y)]
    accuracy = sum(correct) / len(X)
    '''计算查全率P，查准率R，F1，Micro-F1和Macro-F1'''
    P = []
    R = []
    sum_P = 0
    sum_R = 0
    F1 = []
    for i in range(K):
        '''对每一个标签计算数值'''
        index = np.array(np.where(y == i))    # 所有i标签的索引
        correct_2 = [1 if example == i else 0 for example in predictions[index][0]]
        tem_R = sum(correct_2) / (len(index[0]))    # 查全率
        R.append(tem_R)
        index_2 = np.array(np.where(predictions == i))    # 预测值中i标签的索引
        correct_3 = [1 if example == i else 0 for example in y[index_2][0]]
        tem_P = sum(correct_3) / (len(index_2[0]))    # 查准率
        P.append(tem_P)
        tem_F1 = (2 * tem_P * tem_R) / (tem_P + tem_R)
        F1.append(tem_F1)
        sum_P += tem_P
        sum_R += tem_R

    mean_P = sum_P / len(P)
    mean_R = sum_R / len(R)

    return accuracy, P, R, F1, (2 * mean_P * mean_R) / (mean_P + mean_R), np.mean(F1)



def expand_y(y):
    '''将y的元素标签全部转换为向量'''
    new_y = []
    for label in y:
        y_array = np.zeros(10)
        y_array[label-1] = 1
        new_y.append(y_array)

    return np.array(new_y)






def get_ROC_AUC(final_theta, X, y, K):
    '''绘制ROC曲线与计算AUC'''

    z = np.dot(X, final_theta.T)
    probability = sigmoid(z)    # 概率矩阵
    # figure, ax = plt.subplot()    # 画板与画纸
    fig, ax = plt.subplots(figsize=(8, 5))
    AUC = [0, 0, 0]
    for i in range(K):
        '''对每一种标签进行一次循环'''
        tem_probability = probability[:, i]
        tem_y = np.array([1 if label == i else 0 for label in y])  # 获取是i与非i的标签数组
        '''将预测值和实际标签一起重新排序'''
        tem_probability = list(tem_probability)
        tem_y = list(tem_y)
        ZIP = zip(tem_probability, tem_y)    # 捆绑
        ZIP = sorted(ZIP, reverse=True)
        tem_probability, tem_y = zip(*ZIP)
        tem_probability = np.array(tem_probability)
        tem_y = np.array(tem_y)
        ''''''
        m_t = np.sum(tem_y == 1)     # 所有的正例
        m_f = np.sum(tem_y == 0)     # 所有的反例
        TPR = np.zeros(y.shape[0]+1)    # 初始化TPR
        FPR = np.zeros(y.shape[0]+1)  # 初始化FPR
        for j in range(y.shape[0]):
            '''每一次循环选择一个点作为阀值'''
            if tem_y[j] == 1:
                '''如果当前为真正例'''
                TPR[j+1] = TPR[j] + 1 / m_t
                FPR[j+1] = FPR[j]
            elif tem_y[j] == 0:
                '''当前为假正例'''
                FPR[j+1] = FPR[j] + 1 / m_f
                TPR[j+1] = TPR[j]
            if j == 48:
                a = 1
        '''绘制曲线'''
        if i == 0:
            the_color = 'r'
            the_label = '0'
        elif i == 1:
            the_color = 'y'
            the_label = '1'
        else:
            the_color = 'b'
            the_label = '2'

        # ax.scatter(FPR, TPR, s=50, c=color, label=the_label)
        ax.plot(FPR, TPR, color=the_color, lw=2, label=the_label)  ###假正率为横坐标，真正率为纵坐标做曲线
        ele_1 = np.zeros(y.shape[0])
        ele_2 = np.zeros(y.shape[0])
        for j in range(y.shape[0]):
            ele_1[j] = FPR[j+1] - FPR[j]
            ele_2[j] = TPR[j+1] + TPR[j]
        AUC[i] = np.dot(ele_1, ele_2.T) / 2



    plt.xlim((0, 1))
    plt.ylim((0, 1))
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC曲线')
    plt.legend()
    plt.show()

    return AUC




# 得到特征，标签，标签种类数
X, y = load_data()
K = len(np.unique(y))
final_theta = get_all_theta(X, y, K)


print("优化后的theta：")
print(final_theta)
# print("初始代价： %f" % compute_all_cost(np.zeros((K, X.shape[1])), X, y))
# print("最终代价： %f" % compute_all_cost(final_theta, X, y))


'''计算查全率P，查准率R，F1，Micro-F1和Macro-F1'''
accuracy, P, R, F1, Micro_F1, Micro_F2 = evaluate(final_theta, X, y, K)
print("精度：%f" % accuracy)
print("查准率："),
print(P)
print("查全率"),
print(R)
print("F1 score："),
print(F1)
print("Micro-F1：%f" % Micro_F1)
print("Micro-F2：%f" % Micro_F2)

'''绘制ROC，计算AUC'''
AUC = get_ROC_AUC(final_theta, X, y, K)
print("AUC：")
for i in range(K):
    print("%5f" % AUC[i])