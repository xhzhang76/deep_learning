import numpy as np
import matplotlib.pyplot as plt


# def AND(x1, x2):
#     w1, w2, theta = 0.5, 0.5, 0.7
#     tmp = w1 * x1 + w2 * x2
#     if tmp <= theta:
#         return 0
#     else:
#         return 1

# AND/NAND/OR单层感知机，XOR二层感知机
# 组合NAND，感知机实现计算机


def AND(x1, x2):
    x = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.7
    tmp = np.sum(weight * x) + bias
    if tmp <= 0:
        return 0
    else:
        return 1


def NAND(x1, x2):
    x = np.array([x1, x2])
    weight = np.array([-0.5, -0.5])
    bias = 0.7
    tmp = np.sum(weight * x) + bias
    if tmp <= 0:
        return 0
    else:
        return 1


def OR(x1, x2):
    x = np.array([x1, x2])
    weight = np.array([0.5, 0.5])
    bias = -0.2
    tmp = np.sum(weight * x) + bias
    if tmp <= 0:
        return 0
    else:
        return 1


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


# 输入层-中间层（隐藏层）-输出层
# y=h(w1*x1+w2*x2+b)   h激活函数（阶跃-感知机，非阶跃-神经网络）
# sigmoid：h(x)=1/(1+exp(-x))
# ReLU(Rectified Linear Unit)
# 输出层激活函数：回归问题用恒等函数，分类问题用softmax函数


# def step_function(x):
#     """
#     Only support float parameter
#     """
#     if x > 0:
#         return 1
#     else:
#         return 0


# def step_function(x):
#     y = x > 0
#     return y.astype(np.int)


def step_function(x):
    return np.array(x > 0, dtype=int)


def sigmoid(x):
    return 1/(1+np.exp(-x))


def relu(x):
    return np.maximum(0, x)


def identity_function(x):
    return x


# maybe overflow
# def softmax(a):
#     exp_a = np.exp(a)
#     sum_exp_a = np.sum(exp_a)
#     y = exp_a / sum_exp_a
#     return y


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


if __name__ == '__main__':
    print(XOR(0, 0))
    print(XOR(0, 1))
    print(XOR(1, 0))
    print(XOR(1, 1))

    x = np.arange(-30.0, 30.0, 0.1)
    y1 = step_function(x)
    y2 = sigmoid(x)
    y3 = relu(x)
    plt.plot(x, y1)
    plt.plot(x, y2)
    plt.plot(x, y3)
    plt.ylim(-0.1, 1.1)
    plt.show()

    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])
    A1 = np.dot(X, W1) + B1  # [0.3 0.7 1.1]
    Z1 = sigmoid(A1)  # [0.57444252 0.66818777 0.75026011]

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2  # [0.51615984 1.21402696]
    Z2 = sigmoid(A2)  # [0.62624937 0.7710107 ]

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3  # [0.31682708 0.69627909]
    Y = identity_function(A3)  # [0.31682708 0.69627909]

    a = np.array([0.3, 2.9, 4.0])
    Y = softmax(a)  # [0.01821127 0.24519181 0.73659691]


