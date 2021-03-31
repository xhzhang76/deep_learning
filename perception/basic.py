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
    return 1 / (1 + np.exp(-x))


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


def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t) ** 2)


# one-hot
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(t * np.log(y+delta)) / batch_size


# none one-hot
def cross_entropy_error2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    delta = 1e-7
    return -np.sum(np.log(y[np.arange(batch_size), t]+delta)) / batch_size


def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h)-f(x-h)) / (2*h)


def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x)
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)

        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1-fxh2)/(2*h)
        x[idx] = tmp_val
    return grad


def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr*grad
    return x


def function_2(x):
    return x[0]**2 + x[1]**2


if __name__ == '__main__':
    # print(XOR(0, 0))
    # print(XOR(0, 1))
    # print(XOR(1, 0))
    # print(XOR(1, 1))
    #
    # x = np.arange(-30.0, 30.0, 0.1)
    # y1 = step_function(x)
    # y2 = sigmoid(x)
    # y3 = relu(x)
    # plt.plot(x, y1)
    # plt.plot(x, y2)
    # plt.plot(x, y3)
    # plt.ylim(-0.1, 1.1)
    # plt.show()
    #
    # X = np.array([1.0, 0.5])
    # W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    # B1 = np.array([0.1, 0.2, 0.3])
    # A1 = np.dot(X, W1) + B1  # [0.3 0.7 1.1]
    # Z1 = sigmoid(A1)  # [0.57444252 0.66818777 0.75026011]
    #
    # W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    # B2 = np.array([0.1, 0.2])
    # A2 = np.dot(Z1, W2) + B2  # [0.51615984 1.21402696]
    # Z2 = sigmoid(A2)  # [0.62624937 0.7710107 ]
    #
    # W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    # B3 = np.array([0.1, 0.2])
    # A3 = np.dot(Z2, W3) + B3  # [0.31682708 0.69627909]
    # Y = identity_function(A3)  # [0.31682708 0.69627909]
    #
    # a = np.array([0.3, 2.9, 4.0])
    # Y = softmax(a)  # [0.01821127 0.24519181 0.73659691]

    # t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    # y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
    # print(mean_squared_error(np.array(y), np.array(t)))
    # print(cross_entropy_error(np.array(y), np.array(t)))

    # result = numerical_gradient(function_2, np.array([3.0, 4.0]))
    # result1 = numerical_gradient(function_2, np.array([0.0, 0.0]))
    # result2 = numerical_gradient(function_2, np.array([0.0, 1.0]))
    # result3 = numerical_gradient(function_2, np.array([1.0, 0.0]))
    # result4 = numerical_gradient(function_2, np.array([1.0, 1.0]))
    # print(result1)
    # print(result2)
    # print(result3)
    # print(result4)

    init_x = np.array([-3.0, 4.0])
    result = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=200)
    print(result)
