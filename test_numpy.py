import numpy as np

# 一维——向量，二维——矩阵，一般化统称：张量（tensor）
a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 6], [7, 8]])
print(a)
print(a.shape)  # (2,2)
print(a.dtype)  # int64
print(a.ndim)  # 2
print(np.dot(a, b))
# [[19 22]
#  [43 50]]

# 广播
b = np.array([10, 20])
print(a * b)
# [[10 40]
#  [30 80]]

# slice
x = np.array([[0, 1], [2, 3], [4, 5]])
print(x[0])  # [0,1]
print(x[1, 1])  # 3
for row in x:
    print(row)
print(x.flatten())  # [0 1 2 3 4 5]

# filter index
print(x >= 3)
# [[False False]
#  [False  True]
#  [ True  True]]
print(x[x >= 3])  # [3 4 5]
