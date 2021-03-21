import numpy as np
import pandas as pd

from datetime import timedelta


# 创建
s1 = pd.Series([12, 4, -7, 8])
s2 = pd.Series([12, 4, -7, 8], index=['a', 'b', 'c', 'd'])

arr = np.array([1, 2, 3, 4])
s3 = pd.Series(arr)  # 从 Numpy 中创建的 Series，只是引用，对 Series 中值操作的影响会直接反应到原始的 Numpy 中

dic = {'red': 2000, 'blue': 1000, 'green': 100, 'orange': 500}
s4 = pd.Series(dic)

# 查看
print(s1[1])  # 4
print(s2['b'])  # 4
print(s4.index)  # Index(['red', 'blue', 'green', 'orange'], dtype='object')
print(s3.values)  # [1 2 3 4]

# 选取
print(s2[0:2])
# [1 2 3 4]
# a    12
# b     4
# dtype: int64
print(s2[[0, 2]])
# a    12
# c    -7
# dtype: int64
print(s2[['a', 'd']])
# a    12
# d     8
# dtype: int64
print(s2[s2 > 0])
# a    12
# b     4
# d     8
# dtype: int64

# 赋值
s2['a'] = 12
print(s2)

# 去重
s5 = pd.Series([1, 3, 5, 1, 2, 5, 3, 4, 7])
print(s5.unique())  # [1 3 5 2 4 7]
# 统计
print(s5.value_counts())
# normalize=True, sort=True, ascending=True, bins=
# 1    2
# 3    2
# 5    2
# 2    1
# 4    1
# 7    1
# dtype: int64
# 是否存在
print(s5.isin([1, 3]))
# 0     True
# 1     True
# 2    False
# 3     True
# 4    False
# 5    False
# 6     True
# 7    False
# 8    False
# dtype: bool
print(s5[s5.isin([1, 3])])
# 0    1
# 1    3
# 3    1
# 6    3
# dtype: int64
# 空值
s6 = pd.Series([1, 3, np.NaN, 7])
print(s6.isnull())
print(s6.isna())
print(s6[s6.isnull()])
# 0    False
# 1    False
# 2     True
# 3    False
# dtype: bool


