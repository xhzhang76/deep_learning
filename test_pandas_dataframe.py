# row index, column label/column
# 创建
import numpy as np
import pandas as pd

myDict = {
    'color': ['blue', 'green', 'yellow', 'red', 'white'],
    'object': ['ball', 'pen', 'pencil', 'paper', 'mug'],
    'price': [1.2, 1.0, 0.5, 0.8, 1.5]
}
df = pd.DataFrame(myDict)
print(df)
#     color  object  price
# 0    blue    ball    1.2
# 1   green     pen    1.0
# 2  yellow  pencil    0.5
# 3     red   paper    0.8
# 4   white     mug    1.5
df = pd.DataFrame(myDict)  # , index=['one', 'two', 'three', 'four', 'five'])
print(df)

arr = np.arange(16).reshape((4, 4))
df1 = pd.DataFrame(arr, index=['A', 'B', 'C', 'D'], columns=['A', 'B', 'C', 'D'])
print(df1)
#     A   B   C   D
# A   0   1   2   3
# B   4   5   6   7
# C   8   9  10  11
# D  12  13  14  15

# 查看
print(df1.index)  # Index(['A', 'B', 'C', 'D'], dtype='object')
print(df1.columns)  # Index(['A', 'B', 'C', 'D'], dtype='object')
print(df1.values)
# [[ 0  1  2  3]
#  [ 4  5  6  7]
#  [ 8  9 10 11]
#  [12 13 14 15]]

# 选择
print(df[['price', 'color']])
#        price   color
# one      1.2    blue
# two      1.0   green
# three    0.5  yellow
# four     0.8     red
# five     1.5   white
print(df.loc[2])
# color     yellow
# object    pencil
# price        0.5
# Name: 2, dtype: object

# 切片
print(df[1:3])
#     color  object  price
# 1   green     pen    1.0
# 2  yellow  pencil    0.5

# 过滤
print(df[df['price'] > 1])
#    color object  price
# 0   blue   ball    1.2
# 4  white    mug    1.5

# 赋值
df['new'] = 12
print(df)
#     color  object  price  new
# 0    blue    ball    1.2   12
# 1   green     pen    1.0   12
# 2  yellow  pencil    0.5   12
# 3     red   paper    0.8   12
# 4   white     mug    1.5   12
df['new'] = [3, 4, 5, 6, 7]
print(df)

# 删除列
del df['new']

# lambda
arr = np.arange(12).reshape((3, 4)) ** 2
df2 = pd.DataFrame(arr)
df2 = df2.apply(lambda x: np.sqrt(x))
print(df2)
#      0    1     2     3
# 0  0.0  1.0   2.0   3.0
# 1  4.0  5.0   6.0   7.0
# 2  8.0  9.0  10.0  11.0

# 统计
print(df.sum(numeric_only=True))
# price    5.0
# dtype: float64
print(df.mean())
# price    1.0
# dtype: float64
print(df.describe())
#           price
# count  5.000000
# mean   1.000000
# std    0.380789
# min    0.500000
# 25%    0.800000
# 50%    1.000000
# 75%    1.200000
# max    1.500000

# 排序
print(df)
print(df.sort_index())
print(df.sort_values(by=['price'], ascending=True))
#     color  object  price          color  object  price           color  object  price
# 0    blue    ball    1.2      0    blue    ball    1.2       2  yellow  pencil    0.5
# 1   green     pen    1.0      1   green     pen    1.0       3     red   paper    0.8
# 2  yellow  pencil    0.5      2  yellow  pencil    0.5       1   green     pen    1.0
# 3     red   paper    0.8      3     red   paper    0.8       0    blue    ball    1.2
# 4   white     mug    1.5      4   white     mug    1.5       4   white     mug    1.5

# 读写
df3 = pd.read_csv("recent15.csv")
print(df3[df3['Percent'] > 1])
print(df3.sort_values(by='Percent', ascending=False))
df.to_csv("test.csv")

# 数据分析
myDict1 = {
    'id': ['ball', 'pencil', 'pen', 'mug', 'ashtray'],
    'price': [12.33, 11.44, 33.21, 13.23, 33.62]
}
myDict2 = {
    'id': ['pencil', 'pencil', 'ball', 'pen'],
    'color': ['white', 'red', 'red', 'black']
}
df1 = pd.DataFrame(myDict1)
df2 = pd.DataFrame(myDict2)
print(df1)
print(df2)
print(pd.merge(df1, df2))
#         id  price               id  color               id  price  color
# 0     ball  12.33        0  pencil  white        0    ball  12.33    red
# 1   pencil  11.44        1  pencil    red        1  pencil  11.44  white
# 2      pen  33.21        2    ball    red        2  pencil  11.44    red
# 3      mug  13.23        3     pen  black        3     pen  33.21  black
# 4  ashtray  33.62

myDict1 = {
    'id': ['ball', 'pencil', 'pen', 'mug', 'ashtray'],
    'color': ['white', 'red', 'red', 'black', 'green'],
    'brand': ['OMG', 'ABC', 'ABC', 'POD', 'POD']
}
myDict2 = {
    'id': ['pencil', 'pencil', 'ball', 'pen'],
    'brand': ['OMG', 'POD', 'ABC', 'POD']
}
df1 = pd.DataFrame(myDict1)
df2 = pd.DataFrame(myDict2)
print(df1)
print(df2)
print(pd.merge(df1, df2))
print(pd.merge(df1, df2, on='id'))
#         id  color brand             id brand      Empty DataFrame                        id  color brand_x brand_y
# 0     ball  white   OMG      0  pencil   OMG      Columns: [id, color, brand]     0    ball  white     OMG     ABC
# 1   pencil    red   ABC      1  pencil   POD      Index: []                       1  pencil    red     ABC     OMG
# 2      pen    red   ABC      2    ball   ABC                                      2  pencil    red     ABC     POD
# 3      mug  black   POD      3     pen   POD                                      3     pen    red     ABC     POD
# 4  ashtray  green   POD

myDict1 = {
    'id': ['ball', 'pencil', 'pen', 'mug', 'ashtray'],
    'color': ['white', 'red', 'red', 'black', 'green'],
    'brand': ['OMG', 'ABC', 'ABC', 'POD', 'POD']
}
myDict2 = {
    'sid': ['pencil', 'pencil', 'ball', 'pen'],
    'brand': ['OMG', 'POD', 'ABC', 'POD']
}
df1 = pd.DataFrame(myDict1)
df2 = pd.DataFrame(myDict2)
print(pd.merge(df1, df2, left_on='id', right_on='sid'))

# 连接方式
myDict1 = {
    'id': ['ball', 'pencil', 'pen', 'mug', 'ashtray'],
    'color': ['white', 'red', 'red', 'black', 'green'],
    'brand': ['OMG', 'ABC', 'ABC', 'POD', 'POD']
}
myDict2 = {
    'id': ['pencil', 'pencil', 'ball', 'pen'],
    'brand': ['OMG', 'POD', 'ABC', 'POD']
}
df1 = pd.DataFrame(myDict1)
df2 = pd.DataFrame(myDict2)
print(df1)
print(df2)
print(pd.merge(df1, df2, on='id'))  # 默认内连接
print(pd.merge(df1, df2, on='id', how='outer'))
print(pd.merge(df1, df2, on='id', how='left'))
print(pd.merge(df1, df2, on='id', how='right'))
#         id  color brand             id brand
# 0     ball  white   OMG      0  pencil   OMG
# 1   pencil    red   ABC      1  pencil   POD
# 2      pen    red   ABC      2    ball   ABC
# 3      mug  black   POD      3     pen   POD
# 4  ashtray  green   POD
#
#        id  color brand_x brand_y  |          id  color brand_x brand_y  |          id  color brand_x brand_y  |         id  color brand_x brand_y
# 0    ball  white     OMG     ABC  |  0     ball  white     OMG     ABC  |  0     ball  white     OMG     ABC  |  0  pencil    red     ABC     OMG
# 1  pencil    red     ABC     OMG  |  1   pencil    red     ABC     OMG  |  1   pencil    red     ABC     OMG  |  1  pencil    red     ABC     POD
# 2  pencil    red     ABC     POD  |  2   pencil    red     ABC     POD  |  2   pencil    red     ABC     POD  |  2    ball  white     OMG     ABC
# 3     pen    red     ABC     POD  |  3      pen    red     ABC     POD  |  3      pen    red     ABC     POD  |  3     pen    red     ABC     POD
#                                   |  4      mug  black     POD     NaN  |  4      mug  black     POD     NaN  |
#                                   |  5  ashtray  green     POD     NaN  |  5  ashtray  green     POD     NaN  |

# 基于index连接

df1 = pd.DataFrame(np.arange(9).reshape((3, 3)), index=[1, 2, 3], columns=['A', 'B', 'C'])
df2 = pd.DataFrame(np.arange(9, 18).reshape((3, 3)), index=[3, 5, 6], columns=['A', 'B', 'C'])
print(pd.concat([df1, df2]))
#     A   B   C
# 1   0   1   2
# 2   3   4   5
# 3   6   7   8
# 3   9  10  11
# 5  12  13  14
# 6  15  16  17
ser1 = pd.Series(np.arange(5), index=[1, 2, 3, 4, 5])
ser2 = pd.Series(np.arange(4, 8), index=[2, 4, 5, 6])
print(ser1)
print(ser2)
print(ser1.combine_first(ser2))
print(pd.concat([ser1, ser2]))
# 1    0.0                    1    0
# 2    1.0                    2    1
# 3    2.0                    3    2
# 4    3.0                    4    3
# 5    4.0                    5    4
# 6    7.0                    2    4
# dtype: float64              4    5
#                             5    6
#                             6    7
#                             dtype: int64

# 行转列
myDict = {
    'color': ['white', 'white', 'white', 'red', 'red', 'red', 'black', 'black', 'black'],
    'item': ['ball', 'pen', 'mug', 'ball', 'pen', 'mug', 'ball', 'pen', 'mug'],
    'value': np.arange(9)
}
df = pd.DataFrame(myDict)
print(df)
pdf = df.pivot('color', 'item')
print(pdf)
#    color  item  value                    value
# 0  white  ball      0              item   ball mug pen
# 1  white   pen      1              color
# 2  white   mug      2              black     6   8   7
# 3    red  ball      3              red       3   5   4
# 4    red   pen      4              white     0   2   1
# 5    red   mug      5
# 6  black  ball      6
# 7  black   pen      7
# 8  black   mug      8

# 数据转换
myDict = {
    'color': ['white', 'white', 'red', 'red', 'white'],
    'value': [2, 1, 3, 3, 2]
}
df = pd.DataFrame(myDict)
print(df.duplicated())
print(df[df.duplicated()])
#    color  value
# 3    red      3
# 4  white      2

# 基于map的替换和添加
myDict = {
    'item': ['ball', 'mug', 'pen', 'pencil', 'ashtray'],
    'color': ['white', 'rosso', 'verde', 'black', 'yellow'],
    'price': [5.56, 4.20, 1.30, 0.56, 2.75]
}
df = pd.DataFrame(myDict)
newColor = {'rosso': 'red', 'verde': 'green'}
print(df.replace(newColor))
#       item   color  price
# 0     ball   white   5.56
# 1      mug     red   4.20
# 2      pen   green   1.30
# 3   pencil   black   0.56
# 4  ashtray  yellow   2.75
myDict = {
    'item': ['ball', 'mug', 'pen', 'pencil', 'ashtray'],
    'color': ['white', 'rosso', 'verde', 'black', 'yellow']
}
df = pd.DataFrame(myDict)
prices = {
    'ball': 5.56,
    'mug': 4.20,
    'bottle': 1.30,
    'scissors': 3.41,
    'pen': 1.30,
    'pencil': 0.56,
    'ashtray': 2.75
}
df['price'] = df['item'].map(prices)
print(df)
#       item   color  price
# 0     ball   white   5.56
# 1      mug   rosso   4.20
# 2      pen   verde   1.30
# 3   pencil   black   0.56
# 4  ashtray  yellow   2.75

# 离散化
results = [12, 34, 67, 55, 28, 90, 99, 12, 3, 56, 74, 44, 87, 23, 49, 89, 87]
bins = [0, 25, 50, 75, 100]
cat = pd.cut(results, bins)
print(cat)
# [(0, 25], (25, 50], (50, 75], (50, 75], (25, 50], ..., (75, 100], (0, 25], (25, 50], (75, 100], (75, 100]]
# Length: 17
# Categories (4, interval[int64]): [(0, 25] < (25, 50] < (50, 75] < (75, 100]]
print(cat.codes)
# [0 1 2 2 1 3 3 0 0 2 2 1 3 0 1 3 3]
print(pd.value_counts(cat))  # pd.value_counts(cat.codes), pd.value_counts(results, bins=bins)
# (75, 100]    5
# (0, 25]      4
# (25, 50]     4
# (50, 75]     4
# dtype: int64
results = [12, 34, 67, 55, 28, 90, 99, 12, 3, 56, 74, 44, 87, 23, 49, 89, 87]
bins = [0, 25, 50, 75, 100]
bin_names = ['unlikely', 'less likely', 'likely', 'highly likely']
cat = pd.cut(results, bins, labels=bin_names)
print(cat)
# ['unlikely', 'less likely', 'likely', 'likely', 'less likely', ..., 'highly likely', 'unlikely', 'less likely', 'highly likely', 'highly likely']
# Length: 17
# Categories (4, object): ['unlikely' < 'less likely' < 'likely' < 'highly likely']
# 根据指定区间的数量，以及该列值的最大值和最小值，切割成若干等分，然后进行划分
results = [12, 34, 67, 55, 28, 90, 99, 12, 3, 56, 74, 44, 87, 23, 49, 89, 87]
bin_names = ['unlikely', 'less likely', 'likely', 'highly likely']
cat = pd.cut(results, 4, labels=bin_names)
print(cat)
# ['unlikely', 'less likely', 'likely', 'likely', 'less likely', ..., 'highly likely', 'unlikely', 'less likely', 'highly likely', 'highly likely']
# Length: 17
# Categories (4, object): ['unlikely' < 'less likely' < 'likely' < 'highly likely']
# 指定区间数量，根据数据散布的特点，确保每个区间最终划分的值的数量相同，至于区间边界，不一定是等分的
results = [12, 34, 67, 55, 28, 90, 99, 12, 3, 56, 74, 44, 87, 23, 49, 89, 87, 20]
bin_names = ['unlikely', 'likely', 'highly likely']
cat1 = pd.cut(results, 3, labels=bin_names)
cat2 = pd.qcut(results, 3, labels=bin_names)
print(pd.value_counts(cat1))
print(pd.value_counts(cat2))
# unlikely         7      unlikely         6
# highly likely    6      likely           6
# likely           5      highly likely    6
# dtype: int64            dtype: int64

# 随机采样
myDict = {
    'item': ['ball', 'mug', 'pen', 'pencil', 'ashtray'],
    'color': ['white', 'rosso', 'verde', 'black', 'yellow'],
    'price': [5.56, 4.20, 1.30, 0.56, 2.75]
}
df = pd.DataFrame(myDict)
sample = np.random.randint(0, len(df), size=3)
print(df.take(sample))

# 数据聚合
# - 拆分 —— 将数据集拆分成不同的分组
# - 处理 —— 对每个分组分别进行处理
# - 合并 —— 将每个分组处理的结果合并成最终的结果
myDict = {
    'color': ['white', 'red', 'green', 'red', 'green'],
    'object': ['pen', 'pencil', 'pencil', 'ashtray', 'pen'],
    'price1': [5.56, 4.20, 1.30, 0.56, 2.75],
    'price2': [4.75, 4.12, 1.60, 0.75, 3.15]
}
df = pd.DataFrame(myDict)
group = df['price1'].groupby(df['color'])
print(group.groups)
# {'green': [2, 4], 'red': [1, 3], 'white': [0]}
print(group.mean())
# color
# green    2.025
# red      2.380
# white    5.560
# Name: price1, dtype: float64


# 设置index
df = pd.DataFrame({'month': [1, 4, 7, 10],
                   'year': [2012, 2014, 2013, 2014],
                   'sale': [55, 40, 84, 31]})
print(df)
df.set_index('year', drop=False, inplace=True)
df.to_csv('/Users/xhzhang76/123.csv')
print(df)

# 抽取几列生成新的dataframe
df1 = df[['year', 'sale']]
print(df1)

# 斜率
data = {'slopes': {0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0},
        'time': {0: 1, 1: 2, 2: 3, 3: 4, 4: 5},
        'val': {0: 11, 1: 11, 2: 11, 3: 6, 4: 5}}

df = pd.DataFrame(data)

df['diff'] = (df.val-df.val.shift(1))
series = df.val.diff().fillna(0)

print(df)
