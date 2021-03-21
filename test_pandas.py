import pandas as pd

data = {
    '性别': ['男', '女', '女', '男', '男'],
    '姓名': ['小明', '小红', '小芳', '大黑', '张三'],
    '年龄': [20, 21, 25, 24, 29]}
df = pd.DataFrame(data, index=['one', 'two', 'three', 'four', 'five'],
                  columns=['姓名', '性别', '年龄', '职业'])
print(df)
#        姓名 性别  年龄   职业
# one    小明  男  20  NaN
# two    小红  女  21  NaN
# three  小芳  女  25  NaN
# four   大黑  男  24  NaN
# five   张三  男  29  NaN
print(df['姓名'])
print(df.head(2))
print(df.tail(2))
print(df.iloc[:, 1:3])
#       性别  年龄
# one    男  20
# two    女  21
# three  女  25
# four   男  24
# five   男  29
print(df.values, type(df.values))
# [['小明' '男' 20 nan]
#  ['小红' '女' 21 nan]
#  ['小芳' '女' 25 nan]
#  ['大黑' '男' 24 nan]
#  ['张三' '男' 29 nan]]
# <class 'numpy.ndarray'>
print(df.index)
# Index(['one', 'two', 'three', 'four', 'five'], dtype='object')
print(df.columns)
# Index(['姓名', '性别', '年龄', '职业'], dtype='object')
print(df.axes)
# [Index(['one', 'two', 'three', 'four', 'five'], dtype='object'), Index(['姓名', '性别', '年龄', '职业'], dtype='object')]
print(df.T)
#     one  two three four five
# 姓名   小明   小红    小芳   大黑   张三
# 性别    男    女     女    男    男
# 年龄   20   21    25   24   29
# 职业  NaN  NaN   NaN  NaN  NaN
print(df.info())
# <class 'pandas.core.frame.DataFrame'>
# Index: 5 entries, one to five
# Data columns (total 4 columns):
#  #   Column  Non-Null Count  Dtype
# ---  ------  --------------  -----
#  0   姓名      5 non-null      object
#  1   性别      5 non-null      object
#  2   年龄      5 non-null      int64
#  3   职业      0 non-null      object
# dtypes: int64(1), object(3)
# memory usage: 372.0+ bytes
# None
print(df.describe())
#               年龄
# count   5.000000
# mean   23.800000
# std     3.563706
# min    20.000000
# 25%    21.000000
# 50%    24.000000
# 75%    25.000000
# max    29.000000
# 按行遍历
for index, row in df.iterrows():
    print(index, row) # 输出每行的索引值
# one 姓名     小明
# 性别      男
# 年龄     20
# 职业    NaN
# Name: one, dtype: object
# two 姓名     小红
# 性别      女
# 年龄     21
# 职业    NaN
# Name: two, dtype: object
# three 姓名     小芳
# 性别      女
# 年龄     25
# ......
for row in df.itertuples():
    print(row, getattr(row, '姓名'))
# Pandas(Index='one', 姓名='小明', 性别='男', 年龄=20, 职业=nan) 小明
# Pandas(Index='two', 姓名='小红', 性别='女', 年龄=21, 职业=nan) 小红
# Pandas(Index='three', 姓名='小芳', 性别='女', 年龄=25, 职业=nan) 小芳
# Pandas(Index='four', 姓名='大黑', 性别='男', 年龄=24, 职业=nan) 大黑
# Pandas(Index='five', 姓名='张三', 性别='男', 年龄=29, 职业=nan) 张三

# 按列遍历
for index, row in df.iteritems():
    print(index, row)
# 姓名 one      小明
# two      小红
# three    小芳
# four     大黑
# five     张三
# Name: 姓名, dtype: object
# 性别 one      男
# two      女
# three    女
# four     男
# five     男
# Name: 性别, dtype: object
# 年龄 one      20
# ......