import pandas as pd
import numpy as np

# pd.set_option('display.max_columns', None)  # 显示打印的所有列
# pd.set_option('display.max_rows', None)   # 显示打印的所有行

# Series生成一维数据
print(pd.Series(np.arange(9),index=[np.arange(1,10)])) # index是索引;

# Datafarame生成二维数据
grade=np.array(np.random.randint(40,100,[4,5]))
print(grade)
# index为行索引，columns是列索引,   shape[0]代表行，shape[1]代表列；
grade_pd=pd.DataFrame(grade,index=["同学"+ str(i) for i in range(grade.shape[0])],columns=["math","english","basketball","athlete","Chinese"])
print(grade_pd)
print(grade_pd.values)
print(grade_pd.tail(2)) # 获取后两行
# head是获取前几行；
# grade_pd.T   转置

h=pd.DataFrame({"math":[100,90],"english":[80,98],"grade":[88,77]})
s=h.set_index("math") # 以某一列作为索引
print(s)

# multiindex,三维数据,,也就相当于多个索引
arrays=[[1,2,3],['r','w','g']]
a=pd.MultiIndex.from_arrays(arrays,names=("数量","颜色"))
print(a)