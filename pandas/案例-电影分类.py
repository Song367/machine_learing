import pandas as pd
import numpy as np
#读取文件
a=pd.read_csv('movie.csv',usecols=["genres"])
print(a)
#获取每一列的分类
index_movie=[i.split('|') for i in a["genres"]]
print(index_movie)
#获取所有电影类别 unique去重
temp_list=np.unique([y for x in index_movie for y in x])
print(temp_list)

#生成一个全为0的数组
zero=np.zeros([a.shape[0],temp_list.shape[0]]).astype(int)
print(zero)
#s生成dataframe

dat=pd.DataFrame(zero,columns=temp_list)
print(dat)

#更改数据中的0
#首先需要获取具体位置

for i in range(26):
    xx = temp_list[i]
    print(xx[0][:])

# for j in range(5043):
#
#     dat.iloc[j,xx[j]]=1
#     print(j)
    #dat.iloc[j,dat[[temp_list[j]]]]=1
print(dat)
#print(dat)

