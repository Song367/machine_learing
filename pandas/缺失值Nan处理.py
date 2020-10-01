import pandas as pd
import numpy as np
a=pd.read_csv("movie.csv",usecols=["color","director_name","budget","title_year"]).head()

print(a)
#判断是否是缺失值  isnull()  True代表有缺失值
print(a.isnull())

#删除缺失值 dropna()
print(a.dropna())

#替换缺失值 fillna()
a["color"].fillna("blue",inplace=True)#替换某一列，fillna第一个参数是替换的元素，inplace=True 是替换原来的数据，flase是生成新的数据
print(a)

#np.all()判断数据里如果有一个false则返回false 在此例子代表有缺失值  np.any()则相反True代表有缺失值
print(np.any(pd.isnull(a)))

#当缺失值不是Nan时，用pd.replace()替换成Nan

#pd.replace(to_replace="?",value=np.nan) to_replace时替换前的值，value是替换后的值。

