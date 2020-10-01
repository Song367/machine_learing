import pandas as pd
import numpy as np
a=pd.read_csv("NBA.csv")
c=a["height"]
print(a.head())
#x.fillna("2008",inplace=True)
#print(x.head())
#数据进行分组
#qcut自动进行分组
b=pd.qcut(c,10)
print(b.head())
print(b.value_counts())#生成对应值范围以及个数

#cut指定分组
area=[150,170,180,190,200,210,220,230]
e=pd.cut(c,area)      #area的数组必须是单调递增的
print(e)
print(e.value_counts())

#one-hot编码     当分组后的数据在那个区间时，则为1,否则为0
x=pd.get_dummies(e,prefix='身高区间')
print(x.head())


#数据合并
#concat()
s=pd.concat([a,x],axis=1)
print(s)

#merge(left,right,on=,how=)left边表，right右边表  on是指定合并的键，how的值可以是  left，right，outer,inner 意思分别是以左表为主，以右表为主，外连接，内连接
left=pd.DataFrame({
    'A':['a1','a2','a3','a4'],
    'B':['b1','b2','b3','b4'],
    'ks1':['k0','k0','k1','k2'],
    'ks2':['k1','k2','k0','k0']
})
right=pd.DataFrame({
    'C':['c1','c2','c3','c4'],
    'D':['d1','d2','d3','d4'],
    'ks1':['k1','k0','k1','k2'],
    'ks2':['k1','k2','k0','k1']
})
print(left,'\n',right)
mer=pd.merge(left,right,on=["ks1","ks2"],how="inner")
print(mer)
mer=pd.merge(left,right,on=["ks1","ks2"],how="outer")
print(mer)
mer=pd.merge(left,right,on=["ks1","ks2"],how="left")
print(mer)
mer=pd.merge(left,right,on=["ks1","ks2"],how="right")
print(mer)


#交叉表cross_tab(),返回的是个数                        透视表pivot_table返回的是百分比

#分组聚合 要同时使用，分组就用groupby() 聚合就是那些mean（），std(),count()等等
#例如 pd.read_csv("").groupy(["height"]).count()