import pandas as pd
#读取文件
a=pd.read_csv("NBA.csv",encoding='gbk')
a_1=a.drop(["born","birth_city","birth_state"],axis=1)#删除数据
print(a_1.head())

#元素获取,最好选取iloc方法
#print(a_1["Player","1"])#先列后行
print(a_1.iloc[2,3])#先行后列
print(a_1.iloc[0:2,a_1.columns.get_indexer(["Player"])])#即通过坐标获取，又通过列名获取

#排序
b=a_1.sort_values(by=["height"],ascending=True)#按照对应列名进行排序，升序
print(b.head())
b_1=a_1.sort_index()#索引排序
print(b_1.head())

#pandas加法和减法
print(a_1["height"].add(10).head())  #减法则是a_1["height"].sub(10)

#逻辑运算
bbb=a_1[a_1["height"]>180].head()      #取大于180的数据
print(bbb)

#逻辑运算函数query(),isin()
cc=a_1.query("height>180").head()#判断在这范围内的数据
print(cc)
cc=a_1[a_1["height"].isin([180,190])].head()#判断等于这两个值得数据，并输出
print(cc)

#统计运算 describe
print(a_1.describe())#返回该数据的平均值，标准差等等数据
#统计函数
#a_1.std() 标准差，a_1.var() 方差, a_1.median()中位数，a_1.mean() 平均数， a_1.idxmin()索引最大值，a_1.idxmax()索引最小值
#默认按列运算，axis=0,         axis=1则是按行计 算      例如a_1.std(1)就是按行
print(a_1.std(1))

#累计统计函数
#a_1.cumsum()前n个和;
import matplotlib.pyplot as plt
s=a_1["height"].head().cumprod().plot()
plt.show()

#自定义函数 apply()
s=a_1[["height","weight"]].apply(lambda x:x.max()-x.min() , axis=0)#按列计算 lambda 匿名函数
print(s)


#pandas的画图函数plot(kind="") 默认kind='line'折线图，kind='barh'柱状图，等等
