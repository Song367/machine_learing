#特征预处理
#解释：将特征值数据转换成机器容易识别的数据 例如0-1

#1.归一化
#计算方式：
#x'=(x-min)/(max-min)  x''=x'*(mx-mi)+mi
#数据解释
#x是当前数据 也就是特征值中需要计算的当前数据  min,max是特征值中最小值和最大值  mx,mi是你选取的范围最大值和最小值。默认为1和0
#影响
#容易受到最大值和最小值的影响

#2.标准化
#是将数据转换为均值为0，标准差为1范围内
#计算公式：
#x'=(x-mean)/std
#数据解释
#x当前值  mean是一列平均值 std标准差
#标准化比这个更不容易受到影响
#适合大数据场景
from sklearn.datasets import load_iris #获取数据集
from sklearn.preprocessing import MinMaxScaler#归一化
from sklearn.preprocessing import StandardScaler#标准化
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
load_file=load_iris()
pd_data=pd.DataFrame(load_file.data,columns=['sepal length ','sepal width','petal length','petal width'])

#归一化
def guiyi():
    one=MinMaxScaler(feature_range=(0,1))#范围是0-1
    data=one.fit_transform(pd_data)#进行转换
    print(data)
    return data
#s_data=guiyi()

#标准化
def standard():
    stan=StandardScaler()
    data=stan.fit_transform(pd_data)
    print(data,'\n')
    print('平均值',stan.mean_)
    print('方差',stan.var_)
    return data
s_data= standard()
data1=pd.DataFrame(s_data,columns=['sepal length ','sepal width','petal length','petal width'])#为了画图
print(data1)
data1["target"]=load_file.target
def plot_tu(data,col1,col2):
    sns.lmplot(x=col1,y=col2,data=data,hue="target",fit_reg=False)#如果fit_reg=true 则图中有一条线 如果没有目标值，那将所有数据属于一个类别pycharm也自动生成只一个颜色
    plt.show()

plot_tu(data1,'sepal length ','sepal width')