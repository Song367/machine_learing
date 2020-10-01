
from sklearn.datasets import load_iris,fetch_20newsgroups #获取数据集
import seaborn as sns  #画图库
import matplotlib.pyplot as plt
import pandas as pd
from pylab import mpl #显示中文
from sklearn.model_selection import train_test_split  #数据集划分
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False
#基于matplotlib的核心库
#因为sklearn下载后他就自带鸢尾花的数据集 所以直接调用load_iris()

#获取数据集
#1.小数据集 load_iris获取本地
s=load_iris()
# print(s)

#2.大数据集 fetch_* 获取从网上下载的数据
#例如获取20类新闻数据集 fetch_20newsGroups(data_home='',subnet='train')  data_home是下载的目录 subnet是选择train训练集 ，test测试集，all全部
p=fetch_20newsgroups()


#数据集的返回值 brunch(字典类型)
#data:特征值  数组
#target:目标值 是一维数组
#DESCR:数据描述
#feature_name:特征名
#target_name:目标名
#print(s.data)
#print(s.feature_names)


#sns.lmplot(x,y,data,hue,fit_leg) data是数据集 hue目标值 fit_leg是否进行线性拟合


#开始
load_d=pd.DataFrame(s.data,columns=['sepal length ','sepal width','petal length','petal width'])
#添加一列，目标值
load_d["target"]=s.target
#print(load_d)
def plot_tu(data,col1,col2):
    sns.lmplot(x=col1,y=col2,data=data,hue="target",fit_reg=False)#如果fit_reg=true 则图中有一条线 如果没有目标值，那将所有数据属于一个类别pycharm也自动生成只一个颜色
    plt.title("鸢尾花数据显示")

    plt.show()

plot_tu(load_d,'sepal length ','sepal width')
#plot_tu(load_d,'sepal width','petal length')


#数据集的划分
#x特征值得测试集合训练集 y是目标值的训练集和测试集
x_train,x_test,y_train,y_test=train_test_split(s.data,s.target,test_size=0.2,random_state=22) #第一个参数是特征值，第二个是目标值，第三个是测试集占的比例 当random_state不一样时，目标值就不一样

print("训练集的特征值：",x_train)
print("训练集的特征值",x_test)
print("测试集的目标值",y_train)
print("测试集的目标值",y_test)
print(y_train.shape)
print(y_test.shape)
