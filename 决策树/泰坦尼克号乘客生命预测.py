import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split       # 数据分割
from sklearn.tree import DecisionTreeClassifier    # 决策树算法
from sklearn.feature_extraction import DictVectorizer   # 字典特征提取
pddata=pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")


#2数据处理
#2.1 获取特征值
x=pddata[["age","sex","pclass"]]

#2.2 获取目标值
y=pddata["survived"]

# 对数据的空值进行替换 用平均值进行替换   inplace=true  是否对自身进行替换
x['age'].fillna(value=pddata['age'].mean(),inplace=True)
print(x)

#2.3 数据集划分
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=22,test_size=0.2)

#3 字典特征提取
#将数据转换为字典格式
x_train=x_train.to_dict(orient="records")
x_test=x_test.to_dict(orient="records")
print(x_train)

#特征提取
DictV=DictVectorizer()
x_train=DictV.fit_transform(x_train)
x_test=DictV.fit_transform(x_test)


# 4.机器学习
Dec=DecisionTreeClassifier(max_depth=5)
Dec.fit(x_train,y_train)

#模型评估
y_pre=Dec.predict(x_test)
print('预测值：' ,y_pre)

print("准确率：",Dec.score(x_test,y_test))


