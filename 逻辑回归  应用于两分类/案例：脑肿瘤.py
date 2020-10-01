import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import  classification_report ,roc_auc_score #模型评估报告 和 auc指标
import ssl
ssl._create_default_https_context =ssl._create_unverified_context
name=['Sample code number','Clump Thickness','Uniformity of Cell Size','Uniform of Cell Shape','Marginal Adhesion',
      'Single Epithelial Cell Size','Bare Nuclei','Bland Chromatin','Normal Nucleoli','Mitoses','Class'
      ]
datapd=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',names=name)
print(datapd)


# 2.数据处理
# 2.1 处理缺失值
datapd =datapd.replace(to_replace='?',value=np.nan)  #将？替换成 Nan
datapd=datapd.dropna()  #删除Nan

#2.2确定特征值x，目标值y
x=datapd.iloc[:,1:-1]
y=datapd["Class"]

# 2.3 分割数据
x_train,x_test,y_train,y_test=train_test_split( x, y, random_state=22,test_size=0.2)

#3特征工程
Stan=StandardScaler()
x_train=Stan.fit_transform(x_train)
x_test=Stan.fit_transform(x_test)

#4机器学习
Logistic=LogisticRegression()
Logistic.fit(x_train,y_train)

#5模型评估

print("准确率为：",Logistic.score(x_test,y_test))

y_pre=Logistic.predict(x_test)
print("预测值：",y_pre)

report=classification_report(y_test,y_pre,labels=(2,4),target_names=("良性","恶性")) #label中的值是 两个目标值  target_name是对应目标值的名字
print("评估报告：",report)

#auc指标  当数据不均衡时 需要使用这以指标来评估  正例与假例 比 4:1
# 首先需要将数据改成0 和1
y_test=np.where(y_test>3 ,1,0)
auc=roc_auc_score(y_test,y_pre)
print("auc指标为",auc)