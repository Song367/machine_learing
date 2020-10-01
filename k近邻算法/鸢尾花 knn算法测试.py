from sklearn.datasets import load_iris#获取数据
from sklearn.preprocessing import StandardScaler#标准化
from sklearn.model_selection import train_test_split#数据集分割
from  sklearn.neighbors import KNeighborsClassifier#knn算法实现

#第一步
#获取鸢尾花数据
ywh=load_iris()

#第二步
x_train,x_test,y_train,y_test=train_test_split(ywh.data,ywh.target,test_size=0.2,random_state=20)

#第三步 特征工程 标准化
data_stan=StandardScaler()
x_train=data_stan.fit_transform(x_train)
x_test=data_stan.fit_transform(x_test)

#第四步 模型训练
model_train=KNeighborsClassifier(n_neighbors=9)
model_train.fit(x_train,y_train)
result=model_train.predict(x_test)
#第五步 模型评估
print('测试后的数据为',result)
print('是否准确',result==y_test)

#准确率
print('准确率为:',model_train.score(x_test,y_test))


