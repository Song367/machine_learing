from sklearn.datasets import load_iris#获取数据
from sklearn.preprocessing import StandardScaler#标准化
from sklearn.model_selection import train_test_split,GridSearchCV#数据集分割
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

#第四步 机器学习
#1.估计器
estimate=KNeighborsClassifier()
#2.模型调优 交叉验证，网格搜索
#交叉验证 是提高稳定性  网格搜索是提高准确率
n_beibor={'n_neighbors':[1,3,5,7]} #k值
model_train=GridSearchCV(estimate,param_grid=n_beibor,cv=5)#cv=5是5个交叉验证
#模型训练
model_train.fit(x_train,y_train)

#第五步 模型评估
result=model_train.predict(x_test)

print('预测值为',result)
print('预测值与真实值对比',result==y_test)

print("准确率为：",model_train.score(x_test,y_test))

print("在交叉验证中最好的准确率为",model_train.best_score_)

print("在交叉验证中最好的模型是：",model_train.best_estimator_)

print("在交叉验证中的模型数据是：",model_train.cv_results_)