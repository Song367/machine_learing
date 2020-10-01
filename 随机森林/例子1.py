from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
# RandomForestClassifier  随机森林分类算法   RandomForestRegressor 随机森林回归算法
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from scipy.special import comb

# print(comb(25,3))

wine = load_wine()
# 随机森林是非常具有代表性的bagging集成算法

xtrain, xtest, ytrain, ytest = train_test_split(wine.data,wine.target,test_size=0.3)

clf = RandomForestClassifier(random_state= 0,n_estimators=10)   # n_estimators 是评估器数量
clf.fit(xtrain,ytrain)
print(clf.estimators_)   # estimators_ 是显示所有树的数据
print(clf.score(xtest,ytest))
