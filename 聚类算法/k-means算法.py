import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets._samples_generator import make_blobs     # 生成数据集


# x,y=make_blobs(n_samples=1000,n_features=2,centers=[[1,1],[2,2],[3,3]],cluster_std=[0.2,0.3,0.4],random_state=9)
# print(x,'\n',y)
# # n_samples是生成多少个数据  ， n_feature是几个特征值 比如该例子中就是x，y轴
# # centers 是生成数据的中心点或者叫质点 ， cluster_std 标准差  random_state 这个是为了保证每次生成的点都一样
# plt.scatter(x[:,0],x[:,1],marker='o')   # marker='o' 是用圆形显示点
# plt.show()


# 最后输出结果明显看出不同标准差之间的差异了


digits_train = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tra',
                           header=None)
# print(digits_train)
digits_test = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/optdigits/optdigits.tes',
                          header=None)

X_train = digits_train[np.arange(64)]
y_train = digits_train[64]

X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

kmeans = KMeans(n_clusters=2)
kmeans.fit(X_train,y_train)

y_pred = kmeans.predict(X_test)

plt.plot(X_test,y_pred)
plt.show()
from sklearn import metrics

# print(metrics.adjusted_rand_score(y_test, y_pred))

