import sklearn
from sklearn.neighbors import KNeighborsClassifier #实现knn算法

#构造数据
#1.特征值 二维数据
#2.目标值 一维数据
x=[[1,2,3],[2,4,5],[10,11,12],[20,24,23]]#通过自己predict的数据和特征值进行距离计算
y=[0,0,1,1] #目标值，相当于是类别

#训练模型
#1首先实例化一个knn对象
knn=KNeighborsClassifier(n_neighbors=1)#对测试数据附近的一个数据进行判断属于哪个类别
#2训练
knn.fit(x,y)

#3测试数据
s=knn.predict([[1,20,4]])#测试数据必须是二维数据
print(s)

#距离计算
#1.欧式距离 sqrt((x1-x2)**2+(y1-y2)**2 )


#kd树 首先创建根结点 通过查看哪个维度更分散 来作为跟节点
# 例如 ((2,3),(5,4),(9,6),(4,7),(8,1),(7,2))
#第一维度 2 5 9 4 8 7
#第二维度 3 4 6 7 1 2

#因为第一维度更分散 所以找第一维度中的中位数作为根结点

#然后再对y轴进行子节点判断 也是通过中位数
#可以查看 “树的建立.jpg”

#最近领域搜索

