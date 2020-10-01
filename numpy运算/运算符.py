import numpy as np

a=np.random.randint(40,100,[4,5])
print(a[0:]>60)

#np.all()是返回所有数据是否都是大于60，布尔值
b=np.all(a[0:2,:]>60)
print(b)
#np.any()是返回数据中只要有一个大于60，布尔值
c=np.any(a[0:2,:]>60)
print(c)

#三元运算符
d=np.where(a>60,1,0)#如果数据大于60就置1，否则0
print(d)
print('--'*20)

f=np.array([[60,50,90,100,80],[70,69,80,88,89]])
d=np.where(np.logical_and(f>=60,f<=90),1,0)#返回逻辑与运算，必须满足两个条件
e=np.where(np.logical_or(60<=f,f>=90),1,0)#返回逻辑或运算，只需满足其中一个条件
print(d,'\n')
print(e)
#标准差，方差，中位数，平均数
#axis=1,按照行来计算
print('平均数',np.mean(f,axis=1),'\n')#平均数
print('标准差',np.std(f,axis=1),'\n')#标注差
print('中位数',np.median(f,axis=1),'\n')#中位数
print('方差',np.var(f,axis=1),'\n')#方差
print('最小值',np.min(f,axis=1),'\n')#最小值
print('最大值下标',np.argmax(f,axis=1),'\n')#最大值的下标


#数组加法
#条件一：数组某一维度等长，或者条件二:数组某一维度为一
#例如：[2,6],[2,1]就可以相加，[2,6],[2,3]就不可以相加
aa = np.random.randint(1,10,[2,6])
bb=np.random.randint(1,10,[2,1])
print(aa,'\n',bb)
print(aa+bb)