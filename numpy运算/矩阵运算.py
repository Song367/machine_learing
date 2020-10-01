import numpy as np
a=np.array([[70,80,90,40],[60,77,88,99]])
b=np.array([[60,80],[66,60],[88,70], [44,90]])
#矩阵乘法
#[[70,80,90,40],[60,77,88,99]]乘以[[60,80],[66,60],[88,70], [44,90]]
#矩阵位置[0][0]=70*60+80*66+90*88+40*44,        [0][1]=70*80+80*60+90*70+40*90     剩下两个位置以此内推
print(np.dot(a,b))#实现矩阵相乘
print(np.matmul(a,b))#实现矩阵相乘，但是不支持矩阵和标量的乘法
#例如
#print(np.matmul(a,10))#这就会报错
print(np.dot(a,10))