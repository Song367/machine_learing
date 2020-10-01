import numpy as np
a=np.array([[1,2,3,4],[5,6,7,8]])
#转置
print(a.reshape([4,2]))#转换成4行2列


#resize也可以实现和reshape一样的效果
print('--'*20)
print(a.T)#转置

#数组类型转换
print(a.tostring())
print(a.astype(np.int64))

#数组去重
b=np.array([[1,1,2],[22,3,3]])
c=np.array([1,1,1,1,2,2,2,3,3,4,5])
print(np.unique(b,return_index=True))
