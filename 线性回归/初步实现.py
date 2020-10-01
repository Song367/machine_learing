from sklearn.linear_model import LinearRegression#正规方程   适用于小规模数据 （不能解决拟合问题）
from sklearn.linear_model import SGDRegressor #梯度下降  使用于大规模数据

x=[[80,86],[82,80],[85,78],[90,90],[86,82],[82,90],[78,80],[92,94]]
y=[84.2,80.6,80.1,90,83.2,87.6,79.4,93.4]

liner=LinearRegression()
liner.fit(x,y)

#输出系数
print('系数为：',liner.coef_)

print('预测值为',liner.predict([[100,80]]))