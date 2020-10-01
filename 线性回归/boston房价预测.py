#使用均方误差做评估
from sklearn.metrics import mean_squared_error  #均方误差  参数y_true 真是值  y_false是预测值
from sklearn.datasets import load_boston  #数据获取
from sklearn.model_selection import train_test_split  #数据集分割
from sklearn.linear_model import LinearRegression,SGDRegressor,Ridge,RidgeCV  #正规方程和梯度下降
from sklearn.preprocessing import StandardScaler  #标准化
from sklearn.externals import joblib   #保存模型


#线性回归：正规方程
def linear_model():
    #1.
    data=load_boston()
    #2.
    x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=22)
    #3.
    Stan=StandardScaler()
    x_train=Stan.fit_transform(x_train)
    x_test=Stan.fit_transform(x_test)

    #4
    linear=LinearRegression()
    linear.fit(x_train,y_train)

    #5
    ss=linear.predict(x_test)
    #print('正规方程预测值是：',ss)
    print('正规方程均方误差:',mean_squared_error(y_test,ss))
    print('正规方程系数：',linear.coef_)

#线性回归：梯度下降
def linear_model2():
    #1.
    data=load_boston()
    #2.
    x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=22)
    #3.
    Stan=StandardScaler()
    x_train=Stan.fit_transform(x_train)
    x_test=Stan.fit_transform(x_test)

    #4
    linear=SGDRegressor(max_iter=1000)   #最大迭代次数
    linear.fit(x_train,y_train)

    #5
    ss=linear.predict(x_test)
    #print('梯度下降预测值是：',ss)
    print('梯度下降均方误差:',mean_squared_error(y_test,ss))
    print('梯度下降系数：',linear.coef_)

#岭回归
def linear_model3():
    #1.
    data=load_boston()
    #2.
    x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=22)
    #3.
    Stan=StandardScaler()
    x_train=Stan.fit_transform(x_train)
    x_test=Stan.fit_transform(x_test)

    #4
    linear=Ridge(alpha=1.0)
    linear.fit(x_train,y_train)

    #5
    ss=linear.predict(x_test)
    #print('正规方程预测值是：',ss)
    print('岭回归均方误差:',mean_squared_error(y_test,ss))
    print('岭回归系数：',linear.coef_)

#岭回归的交叉验证
def linear_model4():
    #1.获取数据
    data=load_boston()
    #2.数据处理
    x_train,x_test,y_train,y_test=train_test_split(data.data,data.target,test_size=0.2,random_state=22)
    #3.特征工程
    #3.1 标准化
    Stan=StandardScaler()
    x_train=Stan.fit_transform(x_train)
    x_test=Stan.fit_transform(x_test)

    #4 机器学习
    # 4.1 模型训练
    linear=RidgeCV(alphas=[1.0,0.001,2,0.1,0.5,1.5])  #从这个这里面选择最好的
    linear.fit(x_train,y_train)
    # 4.2 模型保存
    joblib.dump(linear,'linearmodel.pkl')

    #4.3模型导入
    #linear=joblib.load('linearmodel.pkl')


    #5 模型评估
    ss=linear.predict(x_test)
    #print('岭回归的交叉验证预测值是：',ss)
    print('岭回归的交叉验证均方误差:',mean_squared_error(y_test,ss))
    print('岭回归的交叉验证系数：',linear.coef_)

# linear_model()
# linear_model2()
# linear_model3()
linear_model4()

