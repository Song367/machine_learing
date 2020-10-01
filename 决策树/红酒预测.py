import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.datasets import load_wine, load_boston
from sklearn.model_selection import train_test_split , cross_val_score    # cross_val_score 交叉验证用于防止数据集划分问题
import graphviz       # 画决策树
from matplotlib.pylab import plt

# 分类树
def ClassifierTree():
    datas = load_wine()
    #print(datas.feature_names)
    data = datas['data']
    target = datas['target']

    x = pd.concat([pd.DataFrame(data),pd.DataFrame(target)],axis=1)   # 将数据进行合并

    xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size=0.30, random_state=9)    # random_state是用于控制随机数 ，如果不控制，则会每次精准度不一样

    clf = tree.DecisionTreeClassifier(criterion='entropy'    # criterion='entropy'是在欠拟合的时候才用
                                      ,splitter='random'
                                      ,random_state=6
                                 #     , max_depth=9       # 树的深度,用于反之过拟合
                                 #     , min_samples_leaf=10       # 防止过拟合
                                 #     , min_samples_split=10     # 防止过拟合
                                 #     ,max_features=8         #最大特征数  也是防止过拟合
                                      )    # random_state是用于控制随机数 ，如果不控制，则会每次精准度不一样  所有参数都是用于调优的，入股哦打不到最好，就删掉
    clf.fit(xtrain,ytrain)

    testIndex = clf.apply(xtest)     # 每个测试样本的叶子节点索引
    testData = clf.predict(xtest)    # 预测值

    print(clf.score(xtest,ytest))
    print(clf.score(xtrain,ytrain))

    # 结果显示 如果训练集大于测试集  存在过拟合

    feature_names = datas.feature_names
    dot_data = tree.export_graphviz(
        clf,
        feature_names=feature_names,
        class_names=['琴酒', '雪梨', '贝尔摩德'],
        filled=True,
        rounded=True,
    )

    graph = graphviz.Source(dot_data)
    graph.save('wins.dot')
    # print(graph)


def Regressor():
    data_boston = load_boston()
    model = tree.DecisionTreeRegressor()
    res = cross_val_score(model, data_boston.data, data_boston.target, cv=10)   # cv=10是执行10次
    print(res)


# 找到回归
def AddRegressor():
    rng = np.random.RandomState(1)   # 和random_state的用法一致   在这可以固定生成的随机数
    x = np.sort(5*rng.rand(80, 1), axis=0)   # rng.rand(80,1)  生成80行1列的随机数 乘以5就是生成0-5的随机数
    y = np.sin(x).ravel()    # ravel()降维
    y[::5] += 0.3*(0.5 - rng.rand(16))

    # plt.show()
    reg1 = tree.DecisionTreeRegressor(max_depth=2)
    reg2 = tree.DecisionTreeRegressor(max_depth=5)
    reg1.fit(x,y)
    reg2.fit(x,y)

    test = np.arange(0.0,5.0,0.01)[:,np.newaxis]
    y1 = reg1.predict(test)
    y2 = reg2.predict(test)
    #plt.figure()
    plt.figure()
    plt.scatter(x, y,label='dian')
    plt.plot(test,y1,color='red',label='max_depth=2')
    plt.plot(test,y2,color='yellow',label="max_depth=5")
    plt.xlabel('data')
    plt.ylabel('target')
    plt.legend(loc='upper right')
    plt.show()


AddRegressor()