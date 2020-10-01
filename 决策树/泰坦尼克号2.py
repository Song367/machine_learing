from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV,train_test_split,cross_val_score
# GridSearchCV 是网格搜索，用来条多个参数的 cross_val_score 交叉验证
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import matplotlib.pyplot as plt


pddata = pd.read_csv("http://biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic.txt")
# 数据处理
pddata.drop(['name','ticket','room','pclass','home.dest','boat','row.names'],inplace=True,axis=1)

pddata.loc[:,'age'] = pddata['age'].fillna(pddata['age'].mean())    # 填充缺失值

data = pddata.dropna()   # 删除所有含缺失值的行

labels = data['embarked'].unique().tolist()  # 删除重复的值，并转换为列表
# loc 取字符索引 ， iloc 取数字索引
data.loc[:,'embarked'] = data['embarked'].apply(lambda x: labels.index(x))     # 因为机器模型识别不了字符串，所以要转换成数字

data.loc[:,'sex'] = (data['sex']=='male').astype('int')       # 将male 转换为1 female为0

# 特征值
x = data.iloc[:,data.columns!='survived']      # 取出所有不等于survive的列
# 目标值
y = data.iloc[:,data.columns=='survived']

xtrain ,xtest ,ytrain ,ytest=train_test_split(x,y,test_size=0.3)

# 将每一个数据集都索引从小到大排
for i in [xtrain,xtest,ytrain,ytest]:
    i.index = range(i.shape[0])

model = DecisionTreeClassifier(random_state=7)
# model.fit(xtrain,ytrain)
# print(model.score(xtest,ytest))
paramers = {'criterion': ('gini','entropy')
            , 'max_depth': [*range(1,10)]
            , 'splitter': ('best', 'random')}

clf = GridSearchCV(model,paramers,cv=10)
clf.fit(xtrain,ytrain)

print(clf.best_params_)          # 返回最佳属性组合
print(clf.best_score_)          # 返回最佳精确度
# 交叉验证

#score = cross_val_score(model,x,y,cv=10).mean()
#print(score)
