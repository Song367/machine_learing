from sklearn.feature_extraction import  DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba
from pylab import mpl
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False
#字典特征提取
def dict1():
    data=[{'name':"张三",'city':'北京','num':100},
          {'name':"李四",'city':'上海','num':80},
          {'name':'历代','city':'天堂','num':60}]

    dacta=DictVectorizer(sparse=True)
    newdata=dacta.fit_transform(data)

    #print('不是spare矩阵\n',newdata)  #里面返回的数据就是代表岁对应属性名所在哪一列
    print('属性名：\n',dacta.get_feature_names())
    print('sparse矩阵\n',newdata)
    # sparse=True时  数据所代表第几行，第几列，对应睡醒名


def text1():
    data=['How are you','I am fine']

    Coune=CountVectorizer()   #这里面还可以传参  stop_words=[" you "]  这样就是停止you这个特征值的提取
    newdata=Coune.fit_transform(data)

    print('英文文本提取特征名：',Coune.get_feature_names())
    print('用数组方式显示数据：',newdata.toarray())  #0代表在该行中没有  ，反之1代表有
    print('原始数据：',newdata)

#中文特征文本提取
def text2():
    data=['花有重开日','人无再少年']
    # 用jieba进行分词 用空格进行分割 因为jieba.cut返回的是对象 ，所以需要用list强制转换   jieba.lcut是返回列表
    newsdata = []
    for temp in data:
        ret =" ".join(list(jieba.cut(temp)))
        newsdata.append(ret)
    print(newsdata)

    Coune = CountVectorizer()
    newdata = Coune.fit_transform(newsdata)
    print('中文文本提取特征名：', Coune.get_feature_names())
    print('用数组方式显示数据：', newdata.toarray())




dict1()
text1()
text2()
