from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import jieba


#计算出词在文本中的占比TfidfVectorizer
def text3():
    data=['花有花有 重开日','人无再少年少年少年']
    # 用jieba进行分词 用空格进行分割 因为jieba.cut返回的是对象 ，所以需要用list强制转换   jieba.lcut是返回列表
    newsdata = []
    for temp in data:
        ret =" ".join(list(jieba.cut(temp)))

        newsdata.append(ret)
    print(newsdata)

    Coune = TfidfVectorizer()
    newdata = Coune.fit_transform(newsdata)
    print('中文文本提取特征名：', Coune.get_feature_names())
    print('用数组方式显示数据：', newdata.toarray())


if __name__=="__main__":
    text3()