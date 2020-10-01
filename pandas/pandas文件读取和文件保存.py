import pandas as pd
# read_xx 代表文件读取， to_xx代表文件存储

a=pd.read_csv("movie.csv",usecols=["director_name","aspect_ratio"],encoding='gbk') # 第一个参数是路径，第二个参数是选取其中的某行读取
print(a)
a.to_csv("movie1.csv",columns=["director_name"],index=False)   # index=false是删除原来的索引