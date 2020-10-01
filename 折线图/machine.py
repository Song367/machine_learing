import matplotlib.pyplot as plt
from pylab import mpl
import random
#显示中文
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False
#创建画布
#plt.figure(figsize=(20,8))
plt.title("实时温度")
x=range(60)
y_shanghai=[random.uniform(15,18) for i in x]#列表生成式,生成15到18的随机数
y_beijing=[random.uniform(3,4) for i in x]
plt.plot(x,y_beijing,color='r',linestyle=':',label='北京')
plt.plot(x,y_shanghai,color='orange',linestyle='-.',label='上海')
#添加x,y的刻度
x_xticks_label=["11点{}分".format(i) for i in x]
y_yticks=range(40)
#print(x[::5])
plt.xticks(x[::5],x_xticks_label[::5])   # 5是步长 不能直接用字符串做x轴刻度，要先用一个x[::5]做一个刻度，然后再用字符串替换他。
plt.xlabel("时间")
plt.yticks(y_yticks[::5])
plt.ylabel("温度")
#添加网格
plt.grid(True,linestyle='--',alpha=0.5)
#保存图片
plt.legend(loc='lower right')
#plt.savefig("test.jpg")
plt.show()