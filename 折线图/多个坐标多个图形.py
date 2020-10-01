import matplotlib.pyplot as plt
from pylab import mpl
import random
mpl.rcParams["font.sans-serif"]=["SimHei"]
mpl.rcParams["axes.unicode_minus"]=False
fig,asex=plt.subplots(nrows=1,ncols=2,figsize=(20,8))
x=range(60)
y_b=[random.uniform(15,18) for i in x]
y_s=[random.uniform(10,13) for i in x]
xlabel=["11点{}分".format(i) for i in x]
asex[0].plot(x,y_b,color='orange',linestyle='--',label='北京')
asex[1].plot(x,y_s,color='red',linestyle='-.',label='上海')
#刻度显示
y_ticks=range(40)
asex[0].set_xticks(x[::5])
asex[0].set_yticks(y_ticks[::5])
asex[0].set_xticklabels(xlabel[::5])
asex[1].set_xticks(x[::5])
asex[1].set_yticks(y_ticks[::5])
asex[1].set_xticklabels(xlabel[::5])

#添加网格
asex[0].grid(True,linestyle='-',alpha=0.5)
asex[1].grid(True,linestyle='-',alpha=0.5)
#标签
asex[0].set_title("实时温度")
asex[0].set_xlabel("时间")
asex[0].set_ylabel("温度")
asex[1].set_title("实时温度")
asex[1].set_xlabel("时间")
asex[1].set_ylabel("温度")

plt.savefig('one.png')
asex[0].legend(loc='best')
asex[1].legend(loc='best')
plt.show()