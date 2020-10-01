from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pylab import mpl
mpl.rcParams["font.sans-serif"] = ["SimHei"]
mpl.rcParams["axes.unicode_minus"] = False

plt.title('城市温度')

x_data = range(60)

shanghai = [np.random.randint(15, 23) for _ in x_data]
#print(shanghai)
beijing = [np.random.randint(10, 22) for _ in x_data]

plt.plot(x_data, shanghai, color='r', label='上海')
plt.plot(x_data, beijing, color='b', label='北京')

x_ticks = ["11:{}".format(i) if i > 5 else "11:0{}".format(i) for i in x_data]
y_ticks = range(40)

plt.xticks(x_data[::5], x_ticks[::5])
plt.yticks(y_ticks[::5])
plt.xlabel('time')
plt.ylabel('weather')

plt.grid(True, linestyle='--', alpha=0.5)

plt.legend(loc='upper left')
plt.show()

