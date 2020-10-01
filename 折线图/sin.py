import numpy as np
import matplotlib.pyplot as plt
x=np.linspace(-10,10,100)
y=np.sin(x)
plt.figure(figsize=(20,8))

plt.plot(x,y)
plt.grid(linestyle='--')
plt.show()