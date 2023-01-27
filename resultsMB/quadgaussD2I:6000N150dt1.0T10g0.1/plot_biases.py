import matplotlib.pyplot as plt
from matplotlib  import cm

import pandas as pd

df = pd.read_csv("convergedCV.csv")

x = df.cvs
pot = df.potential
bias = df.bias

plt.scatter(x,pot)
plt.scatter(x,bias)

plt.show()

plt.scatter(x,pot+bias)
plt.show()
