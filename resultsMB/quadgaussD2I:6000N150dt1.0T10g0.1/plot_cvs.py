import matplotlib.pyplot as plt
from matplotlib  import cm

import matplotlib

import pandas as pd

df = pd.read_csv("convergedCV.csv")

x = df.x1
y = df.x2
z = df.cvs

matplotlib.rc('figure', figsize=(6, 5))

plt.scatter(x,y,s=20,c=z, marker = 'o', cmap = cm.magma)
plt.xlabel("x", fontsize=16)
plt.ylabel("y", fontsize=16)
clb = plt.colorbar()
#clb.ax.set_ylabel("CV", rotation=0, size=13)
clb.ax.set_ylabel('CV',fontsize=13)
#plt.show()
plt.tight_layout()
plt.savefig("cvs.png", format="png", dpi=300)
