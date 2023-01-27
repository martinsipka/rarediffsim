import matplotlib.pyplot as plt
from matplotlib  import cm
import matplotlib

import numpy as np

import pandas as pd

matplotlib.rc('figure', figsize=(6, 5))

cs = ['#00429d', '#A50062', '#ff9200'] 

df = pd.read_csv("convergedCV.csv")

df['bin'] = pd.cut(df['cvs'], np.arange(-3,3,0.1))
agg_df = df.groupby(by='bin').mean()
mids = pd.IntervalIndex(agg_df.index.get_level_values('bin')).mid

agg_df.bias = agg_df.bias - agg_df.bias.min()
agg_df.potential = (agg_df.potential - agg_df.potential.min())

#plt.plot(mids, agg_df.bias, c=cs[0])
plt.plot(mids, agg_df.potential, c=cs[1], label="Original potential")
plt.plot(mids, agg_df.bias+agg_df.potential, c=cs[0], label="Potential + Bias")

plt.xlabel("CV", fontsize=16)
plt.ylabel("Potential [kcal/mol]", fontsize=16)
plt.legend(fontsize=16)
plt.tight_layout()
plt.savefig("biases.eps", format="eps", dpi=1200)
plt.close()

#plt.plot(mids, (agg_df.bias+agg_df.potential) - (agg_df.bias+agg_df.potential).min(), c=cs[2])
#plt.show()
