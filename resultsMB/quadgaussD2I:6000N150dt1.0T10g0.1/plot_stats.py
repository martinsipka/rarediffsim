import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from matplotlib  import cm

import pandas as pd

cs = ['#00429d', '#A50062', '#ff9200', '#6609BD'] 

df = pd.read_csv("training.csv", index_col=0)[1:]

fig, ax1 = plt.subplots(figsize=(6, 5))
ax2 = ax1.twinx()

#ax1.plot(df.index, df.loss_fwd, label="Forward Loss", c=cs[0])
#ax1.plot(df.index, df.loss_s, label="Start Loss", c=cs[1])
l1 = ax1.plot(df.index, (df.loss_fwd + df.loss_s)/600, label="Sum of losses", c=cs[2])
ax1.set_xlabel("Iteration", fontsize=16)
ax1.set_ylabel("Average Loss",  color=cs[2], fontsize=16)
ax1.tick_params(axis="y", labelcolor=cs[2])
#ax1.set_ylim([0,100000])

l2 = ax2.plot(df.index, df.success, c=cs[3])
ax2.set_ylabel("Success rate [%]", color=cs[3], fontsize=16)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
ax2.tick_params(axis="y", labelcolor=cs[3])
#plt.show()

plt.tight_layout()

#fig.legend([l1, l2], labels=labels)
plt.savefig("sucrate.eps", format="eps", dpi=1200)


