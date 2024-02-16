# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 09:37:08 2023
plotting sleep latency graph
output data from frame by frame
@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_excel("Ext Figure9.xlsx", sheet_name='c')
#%%
df = pd.melt(data[data.grp =='grp1'],id_vars='fish',value_vars=['woi2','woi1'])

midday = data[data.grp !='excluded'].woi2
lightoff = data[data.grp !='excluded'].woi1

from scipy.stats import sem
sem_midday = sem(midday, axis=None, ddof=0,nan_policy = 'omit')
sem_lightoff = sem(lightoff, axis=None, ddof=0,nan_policy = 'omit')
#%%
x,y = [-0.2,0.2],[midday.mean(),midday.mean()]
x2,y2 = [0.8,1.2],[lightoff.mean(),lightoff.mean()]

sem = [sem_midday,sem_lightoff]
x_sem = [0,1]
y_sem =[midday.mean(),lightoff.mean()]

def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=8)
    
row = 1
column = 1
fig, ax = plt.subplots(row,column, sharex=True)
ax = plt.subplot(row,column,1)
sns.stripplot(
    data=df, y="value", x="variable",
    dodge=True, alpha=0.5, zorder=1,color='gray')

plt.plot(x,y,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None', capsize = 5, elinewidth=0.8,markeredgewidth=1.5,color='k')
sns.despine()
simpleaxis(ax)
#%%
from scipy import stats
woi2= np.array(data[data.grp !='excluded'].woi2.dropna())
woi1 = np.array(data[data.grp !='excluded'].woi1.dropna())
stats.kruskal(woi2,woi1)
