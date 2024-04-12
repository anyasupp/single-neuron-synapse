# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:26:35 2023

@author: Anya
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

mid2 = pd.read_excel("Figure3.xlsx",sheet_name='3h_mid2')
vs = pd.read_excel("Figure3.xlsx",sheet_name='3h_vs')

def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=15)
    
colors_pal = ['#1f77b4',
              '#d73027']
row = 1
column = 2
fig, ax = plt.subplots(row,column)
ax = plt.subplot(row,column,1)

blx1,bly1 = [0,0],[-0.5,15]

sns.despine(bottom=False, left=False)
ax = plt.subplot(row,column,1)

sns.stripplot(
    data=vs, y="p_diff", x="Condition",
    dodge=False, alpha=0.5, zorder=1,palette=colors_pal
)

sns.pointplot(
    data=vs, y="p_diff", x="Condition",
    join=False, palette=colors_pal,
    markers="o", scale=.75, errorbar=('ci', 68)
)
plt.plot(bly1,blx1,'k--',alpha=0.25,zorder=1)

plt.xlim(-0.5,1.5)
simpleaxis(ax)
plt.xticks(fontname='Arial')
plt.yticks(fontname='Arial')

ax = plt.subplot(row,column,2)

sns.stripplot(
    data=mid2, y="p_diff", x="Condition",
    dodge=False, alpha=0.5, zorder=1,palette=colors_pal
)

sns.pointplot(
    data=mid2, y="p_diff", x="Condition",
    join=False, palette=colors_pal,
    markers="o", scale=.75, errorbar=('ci', 68)
)
plt.plot(bly1,blx1,'k--',alpha=0.25,zorder=1)
plt.ylabel('')
plt.xlim(-0.5,1.5)
simpleaxis(ax)
plt.xticks(fontname='Arial')
plt.yticks(fontname='Arial')
sns.despine(bottom=False, left=False)
#%%
fig.savefig('fig3h_arial.svg', format='svg',dpi=600, transparent=True)
#%%
from scipy.stats import f_oneway

con= mid2[mid2['Condition']=='Control']['p_diff']
sd= mid2[mid2['Condition']=='SD']['p_diff']

# Perform one-way ANOVA
f_statistic, p_value = f_oneway(con, sd)

# Print the results
print("F Statistic:", f_statistic)
print("P-value:", p_value)
