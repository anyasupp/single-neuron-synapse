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
    ax.tick_params(axis='y',labelsize=10)
    
colors_pal = ['#1f77b4',
              '#d73027']
row = 1
column = 2
fig, ax = plt.subplots(row,column, sharex=True)
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
    markers="o", scale=.75, ci=68
)
plt.plot(bly1,blx1,'k--',alpha=0.25,zorder=1)

plt.xlim(-0.5,1.5)
simpleaxis(ax)
ax = plt.subplot(row,column,2)

sns.stripplot(
    data=mid2, y="p_diff", x="Condition",
    dodge=False, alpha=0.5, zorder=1,palette=colors_pal
)

sns.pointplot(
    data=mid2, y="p_diff", x="Condition",
    join=False, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
plt.plot(bly1,blx1,'k--',alpha=0.25,zorder=1)
plt.ylabel('')
plt.xlim(-0.5,1.5)
simpleaxis(ax)
sns.despine(bottom=False, left=False)