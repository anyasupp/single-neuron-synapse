# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 12:08:49 2021

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
dev = pd.read_excel("ExtFigure2.xlsx",sheet_name='cd')

#%% Figure set up
# sns.set_context("paper")
row = 2
column = 2
fig, ax = plt.subplots(row,column, sharex=True)
fig.suptitle('FingR.PSD95 Development ', fontsize=16)
fig.tight_layout()

ax = plt.subplot(row,column,1)
ax = sns.lineplot(x="Time", y="Puncta", data=dev,ci=68)
# plt.xticks(rotation=45)
sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('Puncta Count')    
ax.set_xlabel('')
#
ax = plt.subplot(row,column,2)
ax = sns.lineplot(x="Time", y="Puncta", data=dev,hue="Fish_ID",palette="flare")
ax.get_legend().remove()

# plt.xticks(rotation=45)
sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')


#baseline at zero
x1,y1 = [-0.5,7],[0,0]


ax = plt.subplot(row,column,3)
plt.plot(x1,y1,'k--',alpha=0.35)
ax = sns.lineplot(x="Time", y="RoC", data=dev,ci=68)
# plt.xticks(rotation=45)
plt.xlim(0.65,6.5) 


sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('RoC (%)')    
ax.set_xlabel('')
#
ax = plt.subplot(row,column,4)
plt.plot(x1,y1,'k--',alpha=0.35)
ax = sns.lineplot(x="Time", y="RoC", data=dev,hue="Fish_ID",palette="flare")

plt.xlim(0.65,6.5) 


# plt.xticks(rotation=45)
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()
sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')   


