# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:06:22 2023
Figure 1f 
average activity in different lighting conditions

@author: Anya
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

ld = pd.read_excel("Figure1.xlsx",sheet_name='1f_LD')
ll = pd.read_excel("Figure1.xlsx",sheet_name='1f_LL')
fr = pd.read_excel("Figure1.xlsx",sheet_name='1f_FR')

#%% re arrange data 
df_ld=pd.melt(ld,id_vars='Time')
df_ll =pd.melt(ll,id_vars='Time')
df_fr =pd.melt(fr,id_vars='Time')

#put condition 
cond_ll = np.array(['LL']*len(df_ll.index))
df_ll.insert(2,'Condition',cond_ll,True)
# #put conditions in 
cond_ld = np.array(['LD']*len(df_ld.index))
df_ld.insert(2,'Condition',cond_ld,True)

cond_fr = np.array(['FR']*len(df_fr.index))
df_fr.insert(2,'Condition',cond_fr,True)
#combined into one df
df = df_ld.append(df_ll)
df = df.append(df_fr)
#%%
def simpleaxis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='y',labelsize=12, width=1,length=10)
    ax.tick_params([])
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.set_yticks(np.arange(0, 100, 25))
    ax.set_xlabel('')
#%%
row = 1
column =1
fig, ax = plt.subplots(row,column, sharex=True)
# fig.suptitle('tenminute ', fontsize=16)
ax = plt.subplot(row,column,1)
ax= sns.lineplot(data= df_ld, x='Time', y='value')
ax= sns.lineplot(data= df_ll, x='Time', y='value',color='#dd90b5')
ax= sns.lineplot(data= df_fr, x='Time', y='value',color='#00ac87')


ax.axvspan(38, 48, alpha=0.15, color='gray',zorder=1)
ax.axvspan(62, 72, alpha=0.15, color='gray',zorder=1)
ax.axvspan(86, 96, alpha=0.15, color='gray',zorder=1)
plt.xlim(28,101)
plt.yticks(np.arange(0, 100, 25))
ax.tick_params(axis='y',labelsize=13)
plt.xticks([])
# simpleaxis(ax)
sns.despine(bottom=True, trim=True,offset=0.5)
