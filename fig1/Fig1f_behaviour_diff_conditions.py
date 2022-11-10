# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 19:59:49 2022

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%data
ld = pd.read_excel("F1f_lights_behavior_tenmin_combined.xlsx",sheet_name='LD')
ll = pd.read_excel("F1f_lights_behavior_tenmin_combined.xlsx",sheet_name='LL')
fr = pd.read_excel("F1f_lights_behavior_tenmin_combined.xlsx",sheet_name='FR')

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
row = 1
column =1
fig, ax = plt.subplots(row,column, sharex=True)
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

sns.despine(bottom=True, trim=True,offset=0.5)