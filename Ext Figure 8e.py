# -*- coding: utf-8 -*-
"""
Created on Sat Nov 18 11:09:45 2023

@author: Anya
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
#%%
data = pd.read_excel("Figure3.xlsx",sheet_name='3e')
#%%
'''
take away the fish that has multiple neurons
'''
# Assigning NaN to a specific cell 
# fish id = F11LT_20220223_ON

data.loc[4, 'sleepph_adjusted_exp']= np.nan
data.loc[4, 'late_sleepph_adjusted_exp']= np.nan
# fish id = F11R_20220223_ON
data.loc[5, 'sleepph_adjusted_exp']= np.nan
data.loc[5, 'late_sleepph_adjusted_exp']= np.nan

#%%
#sleepph_adjusted_exp = early nigth sleep per hour
#late_sleepph_adjusted_exp = late night sleep per hour 
early_sleepers = data[data.sleep_type!='late_sleeper']
early_sleepers=early_sleepers[['sleepph_adjusted_exp','late_sleepph_adjusted_exp']].copy()
# early_sleepers_df=pd.melt(early_sleepers.reset_index(),id_vars='Fish_ID')
late_sleepers = data[data.sleep_type!='early_sleeper']
#late sleeper has nan in puncta
late_sleepers = late_sleepers.dropna()
late_sleepers=late_sleepers[['sleepph_adjusted_exp','late_sleepph_adjusted_exp']].copy()
# late_sleepers_df=pd.melt(late_sleepers.reset_index(),id_vars='Fish_ID')

#%% make dotted plot with early night and late night sleepers
# prep data 
early_zt14 = early_sleepers.sleepph_adjusted_exp
early_zt18 = early_sleepers.late_sleepph_adjusted_exp

late_zt14 = late_sleepers.sleepph_adjusted_exp
late_zt18 = late_sleepers.late_sleepph_adjusted_exp
#
#sem works on series so size (x,)
from scipy.stats import sem
sem_early_zt14 = sem(early_zt14, axis=None, ddof=0,nan_policy = 'omit')
sem_early_zt18 = sem(early_zt18, axis=None, ddof=0,nan_policy = 'omit')
#late_sleeper
sem_late_zt14 = sem(late_zt14, axis=None, ddof=0,nan_policy = 'omit')
sem_late_zt18 = sem(late_zt18, axis=None, ddof=0,nan_policy = 'omit')
#
late_sleepers  = late_sleepers.reset_index()
late_sleepers = late_sleepers.drop(columns='index')
#%%
def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=12)
jitter = 0.05
##baseline at zero
x,y = [-0.5,8],[0,0]
x1,y1 = [-0.2,0.2],[early_zt14.mean(),early_zt14.mean()]
x2,y2 = [0.8,1.2],[early_zt18.mean(),early_zt18.mean()]

x3,y3 = [-0.2,0.2],[late_zt14.mean(),late_zt14.mean()]
x4,y4 = [0.8,1.2],[late_zt18.mean(),late_zt18.mean()]

sem = [sem_early_zt14,sem_early_zt18]
x_sem = [0,1]
y_sem =[early_zt14.mean(),early_zt18.mean()]

sem_late = [sem_late_zt14,sem_late_zt18]
x_sem_late = [0,1]
y_sem_late =[late_zt14.mean(),late_zt18.mean()]

row = 1
column = 2
fig, ax = plt.subplots()
ax = plt.subplot(row,column,1)
df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=early_sleepers.values.shape), columns=early_sleepers.columns)
df_x_jitter += np.arange(len(early_sleepers.columns))
for col in early_sleepers:
    ax.plot(df_x_jitter[col], early_sleepers[col], 'o',color='#15607a', alpha=.80, zorder=1, ms=8, mew=1)
ax.set_xticks(range(len(early_sleepers.columns)))
ax.set_xticklabels(early_sleepers.columns)
ax.set_xlim(-0.5,len(early_sleepers.columns)-0.5)

for idx in early_sleepers.index:
    ax.plot(df_x_jitter.loc[idx,['sleepph_adjusted_exp','late_sleepph_adjusted_exp']],
            early_sleepers.loc[idx,['sleepph_adjusted_exp','late_sleepph_adjusted_exp']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
# plt.plot(x,y,'k--',alpha=0.40)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None', capsize = 8, elinewidth=0.8,markeredgewidth=1.5,color='k')
ax.set_ylabel('Sleep (min/ hour)')    
# plt.ylim(-9,7) 
sns.despine()
simpleaxis(ax)

ax = plt.subplot(row,column,2)
df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=late_sleepers.values.shape), columns=late_sleepers.columns)
df_x_jitter += np.arange(len(late_sleepers.columns))
for col in late_sleepers:
    ax.plot(df_x_jitter[col], late_sleepers[col], 'o',color='#65e0ba', alpha=.80, zorder=1, ms=8, mew=1)
ax.set_xticks(range(len(late_sleepers.columns)))
ax.set_xticklabels(late_sleepers.columns)
ax.set_xlim(-0.5,len(late_sleepers.columns)-0.5)

for idx in late_sleepers.index:
    ax.plot(df_x_jitter.loc[idx,['sleepph_adjusted_exp','late_sleepph_adjusted_exp']], late_sleepers.loc[idx,['sleepph_adjusted_exp','late_sleepph_adjusted_exp']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

# plt.plot(x,y,'k--',alpha=0.40)
plt.plot(x3,y3,'k-',alpha=1)
plt.plot(x4,y4,'k-',alpha=1)
plt.errorbar(x_sem_late, y_sem_late, yerr=sem_late, linestyle='None', capsize = 8, elinewidth=0.8,markeredgewidth=1.5,color='k')
  
simpleaxis(ax)
sns.despine()
#%%