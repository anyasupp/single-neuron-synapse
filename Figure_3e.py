# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 11:32:25 2023
Figure 3e
controls fish-categorized into early sleepers or late sleepers (via amount of sleep within fish)

'punc_hr4h' and 'punct_ZT18_0' -> puncta per hour after 4hr of SD (ZT14-18) and at later stage of night (ZT18-24)
'sleepph_adjusted_exp' and 'late_sleepph_adjusted_exp' = amount of sleep (min) per hour adjusted with the time we tracked
 
@author: AnyaS
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

data = pd.read_excel("Figure3.xlsx",sheet_name='3e')

#%%
early_sleepers = data[data.sleep_type!='late_sleeper']
early_sleepers=early_sleepers[['punc_hr4h','punc_ZT18_0']].copy()
# early_sleepers_df=pd.melt(early_sleepers.reset_index(),id_vars='Fish_ID')
late_sleepers = data[data.sleep_type!='early_sleeper']
late_sleepers=late_sleepers[['punc_hr4h','punc_ZT18_0']].copy()
# late_sleepers_df=pd.melt(late_sleepers.reset_index(),id_vars='Fish_ID')

#%% make dotted plot with early night and late night sleepers
# prep data 
early_zt14 = early_sleepers.punc_hr4h
early_zt18 = early_sleepers.punc_ZT18_0

late_zt14 = late_sleepers.punc_hr4h
late_zt18 = late_sleepers.punc_ZT18_0
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

def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=12)
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
    ax.plot(df_x_jitter.loc[idx,['punc_hr4h','punc_ZT18_0']], early_sleepers.loc[idx,['punc_hr4h','punc_ZT18_0']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
plt.plot(x,y,'k--',alpha=0.40)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None', capsize = 8, elinewidth=0.8,markeredgewidth=1.5,color='k')
ax.set_ylabel('Puncta Δ/ hour')    
plt.ylim(-9,7) 
simpleaxis(ax)
sns.despine()
#
ax = plt.subplot(row,column,2)
df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=late_sleepers.values.shape), columns=late_sleepers.columns)
df_x_jitter += np.arange(len(late_sleepers.columns))
for col in late_sleepers:
    ax.plot(df_x_jitter[col], late_sleepers[col], 'o',color='#65e0ba', alpha=.80, zorder=1, ms=8, mew=1)
ax.set_xticks(range(len(late_sleepers.columns)))
ax.set_xticklabels(late_sleepers.columns)
ax.set_xlim(-0.5,len(late_sleepers.columns)-0.5)

for idx in late_sleepers.index:
    ax.plot(df_x_jitter.loc[idx,['punc_hr4h','punc_ZT18_0']], late_sleepers.loc[idx,['punc_hr4h','punc_ZT18_0']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

plt.plot(x,y,'k--',alpha=0.40)
plt.plot(x3,y3,'k-',alpha=1)
plt.plot(x4,y4,'k-',alpha=1)
plt.errorbar(x_sem_late, y_sem_late, yerr=sem_late, linestyle='None', capsize = 8, elinewidth=0.8,markeredgewidth=1.5,color='k')
  
plt.ylim(-9,7) 
simpleaxis(ax)
sns.despine()