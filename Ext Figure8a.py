# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 17:23:53 2022

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#%%#%% for averaage total normalized
normalize_total = pd.read_excel("Ext Figure8.xlsx",sheet_name='8a_total_sleep')

norm_total_con = normalize_total[normalize_total.Condition !='SD'].norm_av_1
norm_total_sd = normalize_total[normalize_total.Condition !='Control'].norm_av_1

#
from scipy.stats import sem
sem_tot_con = sem(normalize_total[normalize_total.Condition !='SD'].norm_av_1, axis=None, ddof=0,nan_policy = 'omit')
sem_tot_sd = sem(normalize_total[normalize_total.Condition !='Control'].norm_av_1, axis=None, ddof=0,nan_policy = 'omit')
#

x1,y1 = [-0.35,0.35],[norm_total_con.mean(),norm_total_con.mean()]
x2,y2 = [0.65,1.35],[norm_total_sd.mean(),norm_total_sd.mean()]

sem = [sem_tot_con,sem_tot_sd]
x_sem = [0,1]
y_sem =[norm_total_con.mean(),norm_total_sd.mean()]


#%%
def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=14)
    
row = 1
column = 2
fig, ax = plt.subplots(row,column, sharex=True)
# fig.suptitle('xxxxxx', fontsize=16)
ax = plt.subplot(row,column,1)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None',
             capsize = 10, elinewidth=1,markeredgewidth=2,color='k',zorder=2)
# ,alpha=0.8,zorder=1,palette=colors
ax = sns.swarmplot(x="Condition", y="norm_av_1", 
                   data=normalize_total,alpha=0.8,zorder=1, palette='tab10')
ax.set_ylabel('av normalized Δ% total sleep')    
sns.despine()
simpleaxis(ax)
# for sleep bout
normalize_bout =  pd.read_excel("Ext Figure8.xlsx",sheet_name='8a_average_sleep')

norm_bout_con = normalize_bout[normalize_bout.Condition !='SD'].norm_av_1
norm_bout_sd = normalize_bout[normalize_bout.Condition !='Control'].norm_av_1

#
from scipy.stats import sem
sem_bout_con = sem(normalize_bout[normalize_bout.Condition !='SD'].norm_av_1, axis=None, ddof=0,nan_policy = 'omit')
sem_bout_sd = sem(normalize_bout[normalize_bout.Condition !='Control'].norm_av_1, axis=None, ddof=0,nan_policy = 'omit')
#

x1,y1 = [-0.35,0.35],[norm_bout_con.mean(),norm_bout_con.mean()]
x2,y2 = [0.65,1.35],[norm_bout_sd.mean(),norm_bout_sd.mean()]

sem = [sem_bout_con,sem_bout_sd]
x_sem = [0,1]
y_sem =[norm_bout_con.mean(),norm_bout_sd.mean()]

ax = plt.subplot(row,column,2)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None',
             capsize = 10, elinewidth=1,markeredgewidth=2,color='k',zorder=2)

ax = sns.swarmplot(x="Condition", y="norm_av_1", data=normalize_bout,
                   alpha=0.8,zorder=1,palette='tab10')
ax.set_ylabel('av normalized Δ% average sleep bout')
sns.despine()    
simpleaxis(ax)
#%