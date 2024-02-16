# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 16:46:55 2022

@author: AnyaS
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
data = pd.read_excel("Figure3.xlsx",sheet_name='3b')

#remove outlier (grubb's)
data_ = data[data.Fish_ID !='F5_20211027_ON']
##remove Type 1 
data_ = data[data.Cluster!=0]

#%%
data_combined = pd.melt(data_,id_vars=['Fish_ID','Condition'],value_vars=['dpf7_18','dpf8_0'],var_name='Time',value_name='punc_hr')

#seperate all into conditions
groups=data_.groupby('Condition')

#make new dataframe with cyan,green,red with correct fish ID
sd = [groups.get_group('SD')]
sd = pd.concat(sd)

control = [groups.get_group('Control')]
control = pd.concat(control)

punc_hr_sd_=pd.melt(sd,id_vars='Fish_ID',value_vars=['dpf7_18','dpf8_0'],var_name='Time',value_name='punc_hr')

punc_hr_=pd.melt(control,id_vars='Fish_ID',value_vars=['dpf7_18','dpf8_0'],var_name='Time',value_name='punc_hr')

early_con = punc_hr_[punc_hr_.Time !='dpf8_0'].punc_hr
late_con = punc_hr_[punc_hr_.Time !='dpf7_18'].punc_hr

early_sd = punc_hr_sd_[punc_hr_sd_.Time !='dpf8_0'].punc_hr
late_sd = punc_hr_sd_[punc_hr_sd_.Time !='dpf7_18'].punc_hr


from scipy.stats import sem
sem_con_early = sem(early_con, axis=None, ddof=0,nan_policy = 'omit')
sem_sd_early = sem(early_sd, axis=None, ddof=0,nan_policy = 'omit')
#later
sem_con_late = sem(late_con, axis=None, ddof=0,nan_policy = 'omit')
sem_sd_late = sem(late_sd, axis=None, ddof=0,nan_policy = 'omit')


punc_hr=control[['dpf7_18','dpf8_0']]
punc_hr_sd=sd[['dpf7_18','dpf8_0']]

punc_hr=punc_hr.reset_index()
punc_hr_sd=punc_hr_sd.reset_index()
punc_hr=punc_hr.drop(columns='index')
punc_hr_sd=punc_hr_sd.drop(columns='index')
#%%
jitter = 0.05
def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=13)
##baseline at zero
x,y = [-0.5,8],[0,0]
x1,y1 = [-0.2,0.2],[early_con.mean(),early_con.mean()]
x2,y2 = [0.8,1.2],[late_con.mean(),late_con.mean()]

x3,y3 = [-0.2,0.2],[early_sd.mean(),early_sd.mean()]
x4,y4 = [0.8,1.2],[late_sd.mean(),late_sd.mean()]

sem = [sem_con_early,sem_con_late]
x_sem = [0,1]
y_sem =[early_con.mean(),late_con.mean()]

sem_sd = [sem_sd_early,sem_sd_late]
x_sem_sd = [0,1]
y_sem_sd =[early_sd.mean(),late_sd.mean()]

row = 1
column = 2
fig, ax = plt.subplots()
ax = plt.subplot(row,column,1)
df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=punc_hr.values.shape), columns=punc_hr.columns)
df_x_jitter += np.arange(len(punc_hr.columns))
for col in punc_hr:
    ax.plot(df_x_jitter[col], punc_hr[col], 'o',color='#0079b9', alpha=.40, zorder=1, ms=8, mew=1)
ax.set_xticks(range(len(punc_hr.columns)))
ax.set_xticklabels(punc_hr.columns)
ax.set_xlim(-0.5,len(punc_hr.columns)-0.5)

for idx in punc_hr.index:
    ax.plot(df_x_jitter.loc[idx,['dpf7_18','dpf8_0']], punc_hr.loc[idx,['dpf7_18','dpf8_0']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
plt.plot(x,y,'k--',alpha=0.40)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None', capsize = 8, elinewidth=0.8,markeredgewidth=1.5,color='k')
ax.set_ylabel('Puncta Δ/ hour')    
plt.ylim(-12,17)

ax.tick_params(axis='both',which='major',labelsize=11)
sns.despine()
simpleaxis(ax)
ax = plt.subplot(row,column,2)
df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=punc_hr_sd.values.shape), columns=punc_hr_sd.columns)
df_x_jitter += np.arange(len(punc_hr_sd.columns))
for col in punc_hr_sd:
    ax.plot(df_x_jitter[col], punc_hr_sd[col], 'o',color='#f47d25', alpha=.40, zorder=1, ms=8, mew=1)
ax.set_xticks(range(len(punc_hr_sd.columns)))
ax.set_xticklabels(punc_hr_sd.columns)
ax.set_xlim(-0.5,len(punc_hr_sd.columns)-0.5)

for idx in punc_hr_sd.index:
    ax.plot(df_x_jitter.loc[idx,['dpf7_18','dpf8_0']], punc_hr_sd.loc[idx,['dpf7_18','dpf8_0']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
plt.plot(x,y,'k--',alpha=0.40)
plt.plot(x3,y3,'k-',alpha=1)
plt.plot(x4,y4,'k-',alpha=1)
plt.errorbar(x_sem_sd, y_sem_sd, yerr=sem_sd, linestyle='None', capsize = 8, elinewidth=0.8,markeredgewidth=1.5,color='k')
ax.tick_params(axis='both',which='major',labelsize=11)
simpleaxis(ax)
plt.ylim(-12,17) 
sns.despine()

#%%
