

# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 07:59:19 2022

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%% This is only Box 14

# 
clo = pd.read_excel("Ext Figure10d.xlsx",sheet_name='clonidine')
dmso = pd.read_excel("Ext Figure10d.xlsx",sheet_name='dmso')

#%%
def act_av(start,end,df):
    '''
    Parameters
    ----------
    def activity_per_hr : TYPE
    use with tenminute activity data 

    Returns
    -------
    None.

    '''
    res=df.drop(columns='Tenmin').iloc[start:end].mean()
    return res


def sleep_per_hr(start,end,df):
    '''
    Parameters
    ----------
    def sleep_per_hr : TYPE
        

    Returns
    -------
    None.

    '''
    time_diff = df.CLOCK.iloc[end]-df.CLOCK.iloc[start]
    print(time_diff)
    if time_diff>1:
        res=df.drop(columns='CLOCK').iloc[start:end].sum()/time_diff
    elif time_diff<1:
        res=df.drop(columns='CLOCK').iloc[start:end].sum()*time_diff
    # res=selected_time/time_diff
    return res

def sleep_total(start,end,df):
    '''
    Parameters
    ----------
    def sleep_per_hr : TYPE
        

    Returns
    -------
    None.

    '''
    res=df.drop(columns='CLOCK').iloc[start:end].sum()
    return res
# '''
#%% do 10 mins bins
minute_bins = 10
clo_ten = clo.groupby(clo.index // minute_bins).sum()
clo_ten['CLOCK']=clo_ten['CLOCK']/10
#minus last row
clo_ten=clo_ten.iloc[0:len(clo_ten)-1].copy()

dmso_ten = dmso.groupby(dmso.index // minute_bins).sum()
dmso_ten['CLOCK']=dmso_ten['CLOCK']/10
#minus last row
dmso_ten=dmso_ten.iloc[0:len(clo_ten)-1].copy()

clo_ten=clo_ten.set_index('CLOCK')
dmso_ten=dmso_ten.set_index('CLOCK')
#%%
clo_ten.iloc[240:244]=np.nan
clo_ten.iloc[271:282]=np.nan

dmso_ten.iloc[240:244]=np.nan
dmso_ten.iloc[271:283]=np.nan

df_clo_ten=pd.melt(clo_ten.reset_index(),id_vars='CLOCK',var_name='Fish_ID',value_name='average_activity')
df_dmso_ten=pd.melt(dmso_ten.reset_index(),id_vars='CLOCK',var_name='Fish_ID',value_name='average_activity')
#%%

def simpleaxis(ax):
    ax.tick_params(axis='x',labelsize=12.5)
    
row = 1
column =1
fig, ax = plt.subplots(row,column, sharex=True)
fig.suptitle('data ', fontsize=16)

ax = plt.subplot(row,column,1)

ax = sns.lineplot(x="CLOCK", y="average_activity", data=df_clo_ten,ci=68,color='#a6611a')
ax = sns.lineplot(x="CLOCK", y="average_activity", data=df_dmso_ten,ci=68,color='#7fbf7b')
ax = ax.axvspan(14, 24, alpha=0.30, color='grey')
plt.axvspan(38,48, alpha=0.30, color='grey') 
plt.axvspan(62,72, alpha=0.30, color='grey')
plt.axvspan(86,96, alpha=0.30, color='grey')
plt.axvspan(53.5,58, alpha=0.30, color='#9970ab')
sns.despine()
plt.xlim(32.0,80)
simpleaxis(ax)
plt.tick_params(axis='both',labelsize=12.5)

#%%
dmso_cropped = dmso[2437:2718]
clo_cropped = clo[2437:2718]
df_clo=pd.melt(clo_cropped,id_vars='CLOCK',var_name='Fish_ID',value_name='activity')
df_dmso=pd.melt(dmso_cropped,id_vars='CLOCK',var_name='Fish_ID',value_name='activity')

#%%zoomed_1 pulse
dmso_zoomed=dmso[2525:2555]
clo_zoomed=clo[2525:2555]
df_clo_zoomed=pd.melt(clo_zoomed,id_vars='CLOCK',var_name='Fish_ID',value_name='activity')
df_dmso_zoomed=pd.melt(dmso_zoomed,id_vars='CLOCK',var_name='Fish_ID',value_name='activity')
#%%
row = 1
column =2
fig, ax = plt.subplots(row,column, sharex=True)
fig.suptitle('data ', fontsize=18)

ax = plt.subplot(row,column,1)

ax = sns.lineplot(x="CLOCK", y="activity", data=df_clo,ci=68,color='#a6611a')
ax = sns.lineplot(x="CLOCK", y="activity", data=df_dmso,ci=68,color='#7fbf7b')
ax = ax.axvspan(53.45, 53.466, alpha=0.30, color='grey') 
plt.axvspan(53.95, 53.966, alpha=0.30, color='grey')

sns.despine()

plt.axvspan(54.45, 54.466, alpha=0.30, color='grey')
plt.axvspan(54.95, 54.966, alpha=0.30, color='grey')

plt.axvspan(55.45, 55.466, alpha=0.30, color='grey')
plt.axvspan(55.95, 55.966, alpha=0.30, color='grey')

plt.axvspan(56.45, 56.466, alpha=0.30, color='grey')
plt.axvspan(56.95, 56.966, alpha=0.30, color='grey')

plt.axvspan(57.45, 57.466, alpha=0.30, color='grey')
plt.axvspan(57.95, 57.966, alpha=0.30, color='grey')
plt.tick_params(axis='both',labelsize=12.5)


ax = plt.subplot(row,column,2)

ax = sns.lineplot(x="CLOCK", y="activity", data=df_clo_zoomed,ci=68,color='#a6611a')
ax = sns.lineplot(x="CLOCK", y="activity", data=df_dmso_zoomed,ci=68,color='#7fbf7b')
plt.axvspan(54.9825, 54.9992, alpha=0.30, color='grey') 

plt.tick_params(axis='both',labelsize=12.5)

sns.despine()

#%%
clo_cropped_sleep=clo_cropped.reset_index()
#%%
#let's pick clock at around 54.98 which is iloc 99
# where fish are sleeping == 0
sleepers = pd.DataFrame(clo_cropped_sleep.iloc[99]==0)
#get their fish ID
sleeper_id = sleepers.index[sleepers[99]==True]
#get fish at 99 that are asleep in dataframe
clo_cropped_sleepers = clo_cropped_sleep[sleeper_id]
clo_cropped_sleepers.insert(1,'CLOCK',clo_cropped_sleep.CLOCK,True)
df_clo_cropped_sleepers=pd.melt(clo_cropped_sleepers,id_vars='CLOCK',var_name='Fish_ID',value_name='activity')


#%%
row = 1
column =1
fig, ax = plt.subplots(row,column, sharex=True)
fig.suptitle('data ', fontsize=16)

ax = plt.subplot(row,column,1)

ax = sns.lineplot(x="CLOCK", y="activity", data=df_clo_cropped_sleepers,alpha=0.75,
                  hue='Fish_ID',palette='copper')
plt.xlim(54.98,55.11)
ax.get_legend().remove()
# ax = sns.lineplot(x="CLOCK", y="activity", data=df_dmso,ci=68,color='#7fbf7b')
# ax = ax.axvspan(14, 24, alpha=0.30, color='grey') 
sns.despine()
plt.tick_params(axis='both',labelsize=11)
