# -*- coding: utf-8 -*-
"""
Created on Sun Nov  6 11:08:39 2022
plot per luc SD data
@author: AnyaS
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

control = pd.read_excel("Ext Figure8.xlsx",sheet_name='8b_control')
sd = pd.read_excel("Ext Figure8.xlsx",sheet_name='8b_sd')

#%% do rolling average to try to smooth
df=control.set_index('Time') 
df_av = df.rolling(window=50).mean()
df_av=df_av.reset_index()

# do for SD
sd_df=sd.set_index('Time') 
df_sd_av = sd_df.rolling(window=50).mean()
df_sd_av=df_sd_av.reset_index()


# Detrend by differencing
#create dataframe with fish values no time to detrend
df_av=df_av.dropna()
df_av_1=df_av.drop(columns='Time')

#make new dataframe with all detrended fish
#numbers of rows is different (-1) post differencing
detrended_df = df_av['Time'].drop(index=49).reset_index()
detrended_df=detrended_df.drop(columns='index')

#bigger loop to go through each columns/fish
for fish in range(1,len(df_av_1.columns)):
    X= df_av_1.iloc[:, fish-1:fish].values
    diff = []
    #inner loop differencing each row (within fish)
    for i in range(1, len(X)):
        value = X[i] - X[i - 1]
        diff.append(value)
    fish_diff= np.concatenate(diff,axis=0)
    detrended_df.insert(loc=fish,column='FISH_{:.0f}'.format(fish),value=fish_diff)

# detrended_df_melt= detrended_df.melt(value_name='detrended',var_name='Fish_ID',id_vars='Time')
#rollwing mean again 
df=detrended_df.set_index('Time') 
df_detrended_control = df.rolling(window=50).mean()
df_detrended_control=df_detrended_control.reset_index()

df_detrended_control_melt= df_detrended_control.melt(value_name='perluc',var_name='Fish_ID',id_vars='Time')
df_detrended_control_melt=df_detrended_control_melt.dropna()

# now do for SD 
#create dataframe with fish values no time to detrend
df_sd_av=df_sd_av.dropna()
df_sd_av_1=df_sd_av.drop(columns='Time')

#make new dataframe with all detrended fish
#numbers of rows is different (-1) post differencing
detrended_df_sd = df_sd_av['Time'].drop(index=49).reset_index()
detrended_df_sd=detrended_df_sd.drop(columns='index')

#bigger loop to go through each columns/fish
for fish in range(1,len(df_sd_av_1.columns)):
    X= df_sd_av_1.iloc[:, fish-1:fish].values
    diff = []
    #inner loop differencing each row (within fish)
    for i in range(1, len(X)):
        value = X[i] - X[i - 1]
        diff.append(value)
    fish_diff= np.concatenate(diff,axis=0)
    detrended_df_sd.insert(loc=fish,column='FISH_{:.0f}'.format(fish),value=fish_diff)

# detrended_df_melt= detrended_df.melt(value_name='detrended',var_name='Fish_ID',id_vars='Time')
#rollwing mean  
df=detrended_df_sd.set_index('Time') 
df_detrended_sd = df.rolling(window=50).mean()
df_detrended_sd=df_detrended_sd.reset_index()

df_detrended_sd_melt= df_detrended_sd.melt(value_name='perluc',var_name='Fish_ID',id_vars='Time')
df_detrended_sd_melt=df_detrended_sd_melt.dropna()

#check fish 
row =2
column = 1
fig, ax = plt.subplots(row,column, sharex=True)

ax = plt.subplot(row,column,1)
ax=sns.lineplot(data=df_detrended_control_melt,x='Time',y='perluc',hue='Fish_ID',palette='Paired')
ax = plt.subplot(row,column,2)
ax=sns.lineplot(data=df_detrended_sd_melt,x='Time',y='perluc',hue='Fish_ID',palette='Paired')
#%% take out fish looked weird for controls
df_detrended_control=df_detrended_control.drop(columns='FISH_6') 
df_detrended_control=df_detrended_control.drop(columns='FISH_2') 
df_detrended_control=df_detrended_control.drop(columns='FISH_10')

#melt
df_detrended_control_melt= df_detrended_control.melt(value_name='perluc',var_name='Fish_ID',id_vars='Time')
df_detrended_control_melt=df_detrended_control_melt.dropna()
#take out Fish 15 from SD
df_detrended_sd=df_detrended_sd.drop(columns='FISH_15') 

#melt
df_detrended_sd_melt= df_detrended_sd.melt(value_name='perluc',var_name='Fish_ID',id_vars='Time')
df_detrended_sd_melt=df_detrended_sd_melt.dropna()

#%%double check each fish looked ok 
row =2
column = 1
fig, ax = plt.subplots(row,column, sharex=True)

ax = plt.subplot(row,column,1)
ax=sns.lineplot(data=df_detrended_control_melt,x='Time',y='perluc',hue='Fish_ID',palette='Paired')
ax = plt.subplot(row,column,2)
ax=sns.lineplot(data=df_detrended_sd_melt,x='Time',y='perluc',hue='Fish_ID',palette='Paired')
#%% make average graph
row =1
column = 1
fig, ax = plt.subplots(row,column, sharex=True)
# fig.suptitle('Daytime melatonin/clonidine experiments', fontsize=16)

def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=12)
    
ax = plt.subplot(row,column,1)

ax=sns.lineplot(data=df_detrended_control_melt,x='Time',y='perluc',color='#1f77b4')
ax=sns.lineplot(data=df_detrended_sd_melt,x='Time',y='perluc',color='#ff7f0e')
simpleaxis(ax)
sns.despine()

