# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:12:14 2023

@author: Anya
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

#%% get sd data (with puncta during extended wake)
sd = pd.read_excel("Ext Figure8.xlsx",sheet_name='8f')
sd = sd[sd.Cluster!=0] # without cluster 1
#%%
def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=12)
row = 1
column = 2
fig, ax = plt.subplots(row,column, sharex=True)
fig.suptitle('puncta post SD vs subsequent sleep excl Type 1', fontsize=16)
# sns.set_context('notebook',font_scale=1.5)

ax = plt.subplot(row,column,1)
ax = sns.regplot(data=sd,y='sleepph_adjusted_exp',x='puncta_post_sd',fit_reg=True,color='#ff7f0e')

x = sd[['sleepph_adjusted_exp','puncta_post_sd']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])

simpleaxis(ax)
sns.despine()
plt.ylim(0.5,48)

ax = plt.subplot(row,column,2)
ax = sns.regplot(data=sd,y='bl_4h_av',x='puncta_post_sd',fit_reg=True,color='#ff7f0e')

x = sd[['bl_4h_av','puncta_post_sd']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])

simpleaxis(ax)
sns.despine()