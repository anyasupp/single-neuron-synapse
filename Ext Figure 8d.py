# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 17:18:44 2022

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
data = pd.read_excel("Ext Figure8.xlsx",sheet_name='8d')

#remove outlier
df = data[data.Fish_ID !='F5_20211027_ON']

#%%
datamelt = pd.melt(df,id_vars=['Fish_ID','Condition','Cluster'],value_vars=['dpf7_18','dpf8_0'],
                   var_name='Time',value_name='puncta_hour')


palette = ['lightseagreen','orangered', 'dodgerblue','gold']

cluster = datamelt.groupby('Cluster')

c_zero = cluster.get_group(0)
c_one = cluster.get_group(1)
c_two = cluster.get_group(2)
c_three = cluster.get_group(3)
#%%
def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=8)
row = 1
column = 4
fig, ax = plt.subplots(row,column, sharex=True)
# sns.set(style='ticks')
ylim=-6
ymax = 8
xlim=-0.5
xmax=1.5
x,y = [-0.5,8],[0,0]

ax = plt.subplot(row,column,1)
ax = sns.pointplot(x="Time", y="puncta_hour", data=c_zero,hue="Condition",ci=68,scale=0.8,errwidth=2.5,dodge=0.1)
plt.plot(x,y,'k--',alpha=0.40)

plt.ylim(ylim,ymax)
plt.xlim(xlim,xmax)


ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
simpleaxis(ax)
ax = plt.subplot(row,column,2)
ax = sns.pointplot(x="Time", y="puncta_hour", data=c_one,hue="Condition",ci=68,scale=0.8,errwidth=2.5,dodge=0.1)
plt.ylim(ylim,ymax)
plt.plot(x,y,'k--',alpha=0.40)
plt.xlim(xlim,xmax)

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
simpleaxis(ax)
ax = plt.subplot(row,column,3)
ax = sns.pointplot(x="Time", y="puncta_hour", data=c_two,hue="Condition",ci=68,scale=0.8,errwidth=2.5,dodge=0.1)
plt.ylim(ylim,ymax)
plt.plot(x,y,'k--',alpha=0.40)
plt.xlim(xlim,xmax)

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
simpleaxis(ax)

ax = plt.subplot(row,column,4)
ax = sns.pointplot(x="Time", y="puncta_hour", data=c_three,hue="Condition",ci=68,scale=0.8,errwidth=2.5,dodge=0.1)
plt.ylim(ylim,ymax)
plt.plot(x,y,'k--',alpha=0.40)
plt.xlim(xlim,xmax)

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
simpleaxis(ax)