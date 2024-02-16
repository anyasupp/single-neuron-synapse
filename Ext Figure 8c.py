# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 14:18:32 2023

@author: Anya
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data_combined = pd.read_excel("Ext Figure8.xlsx",sheet_name='8c')
#%%take out cluster 1 from data
#these fish
# F8_1_210929_ON
# F3R_20210922_ON
#F1_20210811

data_combined = data_combined.set_index('Fish_ID')
data_combined = data_combined.drop(index=(['F8_1_210929_ON']))
data_combined = data_combined.drop(index=(['F3R_20210922_ON']))
data_combined = data_combined.drop(index=(['F1_20210811']))

groups=data_combined.groupby('Condition')

#make new dataframe with cyan,green,red with correct fish ID
sd_group = [groups.get_group('SD')]
sd_group = pd.concat(sd_group).reset_index()


control_group = [groups.get_group('Control')]
control_group = pd.concat(control_group).reset_index()

x1,y1 = [-0.5,2],[0,0]

color = sns.color_palette("tab10")

#%%
SMALL_SIZE = 10
MEDIUM_SIZE = 50
BIGGER_SIZE = 15



paper_rc= {'lines.linewidth': 2}
fig, ax = plt.subplots(1,3, sharex=False)
sns.set_style('ticks',rc=paper_rc)
sns.set_context('notebook',font_scale=1.5)

ax = plt.subplot(1,3,1)
ax = sns.pointplot(x="Time", y="Puncta", data=data_combined,hue="Condition",ci=68)

ax.get_legend().remove()
plt.xlim(-0.5,2.75 ) 
# plt.ylim(75,125) 
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=True,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.axvspan(0.1, 1.9, alpha=0.15, color='grey')
sns.despine()

grey = ['grey']*32

ax = plt.subplot(1,3,2)

ax = sns.lineplot(x="Time", y="RoC", data=control_group[control_group.Time!='dpf7_13'],
                  hue='Fish_ID', palette = grey,alpha=0.2)

ax = sns.pointplot(x="Time", y="RoC", data=control_group[control_group.Time!='dpf7_13'],ci=68,join=True, scale=0.85)

ax.get_legend().remove()
plt.plot(x1,y1,'k--',alpha=0.40)

ax.axvspan(-0.5, 1.9, alpha=0.15, color='grey')
ax.set_xlim(-0.5,1.5) 
ax.set_ylim(-40,80) 
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticklabels([])

grey = ['grey']*28

ax = plt.subplot(1,3,3)

ax = sns.lineplot(x="Time", y="RoC", data=sd_group[sd_group.Time!='dpf7_13'],
                  hue='Fish_ID', palette = grey,alpha=0.2)

ax = sns.pointplot(x="Time", y="RoC", data=sd_group[sd_group.Time!='dpf7_13'],ci=68,join=True, scale=0.85,color=color[1] )

ax.axvspan(-0.5, 1.9, alpha=0.15, color='grey')
ax.set_xlim(-0.5,1.5) 

ax.get_legend().remove()
ax.set_ylim(-40,80) 
ax.set_xlabel('')
ax.set_ylabel('')
ax.set_xticklabels([])