# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 16:37:49 2022

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%% Cluster morphology 
puncta_int_ratio = pd.read_excel("Ext Figure6.xlsx",sheet_name='ext6')

puncta_int_ratio = puncta_int_ratio[puncta_int_ratio.Fish_ID!='F5_1112']
groups=puncta_int_ratio.groupby('Condition')

LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)
LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)
FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)

data_combined_3T_no7dpf =  puncta_int_ratio[puncta_int_ratio.Time !='7dpf_0']
LD_group_no7= LD_group[LD_group.Time !='7dpf_0']
LL_group_no7= LL_group[LL_group.Time !='7dpf_0']
FR_group_no7= FR_group[FR_group.Time !='7dpf_0']
#%%
palette_notype1 = ['orangered', 'dodgerblue','gold']
palette = ['lightseagreen','orangered', 'dodgerblue','gold']
c0 =palette[0]
c1=palette[1]
c2=palette[2]
c3=palette[3]
color='gray'
# color='khaki'

#baseline at zero
x1,y1 = [-0.5,2],[0,0]

rows=2
cols=3

sns.set_context("notebook")
sns.set_style('ticks')
def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=12)
    
fig, ax = plt.subplots(rows,cols, sharex=False)
# fig.suptitle('Day/Night FingR.PSD95 dynamics by morphology cluster LD', fontsize=16)
fig.set_size_inches(11.5,7.5)
#fig.text(0.5, 0.04, 'Time', ha='center', va='center', fontsize=12)


ylim=0.1
ymax=0.65
ax = plt.subplot(rows,cols,1)
ax = sns.pointplot(x="Time", y="int_ratio", data=LD_group,hue="updated_cluster",ci=68,dodge=True, 
                   palette = palette_notype1,linestyles='--')

ax.axvspan(-1, -0.1, alpha=0.55, color=color,zorder=2)
ax.axvspan(1.2, 1.9, alpha=0.55, color=color,zorder=1)
plt.xlim(-0.5,2.5) 
plt.ylim(ylim,ymax) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('Intensity')    
ax.get_legend().remove()
sns.despine()
simpleaxis(ax)
ax = plt.subplot(rows,cols,2)
ax = sns.pointplot(x="Time", y="int_ratio", data=LL_group,hue="updated_cluster",ci=68,dodge=True, 
                   palette = palette,linestyles='--')

ax.axvspan(-1, -0.1, alpha=0.55, color='khaki',zorder=2)
ax.axvspan(1.2, 1.9, alpha=0.55, color='khaki',zorder=1)
plt.xlim(-0.5,2.5) 
plt.ylim(ylim,ymax) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
simpleaxis(ax)
ax = plt.subplot(rows,cols,3)
ax = sns.pointplot(x="Time", y="int_ratio", data=FR_group,hue="updated_cluster",ci=68,dodge=True, 
                   palette = palette,linestyles='--')

ax.axvspan(-1, -0.1, alpha=0.55, color='khaki',zorder=2)
ax.axvspan(1.2, 1.9, alpha=0.55, color='khaki',zorder=1)
plt.xlim(-0.5,2.5) 
plt.ylim(ylim,ymax) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

ylim=-50
ymax=110
simpleaxis(ax)
ax = plt.subplot(rows,cols,4)
plt.plot(x1,y1,'k--',alpha=0.65,zorder=1)
ax = sns.pointplot(x="Time", y="ratio_roc", data=LD_group_no7,hue="updated_cluster",ci=68,dodge=True,
                   palette = palette_notype1,linestyles='--')
#g = sns.catplot(x="Time", y="Change", data=data2,hue="Fish_ID",ci=68,dodge=True, col = 'Segment K-means PCA',kind='point')
ax.axvspan(0.16, 0.92, alpha=0.5, color=color) 
plt.xlim(-0.5,1.5 )  
plt.ylim(ylim,ymax) 


ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('Int RoC (%)')    
ax.get_legend().remove()
sns.despine()
simpleaxis(ax)
ax = plt.subplot(rows,cols,5)
plt.plot(x1,y1,'k--',alpha=0.65,zorder=1)
ax = sns.pointplot(x="Time", y="ratio_roc", data=LL_group_no7,hue="updated_cluster",ci=68,dodge=True,
                   palette = palette,linestyles='--')
ax.axvspan(0.16, 0.92, alpha=0.5, color='khaki',zorder=1) 
plt.xlim(-0.5,1.5 )  
plt.ylim(ylim,ymax) 


ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
simpleaxis(ax)
ax = plt.subplot(rows,cols,6)
plt.plot(x1,y1,'k--',alpha=0.65,zorder=1)
ax = sns.pointplot(x="Time", y="ratio_roc", data=FR_group_no7,hue="updated_cluster",ci=68,dodge=True,
                   palette = palette,linestyles='--')
ax.axvspan(0.16, 0.92, alpha=0.5, color='khaki',zorder=1) 
plt.xlim(-0.5,1.5 )  
plt.ylim(ylim,ymax) 


ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
#%%
#extened figure 6c

ll_fr = pd.read_excel("Ext Figure6.xlsx",sheet_name='ext6c_LLvs_FR')

ll_ld = pd.read_excel("Ext Figure6.xlsx",sheet_name='ext6c_LLvs_LD')


pal = ['#dd90b5', 'dodgerblue','orangered','gold']

palette_notype1 = ['#dd90b5','gold', 'dodgerblue','orangered']


sns.set_context("notebook")
sns.set_style('ticks')
def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=12)

rows,cols = 1,2
fig, ax = plt.subplots(rows,cols, sharex=False)
blx1,bly1 = [-0.5,9],[0,0]
ylim=0.1
ymax=0.65
ax = plt.subplot(rows,cols,1)
plt.plot(blx1,bly1,'k--',alpha=0.25,zorder=1)

ax = sns.pointplot(x="Time", y="ratio_roc", data=ll_ld,hue="type",errorbar=('ci', 68),dodge=True, 
                   palette = palette_notype1,linestyles='--',join=False)
ax.get_legend().remove()
ax.axvspan(0.16, 0.92, alpha=0.5, color='gray') 
plt.xlim(-0.35,1.5)
# ax = sns.pointplot(x="Time", y="ratio_roc", data=LL_group_no7,errorbar=('ci', 68),dodge=True, 
#                    color = '#dd90b5',linestyles='--',join=False)

sns.despine()
ax = plt.subplot(rows,cols,2)

plt.plot(blx1,bly1,'k--',alpha=0.25,zorder=1)
ax = sns.pointplot(x="Time", y="ratio_roc", data=ll_fr,hue="type",errorbar=('ci', 68),dodge=True, 
                   palette = pal,linestyles='--',join=False)
ax.axvspan(0.16, 0.92, alpha=0.5, color='khaki',zorder=1) 

plt.xlim(-0.35,1.5)
ax.get_legend().remove()
simpleaxis(ax)

# 
