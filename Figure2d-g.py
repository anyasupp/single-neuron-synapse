# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 19:42:10 2023
FIGURE2 d-g
subtypeLD_vs_LL
use type => LL is = 0 
as preprocessed data from LD to exclude cluster 0 (morphology type 1) 

average data Figure f,g is from average per fish for day and night dynamics.
This excludes 9dpf ZT10 as we found there is a large developmental effect. 
*Note: Average data panel f,g data comes from all neurons in Ext Fig5
not just tracked to whole 6 timepoints like d and e.
@author: Anya
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%load data
data = pd.read_excel("Figure2.xlsx",sheet_name='2d_e')
data_vs_LL = pd.read_excel("Figure2.xlsx",sheet_name='2f_g')

#take out 0 (type1) only from LD dataset
df_vsLL = data_vs_LL[~((data_vs_LL['Condition'] == 'LD') & (data_vs_LL['type'] == 0))]

groups=data.groupby('Condition')

LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

# LL_group = [groups.get_group('LL')]
# LL_group = pd.concat(LL_group)


LD_group= LD_group[LD_group.Morphology_cluster !=0]
palette = ['orangered', 'dodgerblue','gold']
#%
cluster = LD_group.groupby('Morphology_cluster')
# c_zero = cluster.get_group(0)
c_one = cluster.get_group(1)
c_two = cluster.get_group(2)
c_three = cluster.get_group(3)
color='gray'
c1=palette[0]
c2=palette[1]
c3=palette[2]

#%% plot
rows = 2
cols = 5
sns.set_style('ticks')
sns.set_context('notebook')
fig, ax = plt.subplots(rows,cols, sharex=False)
# fig.suptitle('LD_cluster_thickness+skin+loc', fontsize=16)
fig.set_size_inches(14,8)
#baseline at zero
x1,y1 = [-0.5,6.5],[0,0]


ylim=40
ymax=350
ax = plt.subplot(rows,cols,1)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax = sns.pointplot(x="Time", y="Puncta", data=LD_group,hue="Morphology_cluster",errorbar=('ci',68),dodge=True, palette = palette)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
# plt.ylim(ylim,ymax)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax.set_xlabel('')
ax.set_ylabel('Puncta')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,2)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax = sns.pointplot(x="Time", y="Puncta", data=c_one,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c1)
ax.get_legend().remove()
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax)
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()

ax = plt.subplot(rows,cols,3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax = sns.pointplot(x="Time", y="Puncta", data=c_two,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c2)

plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,4)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax = sns.pointplot(x="Time", y="Puncta", data=c_three,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c3)
ax.get_legend().remove()
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()
#%puncta diff
ylim=-125
ymax=125

ax = plt.subplot(rows,cols,6)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=LD_group,hue="Morphology_cluster",errorbar=('ci',68),dodge=True, palette = palette)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.xlim(-0.5,5.5 )
plt.ylim(-45,45)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax.set_xlabel('')
ax.set_ylabel('Δ Puncta')    
ax.get_legend().remove()
sns.despine()
#%
ax = plt.subplot(rows,cols,7)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.plot(x1,y1,'k--',alpha=0.65)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_one,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c1)

ax.get_legend().remove()
ax.tick_params(axis='both', which='major', labelsize=14)  


# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()

ax = plt.subplot(rows,cols,8)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_two,errorbar=('ci',68),dodge=True,color=c2,hue='Fish_ID')

plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  
ax.get_legend().remove()

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()

ax = plt.subplot(rows,cols,9)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_three,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c3)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax)  
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

colors = ['#dd90b5','orangered','dodgerblue','gold']

ax = plt.subplot(rows,cols,5)
ax.axvspan(0.16, 0.92, alpha=0.15, color='grey') 
ax = sns.pointplot(x="time", y='av_Puncta', data=df_vsLL[df_vsLL.Condition!='LL'],hue="type",
                   errorbar=('ci', 68),dodge=True,palette=palette,join=True)

plt.xlim(-0.5,1.5 )  
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.set_ylabel('')    
ax.set_xlabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,10)
ax.axvspan(0.16, 0.92, alpha=0.15, color='grey') 
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="time", y="av_Puncta_diff", data=df_vsLL,hue="type",
                   errorbar=('ci', 68),dodge=True,palette=colors,join=False)
plt.xlim(-0.5,1.5 )  

ax.tick_params(axis='both', which='major', labelsize=14)  

# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
# ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
