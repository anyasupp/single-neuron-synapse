# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:38:54 2023

@author: Anya
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_excel("Ext Figure5.xlsx",sheet_name='g-j')

#%%
groups=data.groupby('Condition')

LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)
LL_group = LL_group[LL_group.Time !='dpf9_10']



palette = ['lightseagreen', 'dodgerblue','gold']
#

LL_group_no7 = LL_group[LL_group.Time !='dpf7_0']


#%%
cluster = LL_group.groupby('Morphology_cluster')
c_zero = cluster.get_group(0)
# c_one = cluster.get_group(1)
c_two = cluster.get_group(2)
c_three = cluster.get_group(3)
color='khaki'
c0=palette[0]
c2=palette[1]
c3=palette[2]

rows = 2
cols = 5
sns.set_style('ticks')
sns.set_context('notebook')
fig, ax = plt.subplots(rows,cols, sharex=False)
# fig.suptitle('LD_cluster_thickness+skin+loc', fontsize=16)
fig.set_size_inches(14,8)
#baseline at zero
x1,y1 = [-0.5,6.5],[0,0]
xlim = -0.5
xmax=4.5

ylim=40
ymax=350
ax = plt.subplot(rows,cols,1)
ax = sns.pointplot(x="Time", y="Puncta", data=LL_group,hue="Morphology_cluster",errorbar=('ci',68),dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(xlim,xmax)
# plt.ylim(ylim,ymax)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.xlim(-0.5,4.5)
ax.set_xlabel('')
ax.set_ylabel('Puncta')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,2)
ax = sns.pointplot(x="Time", y="Puncta", data=c_zero,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c0)
ax.get_legend().remove()
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(xlim,xmax)

plt.ylim(ylim,ymax)
ax.tick_params(axis='both', which='major', labelsize=14)  
plt.xlim(-0.5,4.5)

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
ax = sns.pointplot(x="Time", y="Puncta", data=c_two,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c2)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(xlim,xmax)

plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  
plt.xlim(-0.5,4.5)

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
ax = sns.pointplot(x="Time", y="Puncta", data=c_three,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c3)
# ax.get_legend().remove()
# ax = sns.pointplot(x="Time", y="Puncta", data=c_two,ci=68,dodge=True,color=c2)
ax.get_legend().remove()

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(xlim,xmax)

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
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=LL_group,hue="Morphology_cluster",errorbar=('ci',68),dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  
plt.xlim(-0.5,4.5)
plt.xlim(-0.5,4.5)

plt.xlim(xlim,xmax)

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
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_zero,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c0)
plt.ylim(ylim,ymax)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(xlim,xmax)

ax.get_legend().remove()
ax.tick_params(axis='both', which='major', labelsize=14)  



# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    

sns.despine()

ax = plt.subplot(rows,cols,8)
plt.plot(x1,y1,'k--',alpha=0.65)
# ax = sns.pointplot(x="Time", y="RoC", data=c_two,ci=68,dodge=True,color=c2)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_two,errorbar=('ci',68),dodge=True,color=c2,hue='Fish_ID')
plt.xlim(xlim,xmax)


ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)

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
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_three,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)

plt.ylim(ylim,ymax)  
ax.tick_params(axis='both', which='major', labelsize=14)  
plt.xlim(xlim,xmax)


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




ax = plt.subplot(rows,cols,5)
ax = sns.pointplot(x="time", y="Puncta", data=LL_group_no7,hue="Morphology_cluster",errorbar=('ci', 68),dodge=True,palette=palette,join=True)

plt.xlim(-0.5,1.5 )  
ax.tick_params(axis='both', which='major', labelsize=14)  

ax.axvspan(0.16, 0.92, alpha=0.15, color='khaki') 
ax.set_ylabel('')    
ax.set_xlabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(rows,cols,10)


plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="time", y="Puncta_diff", data=LL_group_no7,hue="Morphology_cluster",errorbar=('ci', 68),dodge=True,palette=palette,join=False)
plt.xlim(-0.5,1.5 )  
ax.axvspan(0.16, 0.92, alpha=0.15, color='khaki') 
ax.tick_params(axis='both', which='major', labelsize=14)  

# ax = sns.pointplot(x="time", y="Puncta_diff", data=LL_group_no7,errorbar=('ci', 68),dodge=True,join=True)



# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.ylim(-25,23)
# ADDED: Remove labels.
# ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()