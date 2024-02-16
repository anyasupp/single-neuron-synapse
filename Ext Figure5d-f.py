# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 14:19:08 2022

@author: AnyaS
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data = pd.read_excel("Ext Figure5.xlsx",sheet_name='d-f')

#%%
groups=data.groupby('Condition')

LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

palette = ['lightseagreen','orangered', 'dodgerblue','gold']

cluster = LD_group.groupby('Morphology_cluster')
c_zero = cluster.get_group(0)
c_one = cluster.get_group(1)
c_two = cluster.get_group(2)
c_three = cluster.get_group(3)
color='gray'
c0=palette[0]
c1=palette[1]
c2=palette[2]
c3=palette[3]


#%%
rows = 3
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
ax = sns.pointplot(x="Time", y="Puncta", data=LD_group,hue="Morphology_cluster",errorbar=('ci',68),dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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
ax = sns.pointplot(x="Time", y="Puncta", data=c_zero,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c0)
ax.get_legend().remove()
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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
ax = sns.pointplot(x="Time", y="Puncta", data=c_one,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c1)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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
ax = sns.pointplot(x="Time", y="Puncta", data=c_two,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c2)
# ax.get_legend().remove()
# ax = sns.pointplot(x="Time", y="Puncta", data=c_two,ci=68,dodge=True,color=c2)
ax.get_legend().remove()

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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

ax = plt.subplot(rows,cols,5)
ax = sns.pointplot(x="Time", y="Puncta", data=c_three,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c3)
# ax.get_legend().remove()
# ax = sns.pointplot(x="Time", y="Puncta", data=c_two,ci=68,dodge=True,color=c2)
ax.get_legend().remove()

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=LD_group,hue="Morphology_cluster",
                   errorbar=('ci',68),dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
plt.ylim(-45,45)
ax.tick_params(
    axis='x',          # changes apply to the x-axis
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
plt.xlim(-0.5,5.5 )
ax.get_legend().remove()
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

ax = plt.subplot(rows,cols,8)
plt.plot(x1,y1,'k--',alpha=0.65)
# ax = sns.pointplot(x="Time", y="RoC", data=c_two,ci=68,dodge=True,color=c2)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_one,errorbar=('ci',68),dodge=True,color=c1,hue='Fish_ID')

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax) 
ax.tick_params(axis='both', which='major', labelsize=14)  
ax.get_legend().remove()

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

ax = plt.subplot(rows,cols,9)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_two,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c2)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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


ax = plt.subplot(rows,cols,10)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="Puncta_diff", data=c_three,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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


#%Roc

ax = plt.subplot(rows,cols,11)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=LD_group,hue="Morphology_cluster",errorbar=('ci',68),dodge=True, palette = palette)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
ax.tick_params(axis='both', which='major', labelsize=14)  

plt.xlim(-0.5,5.5 )
plt.ylim(-30,30)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('RoC(%)')    
ax.get_legend().remove()
sns.despine()

ylim=-70
ymax=70
ax = plt.subplot(rows,cols,12)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_zero,hue="Fish_ID",errorbar=('ci',68),dodge=True,color=c0)
ax.tick_params(axis='both', which='major', labelsize=14)  
ax.get_legend().remove()
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
plt.xlim(-0.5,5.5 )
plt.ylim(ylim,ymax)


# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
sns.despine()

ax = plt.subplot(rows,cols,13)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_one,errorbar=('ci',68),dodge=True,color=c1,hue='Fish_ID')
ax.get_legend().remove()

ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('')    
sns.despine()

ax = plt.subplot(rows,cols,14)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_two,hue="Fish_ID",errorbar=('ci', 68),dodge=True,color=c2)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()


ax = plt.subplot(rows,cols,15)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=c_three,hue="Fish_ID",errorbar=('ci', 68),dodge=True,color=c3)
ax.axvspan(-1, -0.1, alpha=0.3, color=color)
ax.axvspan(1.2, 1.9, alpha=0.3, color=color)
ax.axvspan(3.2, 3.9, alpha=0.3, color=color)
ax.axvspan(5.2, 5.9, alpha=0.3, color=color)
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
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()