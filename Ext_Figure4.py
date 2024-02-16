# -*- coding: utf-8 -*-
"""
Created on Thu May 13 17:56:36 2021

@author: AnyaS
Graph included all neurons imaged. Even neurons with pigmentated background

"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%

data2 = pd.read_excel("Ext Figure4.xlsx",sheet_name='ext4bc')

#%% replace 0 on RoC at 7dpf_ with nan
data2['RoC'][data2.Time == 'dpf7_0'] = np.nan

#%%
groups=data2.groupby('Condition')

#make new dataframe with cyan,green,red with correct fish ID
LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)


FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)

#%%
palette= ['#5ec0eb','#dd90b5','#00ac87']
#'#dd90b5-- pink','#5ec0eb = blue', '00ac87 = green

#%% change data_to have 'evening' first so day dynamics
target_row = 22 
idx = [target_row] +[i for i in range(len(data2))if i!= target_row]
data = data2.iloc[idx]
#%%Take last timepoint out for combined 
data = data[data.Time!='dpf9_10']
#%%
widthline =2
# 
sns.set_context("talk")
sns.set_style('ticks')
fig, ax = plt.subplots(2,5, sharex=False)
fig.suptitle('FingR.PSD95 dynamics LD, LL and FR', fontsize=16)
fig.set_size_inches(18,8)

##baseline at zero
x1,y1 = [-0.5,8],[0,0]

ax = plt.subplot(2,5,1)
ax = sns.pointplot(x="Time", y="Puncta", data=data2,hue="Condition",ci=68,dodge=True, 
                   palette = palette,scale=0.65,errwidth=2.5)

ax.axvspan(-1, -0.1, alpha=0.15, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.15, color='grey')
ax.axvspan(3.2, 3.9, alpha=0.15, color='grey')
ax.axvspan(5.2, 5.9, alpha=0.15, color='grey')
plt.xlim(-0.5,5.5 ) 
# plt.ylim(40,315) 

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
#ax.set_ylabel('')    
ax.set_xlabel('')

ax = plt.subplot(2,5,2)
# LD = ['dodgerblue']


ax = sns.pointplot(x="Time", y="Puncta", data=LD_group,hue="Fish_ID",ci=68,dodge=True,color='#5ec0eb',scale=0.65)

ax.axvspan(-1, -0.1, alpha=0.55, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.55, color='grey')
ax.axvspan(3.2, 3.9, alpha=0.55, color='grey')
ax.axvspan(5.2, 5.9, alpha=0.55, color='grey')
plt.xlim(-0.5,5.5 ) 
plt.ylim(35,320) 


ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

# ADDED: Remove labels.
ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
# plt.text(1.25,270,'Cluster 1',fontsize=7)

ax = plt.subplot(2,5,3)
# LL = ['orangered']
ax = sns.pointplot(x="Time", y="Puncta", data=LL_group,hue="Fish_ID",ci=68,dodge=True,color='#dd90b5',scale=0.65)
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.axvspan(3.2, 3.9, alpha=0.3, color='khaki')
ax.axvspan(5.2, 5.9, alpha=0.3, color='khaki')
plt.xlim(-0.5,5.5 ) 
plt.ylim(35,320) 
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

ax = plt.subplot(2,5,4)

ax = sns.pointplot(x="Time", y="Puncta", data=FR_group,hue="Fish_ID",ci=68,dodge=True,color='#00ac87',scale=0.65)
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.axvspan(3.2, 3.9, alpha=0.3, color='khaki')
ax.axvspan(5.2, 5.9, alpha=0.3, color='khaki')
plt.xlim(-0.5,5.5 ) 
plt.ylim(35,320) 
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
#ROC
ax = plt.subplot(2,5,6)
ax = sns.pointplot(x="Time", y="RoC", data=data2,hue="Condition",ci=68,dodge=True, palette = palette,scale=0.65,
                   errwidth=2.5)
ax.axvspan(-1, -0.1, alpha=0.15, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.15, color='grey')
ax.axvspan(3.2, 3.9, alpha=0.15, color='grey')
ax.axvspan(5.2, 5.9, alpha=0.15, color='grey')
plt.xlim(-0.5,5.5 ) 
plt.plot(x1,y1,'k--',alpha=0.65)
ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    top=False)         # ticks along the top edge are off
    # labelbottom=False) # labels along the bottom edge are off
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
# ax.set_ylabel('')    
# ax.set_xlabel('')

ax = plt.subplot(2,5,7)
# LD = ['dodgerblue']


ax = sns.pointplot(x="Time", y="RoC", data=LD_group,hue="Fish_ID",ci=68,dodge=True,color='#5ec0eb',scale=0.65)

ax.axvspan(-1, -0.1, alpha=0.55, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.55, color='grey')
ax.axvspan(3.2, 3.9, alpha=0.55, color='grey')
ax.axvspan(5.2, 5.9, alpha=0.55, color='grey')
plt.xlim(-0.5,5.5 ) 
plt.ylim(-65,65) 


ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
# ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
# plt.text(1.25,270,'Cluster 1',fontsize=7)

ax = plt.subplot(2,5,8)
# LL = ['orangered']
ax = sns.pointplot(x="Time", y="RoC", data=LL_group,hue="Fish_ID",ci=68,dodge=True,color='#dd90b5',scale=0.65)
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.axvspan(3.2, 3.9, alpha=0.3, color='khaki')
ax.axvspan(5.2, 5.9, alpha=0.3, color='khaki')
plt.xlim(-0.5,5.5 ) 
plt.ylim(-65,65)

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    # labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
# ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

ax = plt.subplot(2,5,9)
# LL = ['orangered']
ax = sns.pointplot(x="Time", y="RoC", data=FR_group,hue="Fish_ID",ci=68,dodge=True,color='#00ac87',scale=0.65)
ax.axvspan(-1, -0.1, alpha=0.3, color='khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='khaki')
ax.axvspan(3.2, 3.9, alpha=0.3, color='khaki')
ax.axvspan(5.2, 5.9, alpha=0.3, color='khaki')
plt.xlim(-0.5,5.5 ) 
plt.ylim(-65,65)

ax.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    # bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    # labelbottom=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
# ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()

#%%
average = pd.read_excel("Ext Figure4.xlsx",sheet_name='ext4de')


#%%
ax = plt.subplot(2,5,5)
ax = sns.pointplot(x="time", y="av_Puncta", data=average,hue="Condition",scale=0.65,
                   ci=68,dodge=True,palette=palette,join=True,errwidth=2.5)
plt.xlim(-0.5,1.5 )  
ax.axvspan(0.16, 0.92, alpha=0.15, color='grey') 
ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off

ax.set_ylabel('')    
ax.set_xlabel('')    
ax.get_legend().remove()
sns.despine()


ax = plt.subplot(2,5,10)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="time", y="av_RoC", data=average,hue="Condition",scale=0.65,errwidth=2.5,
                   ci=68,dodge=True,palette=palette,join=False)
plt.xlim(-0.5,1.5 )  
ax.axvspan(0.16, 0.92, alpha=0.15, color='grey') 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

# ADDED: Remove labels.
# ax.set_xlabel('')
ax.set_ylabel('')    
ax.get_legend().remove()
sns.despine()
fig.tight_layout()