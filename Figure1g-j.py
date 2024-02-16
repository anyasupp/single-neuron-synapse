# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 10:50:04 2021

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
#puncta count
data_combined_3T = pd.read_excel("Figure1.xlsx",sheet_name='1gh')
# intensity
data_combined_3T_intensity = pd.read_excel("Figure1.xlsx",sheet_name='1ij')

# %%#make new dataframe with cyan,green,red with correct fish ID
groups=data_combined_3T.groupby('Condition')

LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)


FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)

#intensity
groups_intensity=data_combined_3T_intensity.groupby('Condition')

LD_group_int = [groups_intensity.get_group('LD')]
LD_group_int = pd.concat(LD_group_int)

LL_group_int = [groups_intensity.get_group('LL')]
LL_group_int = pd.concat(LL_group_int)


FR_group_int = [groups_intensity.get_group('FR')]
FR_group_int = pd.concat(FR_group_int)


#for puncta count
data_combined_3T_no7dpf =  data_combined_3T[data_combined_3T.Time !='dpf7_0']
LD_group_no7= LD_group[LD_group.Time !='dpf7_0']
LL_group_no7= LL_group[LL_group.Time !='dpf7_0']
FR_group_no7 = FR_group[FR_group.Time !='dpf7_0']

#for intensity
data_combined_3T_no7dpf_int =  data_combined_3T_intensity[data_combined_3T_intensity.Time !='7dpf_0']
LD_group_no7_int= LD_group_int[LD_group_int.Time !='7dpf_0']
LL_group_no7_int= LL_group_int[LL_group_int.Time !='7dpf_0']
FR_group_no7_int= FR_group_int[FR_group_int.Time !='7dpf_0']


palette= ['#5ec0eb','#dd90b5','#00ac87']

#baseline at zero
x1,y1 = [-0.5,2],[0,0]
#font size
size =10
#%% Figure set up
# paper_rc= {'lines.linewidth': 1.2}
sns.set_context('paper')#
# sns.reset_defaults()
# sns.set(style='ticks',rc=paper_rc)

row = 4
column = 4
fig, ax = plt.subplots(row,column, sharex=False)
fig.suptitle('FingR.PSD95 dynamics ', fontsize=16)
fig.set_size_inches(9,9)
fig.tight_layout()

ax = plt.subplot(row,column,1)
ax = sns.pointplot(x="time", y="Puncta", data=data_combined_3T,hue="Condition",
                   ci=68,dodge=True, palette = palette,scale=0.75)

ax.tick_params(axis='both', which='major', labelsize=size)
ax.axvspan(-1, -0.1, alpha=0.2, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.2, color='grey')

plt.xlim(-0.25,2.75 ) 
plt.ylim(95,165) 

# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()

# plt.legend(bbox_to_anchor=(1,1), loc='upper left')

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('Puncta Count')    
ax.set_xlabel('')

# Puncta LD
ax = plt.subplot(row,column,2)
ax = sns.pointplot(x="Time", y="Puncta", data=LD_group,hue="Fish_ID",ci=68,dodge=0.1,color='#5ec0eb',scale=0.75)

ax.tick_params(axis='both', which='major', labelsize=size)
ax.axvspan(-1, -0.1, alpha=0.55, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.55, color='grey')

plt.xlim(-0.5,2.75 ) 
plt.ylim(30,330) 

# ax.tick_params(
#     axis='x',          # changes apply to the x-axis
#     which='both',      # both major and minor ticks are affected
#     bottom=False,      # ticks along the bottom edge are off
#     top=False,         # ticks along the top edge are off
#     labelbottom=False) # labels along the bottom edge are off
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
#
#Puncta LL
ax = plt.subplot(row,column,3)
ax = sns.pointplot(x="Time", y="Puncta", data=LL_group,hue="Fish_ID",ci=68,dodge=0.1,color='#dd90b5',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)

ax.axvspan(-1, -0.1, alpha=0.3, color='Khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='Khaki')

plt.xlim(-0.5,2.75 ) 
plt.ylim(30,330) 


# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
# Puncta FR
ax = plt.subplot(row,column,4)
ax = sns.pointplot(x="Time", y="Puncta", data=FR_group,hue="Fish_ID",ci=68,dodge=0.1,color='#00ac87',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)

ax.axvspan(-1, -0.1, alpha=0.3, color='Khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='Khaki')

plt.xlim(-0.5,2.75 ) 
plt.ylim(30,330) 

# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
#
#ROC all
ax = plt.subplot(row,column,5)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=data_combined_3T_no7dpf,hue="Condition",
                   ci=68,dodge=0.1,palette=palette,join=False,scale=0.75)
#dodge=0.3, join=False
ax.axvspan(0.16, 0.92, alpha=0.2, color='grey') 
ax.tick_params(axis='both', which='major', labelsize=size)
plt.xlim(-0.5,1.5 )  
plt.ylim(-10,25)

ax.get_legend().remove()
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('RoC (%)')    
ax.set_xlabel('')

#ROC LD
ax = plt.subplot(row,column,6)
ax = sns.pointplot(x="Time", y="RoC", data=LD_group_no7,hue="Fish_ID",ci=68,dodge=False, color = '#5ec0eb',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)
plt.xlim(-0.5,1.5 )  
ax.axvspan(0.16, 0.92, alpha=0.55, color='grey') 
plt.ylim(-50,90) 

ax.get_legend().remove()
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
#ROC LL
ax = plt.subplot(row,column,7)
ax = sns.pointplot(x="Time", y="RoC", data=LL_group_no7,hue="Fish_ID",ci=68,dodge=False, color = '#dd90b5',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)  
plt.xlim(-0.5,1.5 )  
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.ylim(-50,90) 


ax.get_legend().remove()
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')


#ROC FR
ax = plt.subplot(row,column,8)
ax = sns.pointplot(x="Time", y="RoC", data=FR_group_no7,hue="Fish_ID",ci=68,dodge=False, color = '#00ac87',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)

plt.xlim(-0.5,1.5 )
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki')
plt.ylim(-50,90) 


ax.get_legend().remove()
# ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')

palette= ['#5ec0eb','#dd90b5','#00ac87']

#baseline at zero
x1,y1 = [-0.5,2],[0,0]

#
ax = plt.subplot(row,column,9)
ax = sns.pointplot(x="Time", y="int_ratio", data=data_combined_3T_intensity,hue="Condition",ci=68,dodge=True, 
                   palette = palette,linestyles='--',scale=0.75)

ax.tick_params(axis='both', which='major', labelsize=size)
ax.axvspan(-1, -0.1, alpha=0.2, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.2, color='grey')

plt.xlim(-0.25,2.75 ) 
# plt.ylim(100,175) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()



sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('Puncta Count')    
ax.set_xlabel('')
#

ax = plt.subplot(row,column,10)
ax = sns.pointplot(x="Time", y="int_ratio", data=LD_group_int,hue="Fish_ID",scale=0.75,ci=68,dodge=False,color='#5ec0eb',linestyles='--')

ax.tick_params(axis='both', which='major', labelsize=size)
ax.axvspan(-1, -0.1, alpha=0.55, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.55, color='grey')

plt.xlim(-0.5,2.75 ) 
plt.ylim(0.1,0.8) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
#
#Puncta LL
ax = plt.subplot(row,column,11)
ax = sns.pointplot(x="Time", y="int_ratio", data=LL_group_int,hue="Fish_ID",ci=68,dodge=False,color='#dd90b5',
                   linestyles='--',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)

ax.axvspan(-1, -0.1, alpha=0.3, color='Khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='Khaki')

plt.xlim(-0.5,2.75 ) 
plt.ylim(0.1,0.8) 

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
# Puncta FR
ax = plt.subplot(row,column,12)
ax = sns.pointplot(x="Time", y="int_ratio", data=FR_group_int,hue="Fish_ID",ci=68,dodge=False,color='#00ac87',
                   linestyles='--',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)

ax.axvspan(-1, -0.1, alpha=0.3, color='Khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='Khaki')

plt.xlim(-0.5,2.75 ) 
plt.ylim(0.1,0.8) 
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')

#ROC all
ax = plt.subplot(row,column,13)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="ratio_roc", data=data_combined_3T_no7dpf_int,hue="Condition",ci=68,
                   dodge=0.1,palette=palette,join=False,scale=0.75)
#dodge=0.3, join=False
ax.axvspan(0.16, 0.92, alpha=0.2, color='grey') 
ax.tick_params(axis='both', which='major', labelsize=size)
plt.xlim(-0.5,1.5 )  

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('RoC (%)')    
ax.set_xlabel('')

ax = plt.subplot(row,column,14)
ax = sns.pointplot(x="Time", y="ratio_roc", data=LD_group_no7_int,hue="Fish_ID",scale=0.75,ci=68,dodge=False, color = '#5ec0eb',linestyles='--')
ax.tick_params(axis='both', which='major', labelsize=size)
plt.xlim(-0.5,1.5 )  
plt.ylim(-70,210) 
ax.axvspan(0.16, 0.92, alpha=0.55, color='grey') 
ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')
#ROC LL
ax = plt.subplot(row,column,15)
ax = sns.pointplot(x="Time", y="ratio_roc", data=LL_group_no7_int,hue="Fish_ID",scale=0.75,ci=68,dodge=False, color = '#dd90b5',linestyles='--')
ax.tick_params(axis='both', which='major', labelsize=size)  
plt.xlim(-0.5,1.5 )  
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.ylim(-70,210) 

ax.get_legend().remove()
sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')


#ROC FR
ax = plt.subplot(row,column,16)
ax = sns.pointplot(x="Time", y="ratio_roc", data=FR_group_no7_int,hue="Fish_ID",scale=0.75,ci=68,dodge=False, color = '#00ac87',linestyles='--')
ax.tick_params(axis='both', which='major', labelsize=size)
plt.ylim(-70,210) 
plt.xlim(-0.5,1.5 )
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki')

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')

