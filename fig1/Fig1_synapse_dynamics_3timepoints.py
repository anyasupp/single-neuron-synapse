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
data_combined_3T = pd.read_excel("20220630_mother_spreadsheet.xlsx",sheet_name='puncta_3T')

##make new dataframe with cyan,green,red with correct fish ID
groups=data_combined_3T.groupby('Condition')

LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)


FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)

data_combined_3T_no7dpf =  data_combined_3T[data_combined_3T.Time !='dpf7_0']
LD_group_no7= LD_group[LD_group.Time !='dpf7_0']
LL_group_no7= LL_group[LL_group.Time !='dpf7_0']
FR_group_no7= FR_group[FR_group.Time !='dpf7_0']

palette= ['#5ec0eb','#dd90b5','#00ac87']

#baseline at zero
x1,y1 = [-0.5,2],[0,0]
#font size
size =10
#%% Figure set up
sns.set_context('paper')#

row = 4
column = 4
fig, ax = plt.subplots(row,column, sharex=True)
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

ax.get_legend().remove()

sns.despine()

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
ax.get_legend().remove()

sns.despine()

ax.set_ylabel('')    
ax.set_xlabel('')

#Puncta LL
ax = plt.subplot(row,column,3)
ax = sns.pointplot(x="Time", y="Puncta", data=LL_group,hue="Fish_ID",ci=68,dodge=0.1,color='#dd90b5',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)

ax.axvspan(-1, -0.1, alpha=0.3, color='Khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='Khaki')

plt.xlim(-0.5,2.75 ) 
plt.ylim(30,330) 

ax.get_legend().remove()

sns.despine()

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
ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('')    
ax.set_xlabel('')

#ROC all
ax = plt.subplot(row,column,5)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="RoC", data=data_combined_3T_no7dpf,hue="Condition",
                   ci=68,dodge=0.1,palette=palette,join=False,scale=0.75)

ax.axvspan(0.16, 0.92, alpha=0.2, color='grey') 
ax.tick_params(axis='both', which='major', labelsize=size)
plt.xlim(-0.5,1.5 )  
plt.ylim(-10,25)

ax.get_legend().remove()

sns.despine()
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

sns.despine()
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

sns.despine()
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
sns.despine()
ax.set_ylabel('')    
ax.set_xlabel('')


#%% intensity
data_combined_3T = pd.read_excel("20220630_mother_spreadsheet.xlsx",sheet_name='int_ratio_3T')

groups=data_combined_3T.groupby('Condition')

LD_group = [groups.get_group('LD')]
LD_group = pd.concat(LD_group)

LL_group = [groups.get_group('LL')]
LL_group = pd.concat(LL_group)


FR_group = [groups.get_group('FR')]
FR_group = pd.concat(FR_group)
#%%
data_combined_3T_no7dpf =  data_combined_3T[data_combined_3T.Time !='7dpf_0']
LD_group_no7= LD_group[LD_group.Time !='7dpf_0']
LL_group_no7= LL_group[LL_group.Time !='7dpf_0']
FR_group_no7= FR_group[FR_group.Time !='7dpf_0']

palette= ['#5ec0eb','#dd90b5','#00ac87']

#baseline at zero
x1,y1 = [-0.5,2],[0,0]

#%% Figure set up
ax = plt.subplot(row,column,9)
ax = sns.pointplot(x="Time", y="int_ratio", data=data_combined_3T,hue="Condition",ci=68,dodge=True, 
                   palette = palette,linestyles='--',scale=0.75)

ax.tick_params(axis='both', which='major', labelsize=size)
ax.axvspan(-1, -0.1, alpha=0.2, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.2, color='grey')

plt.xlim(-0.25,2.75 ) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.get_legend().remove()

sns.despine()

ax.set_ylabel('Puncta Count')    
ax.set_xlabel('')

ax = plt.subplot(row,column,10)
ax = sns.pointplot(x="Time", y="int_ratio", data=LD_group,hue="Fish_ID",scale=0.75,ci=68,dodge=False,color='#5ec0eb',linestyles='--')

ax.tick_params(axis='both', which='major', labelsize=size)
ax.axvspan(-1, -0.1, alpha=0.55, color='grey')
ax.axvspan(1.2, 1.9, alpha=0.55, color='grey')

plt.xlim(-0.5,2.75 ) 
plt.ylim(0.1,0.8) 
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.get_legend().remove()

sns.despine()
ax.set_ylabel('')    
ax.set_xlabel('')

#Puncta LL
ax = plt.subplot(row,column,11)
ax = sns.pointplot(x="Time", y="int_ratio", data=LL_group,hue="Fish_ID",ci=68,dodge=False,color='#dd90b5',
                   linestyles='--',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)

ax.axvspan(-1, -0.1, alpha=0.3, color='Khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='Khaki')

plt.xlim(-0.5,2.75 ) 
plt.ylim(0.1,0.8) 
ax.get_legend().remove()

sns.despine()
ax.set_ylabel('')    
ax.set_xlabel('')

# Puncta FR
ax = plt.subplot(row,column,12)
ax = sns.pointplot(x="Time", y="int_ratio", data=FR_group,hue="Fish_ID",ci=68,dodge=False,color='#00ac87',
                   linestyles='--',scale=0.75)
ax.tick_params(axis='both', which='major', labelsize=size)

ax.axvspan(-1, -0.1, alpha=0.3, color='Khaki')
ax.axvspan(1.2, 1.9, alpha=0.3, color='Khaki')

plt.xlim(-0.5,2.75 ) 
plt.ylim(0.1,0.8) 

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

ax.get_legend().remove()

sns.despine()

ax.set_ylabel('')    
ax.set_xlabel('')

#ROC all
ax = plt.subplot(row,column,13)
plt.plot(x1,y1,'k--',alpha=0.65)
ax = sns.pointplot(x="Time", y="ratio_roc", data=data_combined_3T_no7dpf,hue="Condition",ci=68,
                   dodge=0.1,palette=palette,join=False,scale=0.75)

ax.axvspan(0.16, 0.92, alpha=0.2, color='grey') 
ax.tick_params(axis='both', which='major', labelsize=size)
plt.xlim(-0.5,1.5 )  

ax.get_legend().remove()

sns.despine()
# ADDED: Remove labels.
ax.set_ylabel('RoC (%)')    
ax.set_xlabel('')

ax = plt.subplot(row,column,14)
ax = sns.pointplot(x="Time", y="ratio_roc", data=LD_group_no7,hue="Fish_ID",scale=0.75,ci=68,dodge=False, color = '#5ec0eb',linestyles='--')
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
ax = sns.pointplot(x="Time", y="ratio_roc", data=LL_group_no7,hue="Fish_ID",scale=0.75,ci=68,dodge=False, color = '#dd90b5',linestyles='--')
ax.tick_params(axis='both', which='major', labelsize=size)  
plt.xlim(-0.5,1.5 )  
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki') 
plt.ylim(-70,210) 

ax.get_legend().remove()
sns.despine()

ax.set_ylabel('')    
ax.set_xlabel('')


#ROC FR
ax = plt.subplot(row,column,16)
ax = sns.pointplot(x="Time", y="ratio_roc", data=FR_group_no7,hue="Fish_ID",scale=0.75,ci=68,dodge=False, color = '#00ac87',linestyles='--')
ax.tick_params(axis='both', which='major', labelsize=size)
plt.ylim(-70,210) 
plt.xlim(-0.5,1.5 )
ax.axvspan(0.16, 0.92, alpha=0.3, color='khaki')

ax.get_legend().remove()
sns.despine()

ax.set_ylabel('')    
ax.set_xlabel('')


