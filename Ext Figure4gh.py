# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:07:55 2023

@author: Anya
"""

# -*- coding: utf-8 -*-
"""
Created on Mon May  3 13:48:38 2021

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%% load data 
data = pd.read_excel("Ext Figure4.xlsx",sheet_name='ext4gh')
#%%
gfp_normalized = (data.mean_int-data.min_stack)/(data.max_stack-data.min_stack)
mkate_normalized = (data.mkate_mean-data.min_stack)/(data.mkate_max-data.min_stack)
synapse_int = gfp_normalized/mkate_normalized

data.insert(10,'normalized_GFP',gfp_normalized)
data.insert(11,'normalized_mKate',mkate_normalized)
data.insert(11,'synapse_int',synapse_int)
#%% take out laser different samples check notebook
puncta_int_ratio = data[data.Fish_ID!='F4_0804_R']



time=puncta_int_ratio.groupby('Time')

#make new dataframe with cyan,green,red with correct fish ID
start = [time.get_group('dpf7_0')]
start = pd.concat(start)
end = [time.get_group('dpf9_10')]
end = pd.concat(end)

start = start.set_index('Fish_ID')
end = end.set_index('Fish_ID')
#%%
int_roc = ((end.synapse_int - start.synapse_int)/start.synapse_int)*100

end.insert(8,'int_roc',int_roc)

end=end.reset_index()

#%%
rows = 2
cols = 2
sns.set_style('ticks')
sns.set_context('talk')
fig, ax = plt.subplots(rows,cols)
size=12
x1,y1 = [-0.5,2],[0,0]
#%%
sns.set_context('paper')#
plt.subplot(rows,cols,1)
ax = sns.pointplot(x='Time',y='Puncta',hue='Condition',data=puncta_int_ratio,palette='Set2',dodge=True)
ax.get_legend().remove()
ax.set_xlabel('')
# ax.set_xticks([])
sns.despine()
ax.tick_params(axis='both', which='major', labelsize=size)

plt.subplot(rows,cols,2)
ax = sns.pointplot(x='Time',y='RoC',hue='Condition',data=end,palette='Set2',dodge=0.1)
ax.get_legend().remove()
ax.set_xlabel('')
plt.plot(x1,y1,'k--',alpha=0.65)

ax.set_ylabel('')
plt.xlim(-0.5,0.5 )
sns.despine()
ax.tick_params(axis='both', which='major', labelsize=size)

plt.subplot(rows,cols,3)
ax = sns.pointplot(x='Time',y='synapse_int',hue='Condition',
                   data=puncta_int_ratio,palette='Set2',dodge=True)
ax.get_legend().remove()
ax.tick_params(axis='both', which='major', labelsize=size)

sns.despine()

plt.subplot(rows,cols,4)

plt.plot(x1,y1,'k--',alpha=0.65)
ax.tick_params(axis='both', which='major', labelsize=size)
ax = sns.pointplot(x='Time',y='int_roc',hue='Condition',data=end,palette='Set2',dodge=0.1)
ax.get_legend().remove()
ax.set_ylabel('')
sns.despine()
ax.tick_params(axis='both', which='major', labelsize=size)
