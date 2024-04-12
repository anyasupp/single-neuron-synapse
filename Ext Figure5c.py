# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 17:18:42 2023

@author: Anya
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataframe = pd.read_excel("Ext Figure5.xlsx",sheet_name='c') 
palette = ['lightseagreen','orangered', 'dodgerblue','gold']

dataframe['PA_loc'] = dataframe['PA_loc'].replace({0:np.nan})
#%% Figure
sns.set_style("ticks")
sns.set_context("talk")
fig, ax = plt.subplots(2, 3, sharex=True,figsize=(8,7))
fig.subplots_adjust(wspace=0.5)
fig.suptitle('Cluster morphology')

ax = plt.subplot(2,3,1)
ax= sns.boxplot(y='Filament_Length_Sum',data=dataframe,x='Segment K-means PCA',palette=palette)
ax= sns.stripplot(y='Filament_Length_Sum',data=dataframe,x='Segment K-means PCA',color='.3')
ax.set_xlabel('')
ax.set_ylabel('Filament Length Sum (μm)')
# ax.set_xticks([])
sns.despine()
# plt.ylim(275,1500)

ax = plt.subplot(2,3,2)
ax= sns.boxplot(y='AP_span',data=dataframe,x='Segment K-means PCA',palette=palette)
ax= sns.stripplot(y='AP_span',data=dataframe,x='Segment K-means PCA',color='.3')

ax.set_xlabel('')
ax.set_ylabel('A-P Span (μm)')
# ax.set_xticks([])
sns.despine()
# plt.ylim(15,80)

ax = plt.subplot(2,3,3)
ax= sns.boxplot(y='Distance_Skin_new',data=dataframe,x='Segment K-means PCA',palette=palette)
ax= sns.swarmplot(y='Distance_Skin_new',data=dataframe,x='Segment K-means PCA',color='.3')

ax.set_xlabel('')
ax.set_ylabel('Distance from Skin (μm)')
# ax.set_xticks([])
sns.despine()
# plt.ylim(0,23)

ax = plt.subplot(2,3,4)
ax= sns.boxplot(y='New_arbour_thickness',data=dataframe,x='Segment K-means PCA',palette=palette)
ax= sns.stripplot(y='New_arbour_thickness',data=dataframe,x='Segment K-means PCA',color='.3')

ax.set_xlabel('')
ax.set_ylabel('Distal Arbour Thickness (μm)')
# plt.ylim(4,20)
sns.despine()
# ax.legend(title = 'Trend Cluster',loc='center left', bbox_to_anchor=(0.85,1))


ax = plt.subplot(2,3,5)
ax= sns.boxplot(y='Darbour_loc',data=dataframe,x='Segment K-means PCA',palette=palette)
ax= sns.stripplot(y='Darbour_loc',data=dataframe,x='Segment K-means PCA',color='.3')
sns.despine()
ax.set_xlabel('')
ax.set_ylabel('Distal Arbour Location')
# plt.ylim(0.75,0.93)
ax = plt.subplot(2,3,6)
ax= sns.boxplot(y='PA_loc',data=dataframe,x='Segment K-means PCA',palette=palette)
ax= sns.stripplot(y='PA_loc',data=dataframe,x='Segment K-means PCA',color='.3')
sns.despine()
ax.set_xlabel('')
plt.text(0, 0.4, 'Ø',horizontalalignment='center',verticalalignment='center')
# plt.ylim(0.3,0.9)
ax.set_ylabel('Proximal Arbour Location')

