# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:42:26 2023

@author: Anya
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


data_combined = pd.read_excel("Ext Figure1.xlsx",sheet_name='1e')

#%% extfigure 1e

coloc_data = data_combined[data_combined['coloc'] == True]
##histo
fig,ax = plt.subplots(1,1)
#plt.hist(good_diff,bins=10,histtype='bar',edgecolor='black', linewidth=1.2,    alpha=0.75)
plt.title("Distance between FingR.PSD95 and anti-MAGUK maxima - Coloc", fontsize = 20)
plt.xlabel("Distance between maxima (μm)")
plt.ylabel("Count")
# fig3.savefig('20200524_GPHN_spot_diff.jpg', format='jpg', dpi=1200)
sns.set_context("talk")
sns.distplot(coloc_data.spotsdiff, bins=10,kde=False,color='skyblue')
sns.despine()
#%% Extended figure fg
fig,ax = plt.subplots(1,1)
coloc_data = data_combined[data_combined['coloc'] == True]
non_coloc_data= data_combined[data_combined['coloc'] == False]
ax= sns.regplot(data=coloc_data,y='Delta_MGUK',x='Delta_GFP')

sns.regplot(data=non_coloc_data,y='Delta_MGUK',x='Delta_GFP')

x = coloc_data[['Delta_GFP','Delta_MGUK']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope, intercept, r_value, p_value, std_err = stats.linregress(x[:,0],x[:,1])

plt.annotate('r²:{:.3f}'.format(r_value**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
plt.annotate('p={:.3f}'.format(p_value),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
plt.annotate('R={:.3f}'.format(r_value),xy=(1,0.1),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
#%% 
fig,ax = plt.subplots(1,1)
coloc_data = data_combined[data_combined['coloc'] == True]
non_coloc_data= data_combined[data_combined['coloc'] == False]
ax= sns.regplot(data=coloc_data,y='FWHM_MGUK',x='FWHM_GFP')

sns.regplot(data=non_coloc_data,y='FWHM_MGUK',x='FWHM_GFP')

x = coloc_data[['FWHM_GFP','FWHM_MGUK']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope, intercept, r_value, p_value, std_err = stats.linregress(x[:,0],x[:,1])

plt.annotate('r²:{:.3f}'.format(r_value**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
plt.annotate('p={:.3f}'.format(p_value),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
plt.annotate('R={:.3f}'.format(r_value),xy=(1,0.1),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
#%% ext fig d
rotate_df = pd.read_excel("Ext Figure1.xlsx",sheet_name='rotate_controls')

total = len(data_combined)
positive=sum(data_combined.coloc)
percentgood = (positive/total)*100
percentbad = ((total-positive)/total)*100

total_ro = len(rotate_df.coloc)
positive_ro=sum(rotate_df.coloc)
percentgood_ro = (positive_ro/total_ro)*100
percentbad_ro = ((total_ro-positive_ro)/total_ro)*100

table = np.array([[positive, positive_ro],[(total-positive), (total_ro-positive_ro)]])
from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(table, correction=True)
print(f"Chi2 result of the contingency table: {chi2}, p-value: {p}")

#
fig,ax = plt.subplots(1,2)
# Data
r = [1]
# plot
barWidth = 0.85
# Create green Bars
ax = plt.subplot(1,2,1)
ax.bar(r, percentgood, color='skyblue', edgecolor='white', width=barWidth)
ax.bar(r, percentbad, bottom=percentgood, color='lightcoral', edgecolor='white', width=barWidth)


plt.ylim(0,100)
plt.axis('off')
# Show graphic
plt.show()
ax = plt.subplot(1,2,2)
ax.bar(r, percentgood_ro, color='skyblue', edgecolor='white', width=barWidth)
ax.bar(r, percentbad_ro, bottom=percentgood_ro, color='lightcoral', edgecolor='white', width=barWidth)

plt.ylim(0,100)
plt.axis('off')
#%% ext fig 1h
reverse_overlaid = pd.read_excel("Ext Figure1.xlsx",sheet_name='reverse_overlaid')
reverse_rotated = pd.read_excel("Ext Figure1.xlsx",sheet_name='reverse_rotated')

total = len(reverse_overlaid)
positive=sum(reverse_overlaid.coloc)
percentgood = (positive/total)*100
percentbad = ((total-positive)/total)*100

total_ro = len(reverse_rotated)
positive_ro=sum(reverse_rotated.coloc)
percentgood_ro = (positive_ro/total_ro)*100
percentbad_ro = ((total_ro-positive_ro)/total_ro)*100

table = np.array([[positive, positive_ro],[(total-positive), (total_ro-positive_ro)]])
from scipy.stats import chi2_contingency
chi2, p, dof, ex = chi2_contingency(table, correction=True)
print(f"Chi2 result of the contingency table: {chi2}, p-value: {p}")
#
fig,ax = plt.subplots(1,2)
# Data
r = [1]
# plot
barWidth = 0.85
# Create green Bars
ax = plt.subplot(1,2,1)
ax.bar(r, percentgood, color='skyblue', edgecolor='white', width=barWidth)
ax.bar(r, percentbad, bottom=percentgood, color='lightcoral', edgecolor='white', width=barWidth)


plt.ylim(0,100)
plt.axis('off')
# Show graphic
plt.show()
ax = plt.subplot(1,2,2)
ax.bar(r, percentgood_ro, color='skyblue', edgecolor='white', width=barWidth)
ax.bar(r, percentbad_ro, bottom=percentgood_ro, color='lightcoral', edgecolor='white', width=barWidth)

plt.ylim(0,100)
plt.axis('off')