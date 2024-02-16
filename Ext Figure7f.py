# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:08:55 2023

@author: Anya
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_excel("Ext Figure7_part2.xlsx",sheet_name='ext7e1')

puncta = pd.read_excel("Ext Figure7_part2.xlsx",sheet_name='ext7f')

#%%
arbors= data.drop(['stem','extra_branch'],axis=1)
arbors.insert(5,'p_arbor_dpunc',puncta.proximal_arbour,True)
arbors.insert(5,'d_arbor_dpunc',puncta.distal_arbour,True)

time=arbors.groupby('Time')

#make new dataframe with cyan,green,red with correct fish ID
evening = [time.get_group('dpf7_10')]
evening = pd.concat(evening)

morn = [time.get_group('dpf8_0')]
morn = pd.concat(morn)

#%%make scatter plot to see diff segment within fish
from scipy import stats

def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=12)
    
row = 2
column = 2
fig, ax = plt.subplots(row,column, sharex=False)
fig.suptitle('correlation within fish of arbours')
# fig.tight_layout()

ax = plt.subplot(row,column,1)
ax = sns.regplot(data=evening,y='proximal_arbour',x='distal_arbour',fit_reg=True,color='#ff4300')
plt.xlabel('distal_arbour_roc',fontsize=10)
plt.ylabel('proximal_arbour_roc',fontsize=10)
x = evening[['proximal_arbour','distal_arbour']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
# ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom')
# ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom',fontsize=9)
# ax.annotate('r_pearson:{:.3f}'.format(r_value_a_day),xy=(1,0),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom', fontsize=9)
simpleaxis(ax)
sns.despine()

ax = plt.subplot(row,column,2)
ax = sns.regplot(data=morn,y='proximal_arbour',x='distal_arbour',fit_reg=True,color='#ff4300')
ax.set_facecolor(color='lightgray')
plt.xlabel('distal_arbour_roc',fontsize=10)
plt.ylabel('proximal_arbour_roc',fontsize=10)
x = morn[['proximal_arbour','distal_arbour']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
# ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom')
# ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom',fontsize=9)
# ax.annotate('r_pearson:{:.3f}'.format(r_value_a_day),xy=(1,0),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom', fontsize=9)
simpleaxis(ax)
sns.despine()


ax = plt.subplot(row,column,3)
ax = sns.regplot(data=evening,y='p_arbor_dpunc',x='d_arbor_dpunc',fit_reg=True,color='#ff4300')
plt.xlabel('distal_arbour_Δpuncta',fontsize=10)
plt.ylabel('proximal_arbour_Δpuncta',fontsize=10)
x = evening[['p_arbor_dpunc','d_arbor_dpunc']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
# ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom')
# ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom',fontsize=9)
# ax.annotate('r_pearson:{:.3f}'.format(r_value_a_day),xy=(1,0),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom', fontsize=9)
simpleaxis(ax)
sns.despine()

ax = plt.subplot(row,column,4)
ax = sns.regplot(data=morn,y='p_arbor_dpunc',x='d_arbor_dpunc',fit_reg=True,color='#ff4300')
x = morn[['p_arbor_dpunc','d_arbor_dpunc']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
# ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
# #             horizontalalignment='right',verticalalignment='bottom')
# ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom',fontsize=9)
# ax.annotate('r_pearson:{:.3f}'.format(r_value_a_day),xy=(1,0),xycoords='axes fraction',
#             horizontalalignment='right',verticalalignment='bottom', fontsize=9)
simpleaxis(ax)
ax.set_facecolor(color='lightgray')
plt.xlabel('distal_arbour_Δpuncta',fontsize=10)
plt.ylabel('proximal_arbour_Δpuncta',fontsize=10)
sns.despine()