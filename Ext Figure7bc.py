# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 15:50:18 2023

@author: Anya
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats
data = pd.read_excel("Ext Figure7bc.xlsx")
#%%
df = data.copy()
df=df[df['Segment Cluster']!=0]
palette = ['lightseagreen','orangered', 'dodgerblue','gold']
#%%
rows = 1
cols = 2
def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=12)
    
fig, ax = plt.subplots(rows, cols, sharex=True,figsize=(8,7))
fig.subplots_adjust(wspace=0.5)
fig.suptitle('Nighttime sleep across subtypes')
# fig.text(0.5,0.04,'Trend Cluster No.',va='center',rotation='horizontal')
palette = ['orangered', 'dodgerblue','gold']

ax = plt.subplot(rows,cols,1)
ax= sns.pointplot(y='sleep_night',data=df,x='Segment Cluster',palette=palette,
                  join=False,errorbar=('ci', 68))
ax= sns.stripplot(y='sleep_night',data=df,x='Segment Cluster',color='.3')
ax.set_xlabel('')
ax.set_ylabel('Average sleep (min/10min)')
# ax.set_xticks([])
sns.despine()
simpleaxis(ax)
ax = plt.subplot(rows,cols,2)
ax= sns.pointplot(y='night_average_boutlength',data=df,x='Segment Cluster',
                  palette=palette,join=False,errorbar=('ci', 68))
ax= sns.stripplot(y='night_average_boutlength',data=df,x='Segment Cluster',color='.3')
ax.set_xlabel('')
ax.set_ylabel('Average sleep bout length (min)')
simpleaxis(ax)
sns.despine()
#%% reg plot
groups=data.groupby('Segment Cluster')

#make new dataframe with cyan,green,red with correct fish ID
c0 = [groups.get_group(0)]
c0 = pd.concat(c0)
c1 = [groups.get_group(1)]
c1 = pd.concat(c1)
c2 = [groups.get_group(2)]
c2 = pd.concat(c2)
c3 = [groups.get_group(3)]
c3 = pd.concat(c3)

data = data[data['Segment Cluster']!=0]

row = 1
column = 4
fig, ax = plt.subplots(row,column, sharex=True)
ax = plt.subplot(row,column,1)
ax = sns.regplot(data=data,x='percentage_change_night',y='sleep_night',color='grey')
plt.ylim(0,9)
plt.xlim(-20,26)
x = data[['percentage_change_night','sleep_night']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9) 
sns.despine()
simpleaxis(ax)
ax = plt.subplot(row,column,2)
ax = sns.regplot(data=c1,x='percentage_change_night',y='sleep_night',color='orangered')
plt.ylim(0,9)
plt.xlim(-20,26) 
plt.ylabel('')
x = c1[['percentage_change_night','sleep_night']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
sns.despine()
simpleaxis(ax)

ax = plt.subplot(row,column,3)
ax = sns.regplot(data=c2,x='percentage_change_night',y='sleep_night',color='dodgerblue')
plt.ylim(0,9)
plt.xlim(-20,26)
plt.ylabel('')
x = c2[['percentage_change_night','sleep_night']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
sns.despine()
simpleaxis(ax)

ax = plt.subplot(row,column,4)
ax = sns.regplot(data=c3,x='percentage_change_night',y='sleep_night',color='gold')
plt.ylim(0,9)
plt.xlim(-20,26)
plt.ylabel('')
x = c3[['percentage_change_night','sleep_night']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9) 
sns.despine()
simpleaxis(ax)

