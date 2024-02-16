# -*- coding: utf-8 -*-
"""
Created on Sat Oct  8 11:45:14 2022

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import linear_model, datasets
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

puncta = pd.read_excel("Figure3.xlsx",sheet_name='3c_firstsleep')
control_latter = pd.read_excel("Figure3.xlsx",sheet_name='3c_lattersleep')
#%%
control = puncta[puncta.Condition!='SD']
sd = puncta[puncta.Condition!='Control']

control_latter = control_latter[control_latter.Condition!='SD']
col_later ='#0088e8'

#group into cluster 
groups=puncta.groupby('Cluster')

cluster_1 = [groups.get_group(0)]
cluster_1 = pd.concat(cluster_1)
cluster_2 = [groups.get_group(1)]
cluster_2 = pd.concat(cluster_2)
cluster_3 = [groups.get_group(2)]
cluster_3 = pd.concat(cluster_3)
cluster_4 = [groups.get_group(3)]
cluster_4 = pd.concat(cluster_4)

groups=control_latter.groupby('Cluster')

c1 = [groups.get_group(0)]
c1 = pd.concat(c1)
c2 = [groups.get_group(1)]
c2 = pd.concat(c2)
c3 = [groups.get_group(2)]
c3 = pd.concat(c3)
c4 = [groups.get_group(3)]
c4 = pd.concat(c4)

#now take out all of morphology type one and see the correlation? 
puncta = puncta[puncta.Cluster!=0]
control_latter = control_latter[control_latter.Cluster!=0]

control = puncta[puncta.Condition!='SD']
sd = puncta[puncta.Condition!='Control']

control_latter = control_latter[control_latter.Condition!='SD']
col_later ='#4393c3'

#take out cluster 0 type 1 from average data. 
early_sleep_av = control.sleepph_adjusted_exp.mean()
early_sleep_sem = control.sleepph_adjusted_exp.sem()
#insanity check 
#print(control.sleepph_adjusted_exp.std()/np.sqrt(len(control.sleepph_adjusted_exp)))
early_punc_av = control.punc_hr4h.mean()
early_punc_sem = control.punc_hr4h.sem()

early_bout_av = control.bl_4h_av.mean()
early_bout_sem = control.bl_4h_av.sem()

#for control latter
late_sleep_av = control_latter.late_sleepph_adjusted_exp.mean()
late_sleep_sem = control_latter.late_sleepph_adjusted_exp.sem()

late_punc_av = control_latter.punc_ZT18_0.mean()
late_punc_sem = control_latter.punc_ZT18_0.sem()

late_bout_av = control_latter.bl_late_av.mean()
late_bout_sem = control_latter.bl_late_av.sem()

#for SD
sd_sleep_av = sd.sleepph_adjusted_exp.mean()
sd_sleep_sem = sd.sleepph_adjusted_exp.sem()

sd_punc_av = sd.punc_hr4h.mean()
sd_punc_sem = sd.punc_hr4h.sem()

sd_bout_av = sd.bl_4h_av.mean()
sd_bout_sem = sd.bl_4h_av.sem()

#%%
def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=12)
    # ax.tick_params(axis='x',labelsize=10)
    
# simpleaxis(ax)

row = 2
column = 4
fig, ax = plt.subplots(row,column, sharex=False)
fig.suptitle('sleep min per hour /average boutlength per hour vs synapse per hour excl Type 1', fontsize=16)

ax = plt.subplot(row,column,1)
ax = sns.regplot(data=control,y='sleepph_adjusted_exp',x='punc_hr4h',fit_reg=True)
plt.xlim(-7,7)
# plt.ylim(-20,50)
x = control[['sleepph_adjusted_exp','punc_hr4h']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
plt.ylim(0,50)
ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
sns.despine()


ax = plt.subplot(row,column,2)
ax = sns.regplot(data=sd,y='sleepph_adjusted_exp',x='punc_hr4h',fit_reg=True,color='#ff7f0e')
plt.xlim(-7,7)
plt.ylim(0,50)
x = sd[['sleepph_adjusted_exp','punc_hr4h']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
sns.despine()

ax = plt.subplot(row,column,3)
ax = sns.regplot(data=control_latter,y='late_sleepph_adjusted_exp',x='punc_ZT18_0',fit_reg=True, color=col_later)
x = control_latter[['late_sleepph_adjusted_exp','punc_ZT18_0']].to_numpy()
plt.xlim(-7,7)
plt.ylim(0,50)
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])
ax.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
ax.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)

sns.despine()

ax = plt.subplot(row,column,4)
plt.plot(early_punc_av,early_sleep_av,'o',color='#1f77b4',label='early',markersize=5)
plt.errorbar(early_punc_av,early_sleep_av,xerr=early_punc_sem,yerr=early_sleep_sem,color='#1f77b4')

plt.plot(sd_punc_av,sd_sleep_av,'o',color='#ff7f0e',label='SD',markersize=5)
plt.errorbar(sd_punc_av,sd_sleep_av,xerr=sd_punc_sem,yerr=sd_sleep_sem,color='#ff7f0e')

plt.plot(late_punc_av,late_sleep_av,'o',color=col_later,label='late', markersize=5)
plt.errorbar(late_punc_av,late_sleep_av,xerr=late_punc_sem,yerr=late_sleep_sem,color=col_later)
plt.xlim(-3,2)
plt.ylim(0,40)
plt.xlabel('puncta per hour')
plt.ylabel('sleep per hour adjusted')
# plt.legend()
sns.despine()


ax = plt.subplot(row,column,5)
ax = sns.regplot(data=control,y='bl_4h_av',x='punc_hr4h',fit_reg=True)
plt.xlim(-7,7)
plt.ylim(-0.5,17.5)

sns.despine()
simpleaxis(ax)

ax = plt.subplot(row,column,6)
ax = sns.regplot(data=sd,y='bl_4h_av',x='punc_hr4h',fit_reg=True,color='#ff7f0e')
plt.xlim(-7,7)
plt.ylim(-0.5,17.5)

sns.despine()
simpleaxis(ax)

ax = plt.subplot(row,column,7)
ax = sns.regplot(data=control_latter,y='bl_late_av',x='punc_ZT18_0',fit_reg=True,color=col_later)
x = control_latter[['bl_late_av','punc_ZT18_0']].to_numpy()
plt.xlim(-7,7)
plt.ylim(-0.5,17.5)

sns.despine()
simpleaxis(ax)

ax = plt.subplot(row,column,8)
plt.plot(early_punc_av,early_bout_av,'o',color='#1f77b4',label='early_exclude', markersize=5)
plt.errorbar(early_punc_av,early_bout_av,xerr=early_punc_sem,yerr=early_bout_sem,color='#1f77b4')

plt.plot(sd_punc_av,sd_bout_av,'o',color='#ff7f0e',label='SD', markersize=5)
plt.errorbar(sd_punc_av,sd_bout_av,xerr=sd_punc_sem,yerr=sd_bout_sem,color='#ff7f0e')

plt.plot(late_punc_av,late_bout_av,'o',color=col_later,label='late', markersize=5)
plt.errorbar(late_punc_av,late_bout_av,xerr=late_punc_sem,yerr=late_bout_sem,color=col_later)
simpleaxis(ax)

plt.xlim(-3,1)
plt.ylim(2.5,7)

plt.xlabel('puncta per hour')
plt.ylabel('average boutlength')
# plt.legend()
sns.despine()
