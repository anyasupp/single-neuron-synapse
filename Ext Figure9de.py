# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:17:51 2023

@author: Anya
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
def act_av(start,end,df):
    '''
    Parameters
    ----------
    def activity_per_hr : TYPE
    use with tenminute activity data 

    Returns
    -------
    None.

    '''
    res=df.iloc[start:end].mean()
    return res


def sleep_per_hr(start,end,df):
    '''
    Parameters
    ----------
    def sleep_per_hr : TYPE
        

    Returns
    -------
    None.

    '''
    time_diff = df.Clock.iloc[end]-df.Clock.iloc[start]
    print(time_diff)
    if time_diff>1:
        res=df.drop(columns='Clock').iloc[start:end].sum()/time_diff
    elif time_diff<1:
        res=df.drop(columns='Clock').iloc[start:end].sum()*time_diff
    # res=selected_time/time_diff
    return res

def sleep_total(start,end,df):
    '''
    Parameters
    ----------
    def sleep_per_hr : TYPE
        

    Returns
    -------
    None.

    '''
    res=df.drop(columns='Clock').iloc[start:end].sum()
    return res

#%%
data2 = pd.read_excel("Ext Figure9.xlsx",sheet_name='d')

#%% 
minute_bins = 10
data_ten = data2.groupby(data2.index // minute_bins).sum()
data_ten['CLOCK']=data_ten['CLOCK']/10
#minus last row
data_ten=data_ten.iloc[0:len(data_ten)-1].copy()
df_ten = data_ten.drop(columns=['start','end'])

df_ten = df_ten.set_index('CLOCK')
# replace 0s with NaNs with we wash drug/handle fish for df ten 
df_ten.iloc[263:265]=np.nan
df_ten.iloc[297:300]=np.nan

df_ten = df_ten.transpose(copy=True)
#%% conditions for tenmin
fish_dmso = np.array([1,2,7,8,13,14,19,20,25,26,31,32,37,38,43,44])-1
fish_melatonin = np.array([3,4,9,10,15,16,21,22,27,28,33,34,39,40,45,46])-1
fish_clonidine = np.array([5,6,11,12,17,18,23,24,29,30,35,36,41,42,47,48])-1

dmso_ten = df_ten.iloc[fish_dmso]

melatonin_ten = df_ten.iloc[fish_melatonin]

clonidine_ten = df_ten.iloc[fish_clonidine]

dmso_ten =dmso_ten.transpose()
clonidine_ten =clonidine_ten.transpose()
melatonin_ten =melatonin_ten.transpose()

#%%
dmso_ten_df=dmso_ten.reset_index()
clo_ten_df=clonidine_ten.reset_index()
mel_ten_df=melatonin_ten.reset_index()

dmso_ten_df=pd.melt(dmso_ten_df,id_vars='CLOCK')
clo_ten_df=pd.melt(clo_ten_df,id_vars='CLOCK')
mel_ten_df=pd.melt(mel_ten_df,id_vars='CLOCK')


#%%
def simpleaxis(ax):
    ax.tick_params(axis='both',labelsize=12.5)
    
row = 1
column =1
fig, ax = plt.subplots(row,column, sharex=True)
# fig.suptitle('tenminute ', fontsize=16)
ax = plt.subplot(row,column,1)

ax.axvspan(53.74, 58.91, alpha=0.25, color='#c2a5cf',zorder=1) 
ax.axvspan(14, 24, alpha=0.2, color='grey',zorder=1) 
ax.axvspan(38, 48, alpha=0.2, color='grey',zorder=1) 
ax.axvspan(62, 72, alpha=0.2, color='grey',zorder=1) 
ax=sns.lineplot(data= dmso_ten_df,x='CLOCK',y='value',color='#7fbf7b',zorder=2)
ax=sns.lineplot(data= clo_ten_df,x='CLOCK',y='value',color='#a6611a',zorder=2)
ax=sns.lineplot(data= mel_ten_df,x='CLOCK',y='value',color='#af8dc3',zorder=2)

blx1,bly1 = [53.739,53.739],[-0.5,125]
blx2,bly2 = [58.9,58.9],[-0.5,125]
plt.plot(blx1,bly1,linestyle='--',color='#c2a5cf',alpha=0.3,zorder=1)
plt.plot(blx2,bly2,linestyle='--',color='#c2a5cf',alpha=0.25,zorder=1)
plt.legend(['DMSO','Melatonin','Clonidine'])
plt.xlim(45,80)
sns.despine()
simpleaxis(ax)
#%%
drug_activity_df = pd.read_excel("Ext Figure9.xlsx",sheet_name='e')

def simpleaxis(ax):
    ax.tick_params(axis='y',labelsize=5)
    # ax.tick_params(axis='x',labelsize=15)
colors_pal = ['#7fbf7b',
              '#af8dc3',
              '#a6611a']
# sns.set_theme(style="whitegrid")
sns.set_style('ticks')
# Initialize the figure
# f, ax = plt.subplots(2,1)
row = 3
column = 3
fig, ax = plt.subplots(row,column, sharex=False)
fig.suptitle('Box 4_5_box67 (24wells) activity')

blx1,bly1 = [-0.5,9],[0,0]

sns.despine(bottom=True, left=True)
# Show each observation with a scatterplot
ax = plt.subplot(row,column,1)
sns.stripplot(
    data=drug_activity_df, y="drugged_average_act_10min", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="drugged_average_act_10min", x="Conditions",
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
# simpleaxis(ax)
sns.despine()

ax = plt.subplot(row,column,2)
sns.stripplot(
    data=drug_activity_df, y="post_drug_average_act", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="post_drug_average_act", x="Conditions",    
    # hue='Box_ID',
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
# simpleaxis(ax)
sns.despine()


ax = plt.subplot(row,column,3)
sns.stripplot(
    data=drug_activity_df, y="night_1h_act", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="night_1h_act", x="Conditions",
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
# simpleaxis(ax)
sns.despine()

ax = plt.subplot(row,column,4)
sns.stripplot(
    data=drug_activity_df, y="drugged_sleep_perhr", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="drugged_sleep_perhr", x="Conditions",
    # hue='Box_ID',
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
# simpleaxis(ax)
sns.despine()

ax = plt.subplot(row,column,5)
sns.stripplot(
    data=drug_activity_df, y="postdrug_sleep_phr", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="postdrug_sleep_phr", x="Conditions",
    # hue='Box_ID',
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
simpleaxis(ax)
sns.despine()


ax = plt.subplot(row,column,6)
sns.stripplot(
    data=drug_activity_df, y="night_1h_sleep", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="night_1h_sleep", x="Conditions",
    # hue='Box_ID',
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
simpleaxis(ax)
sns.despine()

ax = plt.subplot(row,column,7)
sns.stripplot(
    data=drug_activity_df, y="night_1h_sleepbout_av", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="night_1h_sleepbout_av", x="Conditions",
    # hue='Box_ID',
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
simpleaxis(ax)
sns.despine()

ax = plt.subplot(row,column,8)
sns.stripplot(
    data=drug_activity_df, y="night_1h_sleepbout_median", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="night_1h_sleepbout_median", x="Conditions",
    # hue='Box_ID',
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
simpleaxis(ax)
sns.despine()


ax = plt.subplot(row,column,9)
sns.stripplot(
    data=drug_activity_df, y="sleep_latency_night", x="Conditions",
    # hue='Box_ID',
    dodge=True, alpha=.25, zorder=1,palette=colors_pal
)

# Show the conditional means, aligning each pointplot in the
# center of the strips by adjusting the width allotted to each
# category (.8 by default) by the number of hue levels
sns.pointplot(
    data=drug_activity_df, y="sleep_latency_night", x="Conditions",
    # hue='Box_ID',
    join=False, dodge=.8 - .8 / 3, palette=colors_pal,
    markers="o", scale=.75, ci=68
)
simpleaxis(ax)
sns.despine()
