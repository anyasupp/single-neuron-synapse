# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 16:02:18 2023

@author: Anya
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 21:50:41 2022

@author: AnyaS
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

data = pd.read_excel("Ext Figure7_part2.xlsx",sheet_name='ext7e1')

datamelt = pd.melt(data,id_vars=['Fish_ID','Time'],value_vars=['stem','proximal_arbour','extra_branch','distal_arbour'],
                   var_name='segment',value_name='RoC')

x1,y1 = [-0.5,2],[0,0]
palette = ['#ff4300','#ff6100','#ff7f00','#ff9d00']
grey = ['grey']*11

stem = [palette[0]]*11
proximal_arbour = [palette[1]]*11
extra_branch = [palette[2]]*11
distal_arbour = [palette[3]]*11 # for using lineplot needs to have the same n as Hue

# data lies at 0 and 1 
datamelt['xloc'] = datamelt['Time']
datamelt['xloc'].replace({'dpf7_10':0,'dpf8_0':1},inplace= True)
datamelt['xloc_jitter'] = datamelt['segment']
datamelt['xloc_jitter'].replace({'stem':-0.04,'proximal_arbour':-0.08,'extra_branch':0.04,'distal_arbour':0.08},inplace= True)
datamelt['time_jitter'] = datamelt['xloc_jitter']+datamelt['xloc']

SMALL_SIZE = 10
MEDIUM_SIZE = 50
BIGGER_SIZE = 15

import matplotlib.pyplot as plt

figure_mosaic = """
.BCDE
"""
def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=15)
    
# Set the figure style
paper_rc = {'lines.linewidth': 2}
plt.rcParams.update(paper_rc)

# Create subplots based on the mosaic layout
fig, axes = plt.subplot_mosaic(figure_mosaic, figsize=(13, 3))

# Show the plot

sns.set_style('ticks',rc=paper_rc)
sns.set_context('notebook',font_scale=0.8)


ax = sns.lineplot(x="Time", y="stem", data=data,hue='Fish_ID', palette = grey,ax=axes['B'],alpha=0.2)
ax = sns.lineplot(x="Time", y="RoC", data=datamelt[datamelt.segment=='stem'],ci=68, color=palette[0],
                  ax=axes['B'],zorder=2,err_style='bars')

simpleaxis(ax)
ax = sns.lineplot(x="Time", y="proximal_arbour", data=data,hue='Fish_ID', palette = grey,
                  ax=axes['C'],alpha=0.2)
ax = sns.lineplot(x="Time", y="RoC", data=datamelt[datamelt.segment=='proximal_arbour'],ci=68, color=palette[1],
                  ax=axes['C'],zorder=2,err_style='bars')
simpleaxis(ax)
ax = sns.lineplot(x="Time", y="extra_branch", data=data,hue='Fish_ID', palette = grey,ax=axes['D'],
                  alpha=0.2)
ax = sns.lineplot(x="Time", y="RoC", data=datamelt[datamelt.segment=='extra_branch'],ci=68, color=palette[2],
                  ax=axes['D'],zorder=2,err_style='bars')
simpleaxis(ax)
ax = sns.lineplot(x="Time", y="distal_arbour", data=data,hue='Fish_ID', palette =grey,ax=axes['E'],
                 alpha=0.2)
ax = sns.lineplot(x="Time", y="RoC", data=datamelt[datamelt.segment=='distal_arbour'],ci=68, color=palette[3],
                  ax=axes['E'],zorder=2,err_style='bars')
simpleaxis(ax)
for label, ax in axes.items():
    ax.get_legend().remove()
    ax.axvspan(0.16, 0.92, alpha=0.15, color='grey') 
    ax.set_xlim(-0.5,1.5) 
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

sns.despine()
#%%
data = pd.read_excel("Ext Figure7_part2.xlsx",sheet_name='ext7e2')

datamelt = pd.melt(data,id_vars=['Fish_ID','Time'],value_vars=['stem_iroc','proxy_iroc','extra_iroc','distal_iroc'],
                    var_name='segment',value_name='Intensity_ratio_roc')

data=data.dropna()
datamelt= datamelt.dropna()

SMALL_SIZE = 10
MEDIUM_SIZE = 50
BIGGER_SIZE = 15


figure_mosaic = """
.BCDE
"""
paper_rc= {'lines.linewidth': 2}
fig,axes = plt.subplot_mosaic(figure_mosaic,figsize=(13,3))
paper_rc = {'lines.linewidth': 2}
plt.rcParams.update(paper_rc)

def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=15)
    

ax = sns.lineplot(x="Time", y="stem_iroc", data=data,hue='Fish_ID', palette = grey,ax=axes['B'],alpha=0.2,linestyle='--')
ax = sns.lineplot(x="Time", y="Intensity_ratio_roc", data=datamelt[datamelt.segment=='stem_iroc'],ci=68, color=palette[0],
                  ax=axes['B'],zorder=2,err_style='bars',linestyle='--')

simpleaxis(ax)
ax = sns.lineplot(x="Time", y="proxy_iroc", data=data,hue='Fish_ID', palette = grey,
                  ax=axes['C'],alpha=0.2,linestyle='--')
ax = sns.lineplot(x="Time", y="Intensity_ratio_roc", data=datamelt[datamelt.segment=='proxy_iroc'],ci=68, color=palette[1],
                  ax=axes['C'],zorder=2,err_style='bars',linestyle='--')
simpleaxis(ax)
ax = sns.lineplot(x="Time", y="extra_iroc", data=data,hue='Fish_ID', palette = grey,ax=axes['D'],
                  alpha=0.2,linestyle='--')
ax = sns.lineplot(x="Time", y="Intensity_ratio_roc", data=datamelt[datamelt.segment=='extra_iroc'],ci=68, color=palette[2],
                  ax=axes['D'],zorder=2,err_style='bars',linestyle='--')
simpleaxis(ax)
ax = sns.lineplot(x="Time", y="distal_iroc", data=data,hue='Fish_ID', palette =grey,ax=axes['E'],
                 alpha=0.2,linestyle='--')
ax = sns.lineplot(x="Time", y="Intensity_ratio_roc", data=datamelt[datamelt.segment=='distal_iroc'],ci=68, color=palette[3],
                  ax=axes['E'],zorder=2,err_style='bars',linestyle='--')
simpleaxis(ax)
for label, ax in axes.items():
    ax.get_legend().remove()
    ax.axvspan(0.16, 0.92, alpha=0.15, color='grey') 
    ax.set_xlim(-0.5,1.5) 
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticklabels([])
    
    
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

sns.despine()
#%%
