# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 10:36:12 2022

@author: AnyaS
"""

import pandas as pd
from plotly.offline import plot
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#%%
wide = pd.read_excel("ext_fig2_clustered_timelapse_intensity.xlsx",sheet_name='Sheet1')

wide=wide.set_index('SynapseID')
#create df with prescence/absence synapse
wide_df= wide.copy()
wide_df[wide_df>0]=1

line_df = wide_df.copy() #want to copy for line before fillna because line will find min/max later needs nan
wide_df = wide_df.fillna(0)
line_df[2] = line_df[2].map({1:2})
line_df[3] = line_df[3].map({1:3})
line_df[4] = line_df[4].map({1:4})
line_df[5] = line_df[5].map({1:5})
line_df[6] = line_df[6].map({1:6})
line_df[7] = line_df[7].map({1:7})

synapseID = len(wide) #number of rows in y
timepoint = len(wide.columns) #number if x (column) time point
ylabel = np.flip(wide_df.reset_index()['SynapseID'].tolist())
x, y = np.meshgrid(np.arange(timepoint), np.arange(synapseID))
#flip
size = np.flip(wide_df.to_numpy(),0)
color = np.flip(wide.to_numpy(),0)
line_df = np.flip(line_df.to_numpy(),0)

line_df=line_df.astype('float')
line_df[line_df==0]='nan' #make 0into nan make sure
#find min max of each synapse(row) to draw lines
linemin = np.nanargmin(line_df,axis=1) 
linemax = np.nanargmax(line_df,axis=1) 

#%% plot
from matplotlib.collections import PatchCollection

wide_array = np.flip(wide_df.to_numpy(),0)
# size of circle
R = wide_array/wide_array.max()/3
y_range = np.arange(0, len(wide.index)) #where to plot y

xlabel = ['1030','1300','1530','1800','2030','2300','0130']

def simpleaxis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='y',labelsize=10, width=1,length=10)
    ax.tick_params(axis='x',labelsize=10, width=1,length=10, labelrotation=90)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    

fig, ax = plt.subplots()
plt.vlines(x=y_range, ymin=linemin, ymax=linemax, #plot horizonal lines starting min/max according to circle
            color='grey', lw=2.5,zorder=2)

circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, y.flat, x.flat)]

coloring = PatchCollection(circles, array=color.flatten(), cmap="RdYlGn", zorder=3)

ax.add_collection(coloring)

coloring.set_clim([-2,1])
#shaded night
ax.axhspan(1.5, 5.5, alpha=0.3, facecolor='grey',zorder=1)
ax.set(yticks=np.arange(timepoint),xticks=np.arange(synapseID),
        yticklabels=xlabel, xticklabels=ylabel)


ax.grid(False)
sns.despine(top=True, right=True,left=False,bottom=False,offset=0.5,trim=True)

simpleaxis(ax)
plt.show()
