# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 12:36:00 2023

@author: Anya
"""

import pandas as pd
from plotly.offline import plot
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%
wide = pd.read_excel("ExtFigure2.xlsx",sheet_name='a')
wide=wide.set_index('SynapseID')


wide_df= wide.copy()
# change values to 1
wide_df[wide_df>0]=1

# try changing value each column to correct time
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
#get labels but we flip so plot looks nice (so flip labels too) 
ylabel = np.flip(wide_df.reset_index()['SynapseID'].tolist())
# xlabel = ['ZT1030','ZT1300','ZT1530','ZT1800','ZT2030','ZT2300','ZT0130']
xlabel = ['10.5','13.0','15.5','18.0','20.5','23.0','1.5']
flip_xlabel= xlabel
flip_ylabel = np.flip(ylabel)

x, y = np.meshgrid(np.arange(timepoint), np.arange(synapseID))
#flip array so looks nice
size = np.flip(wide_df.to_numpy(),0)
color = np.flip(wide.to_numpy(),0)
line_df = np.flip(line_df.to_numpy(),0)

line_df=line_df.astype('float')
line_df[line_df==0]='nan' #make 0into nan make sure
#find min max of each synapse(row) to draw lines
linemin = np.nanargmin(line_df,axis=1) 
linemax = np.nanargmax(line_df,axis=1) 
#%% flipping line min/max along x axis so now synapse 1 starts at left handside
reverse_linemin  = np.flip(linemin)
reverse_linemax  = np.flip(linemax)
#flipud is fliping along axis 0 (up/down)
size_flip = np.flipud(size)
color_flip = np.flipud(color)

wide_array = np.flip(wide_df.to_numpy(),0)
# scale the circle size down 
R = wide_array/wide_array.max()/3

#makes the lines that track neurons 
y_range = np.arange(0, len(wide.index)) #where to plot y

#reverse meshgrid
x_flip = np.flipud(x)
y_flip = np.flip(y)

#reverse R = size 
R_flip = np.flipud(R)

def simpleaxis(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='y',labelsize=15, width=1,length=10)
    ax.tick_params(axis='x',labelsize=15, width=1,length=10, labelrotation=90)
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)


#%%

from matplotlib.collections import PatchCollection
fig, ax = plt.subplots()
#plot horizonal lines starting min/max according to circle
plt.vlines(x=y_range, ymin=reverse_linemin, ymax=reverse_linemax,
            color='grey', lw=2.5,zorder=2)

##plot circles using x y coordinates and radius R in each position 
circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R_flip.flat, y.flat, x.flat)]

coloring = PatchCollection(circles, array=color_flip.flatten(), cmap="RdYlGn", zorder=3)

ax.add_collection(coloring)

coloring.set_clim([-2,1])

ax.axhspan(ymin=1.5, ymax=5.5, alpha=0, facecolor='grey',zorder=1)
        

ax.set(yticks=np.arange(timepoint),xticks=np.arange(synapseID),
        yticklabels=flip_xlabel, xticklabels=flip_ylabel)

ax.grid(False)
sns.despine(top=True, right=True,left=False,bottom=False,offset=0.5,trim=True)

simpleaxis(ax)

ax.invert_yaxis()

plt.show()

ax.axhspan(ymin=1.5, ymax=5.5, alpha=0.3, facecolor='grey',zorder=1)
