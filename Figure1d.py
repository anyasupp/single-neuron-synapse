# -*- coding: utf-8 -*-
"""
Created on Sun Jun 19 11:16:03 2022
do selection
@author: AnyaS
"""


import pandas as pd
from plotly.offline import plot
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#%%load data that is regional selection 
data = pd.read_excel("Figure1.xlsx",sheet_name='1d') 


data['TrackID'] = data['TrackID'].astype(str)
#make track ID becomes names(objects) rather than number
data['spotID']= data['TrackID']+data['neuron']


wide = data.pivot(index='spotID',columns='Time',values='intensity_ratio')


#%%sort data in some way 
wide = wide.sort_values([1,2,3,4])
#munually move rows in aesthetically pleasig way for bottom
wide1 = wide[0:3]
wide2 = wide[4:12]
wide3= wide[12:]
wide_new = wide1.append(wide2)
wide_new=wide_new.append(wide[3:4])
wide_new=wide_new.append(wide3)

wide_df= wide_new.copy()

# change values to 1
wide_df[wide_df>0]=1


# try changing value each column to correct time
# wide_df[1] = wide_df[1].map({0:np.})
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
ylabel = np.flip(wide_df.reset_index()['spotID'].tolist())
xlabel = ['10.5','13.0','15.5','18.0','20.5','23.0','1.5']


x, y = np.meshgrid(np.arange(timepoint), np.arange(synapseID))
#flip array so looks nice
size = np.flip(wide_df.to_numpy(),0)
color = np.flip(wide_new.to_numpy(),0) #for bottome using wide_new
# color = np.flip(wide.to_numpy(),0) #for middle using wide

line_df = np.flip(line_df.to_numpy(),0)

line_df=line_df.astype('float')
line_df[line_df==0]='nan' #make 0into nan make sure
#find min max of each synapse(row) to draw lines
linemin = np.nanargmin(line_df,axis=1) 
linemax = np.nanargmax(line_df,axis=1) 


#%% plot
from matplotlib.collections import PatchCollection
fig, ax = plt.subplots()

wide_array = np.flip(wide_df.to_numpy(),0)
# scale the circle size down 
R = wide_array/wide_array.max()/3
#makes the lines that track neurons 
y_range = np.arange(0, len(wide.index)) #where to plot y
plt.hlines(y=y_range, xmin=linemin, xmax=linemax, #plot horizonal lines starting min/max according to circle
            color='grey', lw=4,zorder=2)
#zorder lets plot on top lower is at the back
circles = [plt.Circle((j,i), radius=r) for r, j, i in zip(R.flat, x.flat, y.flat)]

coloring = PatchCollection(circles, array=color.flatten(), cmap="RdYlGn", zorder=3)

ax.add_collection(coloring)

coloring.set_clim([-1.2,1])

ax.set(xticks=np.arange(timepoint),yticks=np.arange(synapseID),
        xticklabels=xlabel, yticklabels=ylabel)

plt.show()
sns.despine(top=False, right=False,left=True,bottom=True,offset=0.5,trim=True)
# sns.despine(bottom=True,right=False,left=True, trim= True)
def simpleaxis(ax):
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.get_xaxis().tick_top()
    ax.get_yaxis().tick_right()
    ax.tick_params(axis='y',labelsize=20, width=2,length=10)
    ax.tick_params(axis='x',labelsize=20, width=2,length=10, labelrotation=45)
    ax.spines['top'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    
simpleaxis(ax)
