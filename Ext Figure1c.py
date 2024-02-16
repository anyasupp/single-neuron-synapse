# -*- coding: utf-8 -*-
"""
Created on Sat May 30 12:38:53 2020

@author: AnyaS
"""


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib import figure
from matplotlib.ticker import AutoMinorLocator, FormatStrFormatter

fig =plt.figure(figsize=(10,10))
#%%


dt = pd.read_excel("MAX_20190125_PK2mnx_MGUK_2dpf_1.5h_acetone.xlsx",sheet_name=6); minGFP=0 ; maxGFP=255; minMGUK = 0; maxMGUK = 242
#ax = plt.axes()
#transform into array
distance = np.array(dt['distance'])
GFP = np.array(dt['GFP'])
MGUK = np.array(dt['MGUK'])
#max min value of the two channels from Histogram on ImageJ
GFPn=(GFP-minGFP)/(maxGFP-minGFP)
MGUKn=(MGUK-minMGUK)/(maxMGUK-minMGUK)
Distance = np.around(distance,2)

print(GFPn, MGUKn)

#plt.subplot(2,2,1)
# ax=fig.add_subplot(2,2,1)
ax= plt.subplot(2,2,1)
plt.text(0.9, 0.9,'492', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16) 

plt.plot(Distance,GFPn, linestyle='-', marker='o', color='#2ecc71',linewidth = 2,label='FingR(PSD95)-GFP',markersize=4.5)
plt.plot(Distance,MGUKn, linestyle='-', marker='o', color='magenta',linewidth = 2,label='antiMAGUK',markersize=4.5)
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
sns.despine()


#%%

ax=fig.add_subplot(2,2,2)
dt = pd.read_excel("MAX_20190125_PK2mnx_MGUK_2dpf_1.5h_acetone.xlsx",sheet_name=7); minGFP=0 ; maxGFP=255; minMGUK = 0; maxMGUK = 242

#transform into array
distance = np.array(dt['distance'])
GFP = np.array(dt['GFP'])
MGUK = np.array(dt['MGUK'])
#max min value of the two channels from Histogram on ImageJ
GFPn=(GFP-minGFP)/(maxGFP-minGFP)
MGUKn=(MGUK-minMGUK)/(maxMGUK-minMGUK)
Distance = np.around(distance,2)


plt.text(0.9, 0.9,'493', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16) 

plt.plot(Distance,GFPn, linestyle='-', marker='o', color='#2ecc71',linewidth = 2,markersize=4.5)
plt.plot(Distance,MGUKn, linestyle='-', marker='o', color='magenta',linewidth = 2,markersize=4.5)
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
sns.despine()

#%%
# plot 3 
ax=fig.add_subplot(2,2,3)
dt = pd.read_excel("MAX_20190125_PK2mnx_MGUK_2dpf_1.5h_acetone.xlsx",sheet_name=9); minGFP=0 ; maxGFP=255; minMGUK = 0; maxMGUK = 242

#transform into array
distance = np.array(dt['distance'])
GFP = np.array(dt['GFP'])
MGUK = np.array(dt['MGUK'])
#max min value of the two channels from Histogram on ImageJ
GFPn=(GFP-minGFP)/(maxGFP-minGFP)
MGUKn=(MGUK-minMGUK)/(maxMGUK-minMGUK)
Distance = np.around(distance,2)


plt.text(0.9, 0.9,'495', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16) 

plt.plot(Distance,GFPn, linestyle='-', marker='o', color='#2ecc71',linewidth = 2,markersize=4.5)
plt.plot(Distance,MGUKn, linestyle='-', marker='o', color='magenta',linewidth = 2,markersize=4.5)
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
sns.despine()
#%%
#plot 4 
ax=fig.add_subplot(2,2,4)
dt = pd.read_excel("MAX_20190125_PK2mnx_MGUK_2dpf_1.5h_acetone.xlsx",sheet_name=10); minGFP=0 ; maxGFP=255; minMGUK = 0; maxMGUK = 242

#transform into array
distance = np.array(dt['distance'])
GFP = np.array(dt['GFP'])
MGUK = np.array(dt['MGUK'])
#max min value of the two channels from Histogram on ImageJ
GFPn=(GFP-minGFP)/(maxGFP-minGFP)
MGUKn=(MGUK-minMGUK)/(maxMGUK-minMGUK)
Distance = np.around(distance,2)


plt.text(0.9, 0.9,'496', horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=16) 

plt.plot(Distance,GFPn, linestyle='-', marker='o', color='#2ecc71',linewidth = 2,markersize=4.5)
plt.plot(Distance,MGUKn, linestyle='-', marker='o', color='magenta',linewidth = 2,markersize=4.5)
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
sns.despine()
#
