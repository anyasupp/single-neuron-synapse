# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 14:38:38 2023
make day time drug graph 

@author: AnyaS
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data_combined = pd.read_excel("Figure4.xlsx")

#%% grubb's test on chloroadenosin
#from outliers import smirnov_grubbs as grubbs
#
#chloroadenosine = data[data.Condition =='Chloro_adenosine']
#adeno = np.array(chloroadenosine.puncta_perhour)
#exclude_outliers = grubbs.test(adeno, alpha=0.05)
#F7L_20230106 = excluded
data_combined =data_combined[data_combined.Fish_ID != 'F7L_20230106']
   
#%%
colors_pal = ['#7fbf7b',
              '#a6611a','#949494',
              '#543005','#af8dc3','#762a83']
              
def simpleaxis(ax):
#ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=12)

dmso_data = data_combined[data_combined.Condition =='DMSO']
clo_data = data_combined[data_combined.Condition =='Clonidine']
mel_data = data_combined[data_combined.Condition =='Melatonin']
clo_adeno_data = data_combined[data_combined.Condition =='Chloroadenosine_Clonidine']
mel_adeno_data = data_combined[data_combined.Condition =='Chloro2adenosine_melatonin']
adeno_data = data_combined[data_combined.Condition =='Chloro_adenosine']


#prep data for SEM lines
dmso_sleep = dmso_data.sleep_perhour
clonidine_sleep = clo_data.sleep_perhour
melatonin_sleep = mel_data.sleep_perhour
clo_adeno_sleep = clo_adeno_data.sleep_perhour
mel_adeno_sleep = mel_adeno_data.sleep_perhour
adeno_sleep = adeno_data.sleep_perhour

dmso = dmso_data.puncta_perhour
clonidine = clo_data.puncta_perhour
melatonin = mel_data.puncta_perhour
clo_adeno = clo_adeno_data.puncta_perhour
mel_adeno = mel_adeno_data.puncta_perhour
adeno = adeno_data.puncta_perhour

#sem works on series so size (x,)
from scipy.stats import sem
sem_dmso_sleep = sem(dmso_sleep, axis=None, ddof=0,nan_policy = 'omit')
sem_clonidine_sleep = sem(clonidine_sleep, axis=None, ddof=0,nan_policy = 'omit')
sem_melatonin_sleep= sem(melatonin_sleep, axis=None, ddof=0,nan_policy = 'omit')
sem_clo_adeno_sleep = sem(clo_adeno_sleep, axis=None, ddof=0,nan_policy = 'omit')
sem_mel_adeno_sleep = sem(mel_adeno_sleep, axis=None, ddof=0,nan_policy = 'omit')
sem_adeno_sleep = sem(adeno_sleep, axis=None, ddof=0,nan_policy = 'omit')

sem_dmso = sem(dmso, axis=None, ddof=0,nan_policy = 'omit')
sem_clonidine = sem(clonidine, axis=None, ddof=0,nan_policy = 'omit')
sem_melatonin= sem(melatonin, axis=None, ddof=0,nan_policy = 'omit')
sem_clo_adeno = sem(clo_adeno, axis=None, ddof=0,nan_policy = 'omit')
sem_mel_adeno = sem(mel_adeno, axis=None, ddof=0,nan_policy = 'omit')
sem_adeno = sem(adeno, axis=None, ddof=0,nan_policy = 'omit')
                 
# axes for SEM lines 
#sleep
x1,y1 = [-0.2,0.2],[dmso_sleep.mean(),dmso_sleep.mean()] #dmso
x2,y2 = [0.8,1.2],[clonidine_sleep.mean(),clonidine_sleep.mean()] #clonidine
x3,y3 = [1.8,2.2],[adeno_sleep.mean(),adeno_sleep.mean()]# adeno 
x4,y4 = [2.8,3.2],[clo_adeno_sleep.mean(),clo_adeno_sleep.mean()] #clo adeno
x5,y5 = [3.8,4.2],[melatonin_sleep.mean(),melatonin_sleep.mean()]# mel
x6,y6 = [4.8,5.2],[mel_adeno_sleep.mean(),mel_adeno_sleep.mean()]# mel adeno

sem_sleep = [sem_dmso_sleep,sem_clonidine_sleep,sem_adeno_sleep,sem_clo_adeno_sleep,sem_melatonin_sleep,sem_mel_adeno_sleep]
x_sem_sleep = [0,1,2,3,4,5]
y_sem_sleep =[dmso_sleep.mean(),clonidine_sleep.mean(),
        adeno_sleep.mean(),clo_adeno_sleep.mean(),
        melatonin_sleep.mean(),mel_adeno_sleep.mean()]

sem = [sem_dmso,sem_clonidine,sem_adeno,sem_clo_adeno,sem_melatonin,sem_mel_adeno]
x_sem = [0,1,2,3,4,5]
y_sem =[dmso.mean(),clonidine.mean(),
        adeno.mean(),clo_adeno.mean(),
        melatonin.mean(),mel_adeno.mean()]
        
#
row = 2
column = 1
fig, ax = plt.subplots(row,column, sharex=True)

blx1,bly1 = [0,0],[-1,6.5]

sns.despine(bottom=True, left=True)
# Show each observation with a scatterplot
ax = plt.subplot(row,column,1)
sns.stripplot(
    data=data_combined, y="sleep_perhour", x="Condition",
    dodge=False, alpha=.5, zorder=1,palette=colors_pal
)
plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.plot(x4,y4,'k-',alpha=1)
plt.plot(x5,y5,'k-',alpha=1)
plt.plot(x6,y6,'k-',alpha=1)
plt.errorbar(x_sem_sleep, y_sem_sleep, yerr=sem_sleep, linestyle='None', capsize = 5, elinewidth=0.8,markeredgewidth=1.5,color='k')

# ax.set(ylabel=None)
ax.set(xlabel=None)
simpleaxis(ax)

ax = plt.subplot(row,column,2)

x1,y1 = [-0.2,0.2],[dmso.mean(),dmso.mean()] #dmso
x2,y2 = [0.8,1.2],[clonidine.mean(),clonidine.mean()] #clonidine
x3,y3 = [1.8,2.2],[adeno.mean(),adeno.mean()]# adeno
x4,y4 = [2.8,3.2],[clo_adeno.mean(),clo_adeno.mean()] #clo adeno
x5,y5 = [3.8,4.2],[melatonin.mean(),melatonin.mean()]# mel
x6,y6 = [4.8,5.2],[mel_adeno.mean(),mel_adeno.mean()]# mel adeno

plt.plot(bly1,blx1,'k--',alpha=0.25,zorder=1)

# Show each observation with a scatterplot
sns.stripplot(
    data=data_combined, y="puncta_perhour", x="Condition",
    dodge=False, alpha=.5, zorder=2,palette=colors_pal
)

plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.plot(x4,y4,'k-',alpha=1)
plt.plot(x5,y5,'k-',alpha=1)
plt.plot(x6,y6,'k-',alpha=1)
plt.errorbar(x_sem, y_sem, yerr=sem, linestyle='None', capsize = 5, elinewidth=0.8,markeredgewidth=1.5,color='k')

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# ax.set(ylabel=None)
ax.set(xlabel=None)

# ax.set(yticklabels=[])
sns.despine()

# sns.despine(offset=5,trim=True)
simpleaxis(ax)