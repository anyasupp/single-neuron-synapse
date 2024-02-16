
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:05:20 2023
make day time drug graph 
extended figs 
@author: AnyaS
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


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
dmso_data = data_combined[data_combined.Condition =='DMSO']
clo_data = data_combined[data_combined.Condition =='Clonidine']
mel_data = data_combined[data_combined.Condition =='Melatonin']
clo_adeno_data = data_combined[data_combined.Condition =='Chloroadenosine_Clonidine']
mel_adeno_data = data_combined[data_combined.Condition =='Chloro2adenosine_melatonin']
adeno_data = data_combined[data_combined.Condition =='Chloro_adenosine']

#prep data for SEM lines
#ignore the variable names! 
dmso_act = dmso_data.average_activity
clonidine_act = clo_data.average_activity
melatonin_act = mel_data.average_activity
clo_adeno_act = clo_adeno_data.average_activity
mel_adeno_act = mel_adeno_data.average_activity
adeno_act = adeno_data.average_activity

dmso = dmso_data.average_sleepbout
clonidine = clo_data.average_sleepbout
melatonin = mel_data.average_sleepbout
clo_adeno = clo_adeno_data.average_sleepbout
mel_adeno = mel_adeno_data.average_sleepbout
adeno = adeno_data.average_sleepbout

#sem works on series so size (x,)
from scipy.stats import sem
sem_dmso_sleep = sem(dmso_act, axis=None, ddof=0,nan_policy = 'omit')
sem_clonidine_sleep = sem(clonidine_act, axis=None, ddof=0,nan_policy = 'omit')
sem_melatonin_sleep= sem(melatonin_act, axis=None, ddof=0,nan_policy = 'omit')
sem_clo_adeno_sleep = sem(clo_adeno_act, axis=None, ddof=0,nan_policy = 'omit')
sem_mel_adeno_sleep = sem(mel_adeno_act, axis=None, ddof=0,nan_policy = 'omit')
sem_adeno_sleep = sem(adeno_act, axis=None, ddof=0,nan_policy = 'omit')

sem_dmso = sem(dmso, axis=None, ddof=0,nan_policy = 'omit')
sem_clonidine = sem(clonidine, axis=None, ddof=0,nan_policy = 'omit')
sem_melatonin= sem(melatonin, axis=None, ddof=0,nan_policy = 'omit')
sem_clo_adeno = sem(clo_adeno, axis=None, ddof=0,nan_policy = 'omit')
sem_mel_adeno = sem(mel_adeno, axis=None, ddof=0,nan_policy = 'omit')
sem_adeno = sem(adeno, axis=None, ddof=0,nan_policy = 'omit')

#ROC
dmso_roc = dmso_data.RoC
clonidine_roc = clo_data.RoC
melatonin_roc = mel_data.RoC
clo_adeno_roc = clo_adeno_data.RoC
mel_adeno_roc = mel_adeno_data.RoC
adeno_roc = adeno_data.RoC

from scipy.stats import sem
sem_dmso_roc = sem(dmso_roc, axis=None, ddof=0,nan_policy = 'omit')
sem_clonidine_roc = sem(clonidine_roc, axis=None, ddof=0,nan_policy = 'omit')
sem_melatonin_roc= sem(melatonin_roc, axis=None, ddof=0,nan_policy = 'omit')
sem_clo_adeno_roc = sem(clo_adeno_roc, axis=None, ddof=0,nan_policy = 'omit')
sem_mel_adeno_roc = sem(mel_adeno_roc, axis=None, ddof=0,nan_policy = 'omit')
sem_adeno_roc = sem(adeno_roc, axis=None, ddof=0,nan_policy = 'omit')

#%% axes for SEM lines 

def simpleaxis(ax):
    # ax.tick_params(axis='both',labelsize=14)
    ax.tick_params(axis='y',labelsize=12)
colors_pal = ['#7fbf7b',
              '#a6611a','#949494',
              '#543005','#af8dc3','#762a83']


sem_sleep = [sem_dmso_sleep,sem_clonidine_sleep,sem_adeno_sleep,sem_clo_adeno_sleep,
             sem_melatonin_sleep,sem_mel_adeno_sleep]
x_sem_sleep = [0,1,2,3,4,5]
y_sem_sleep =[dmso_act.mean(),clonidine_act.mean(),
        adeno_act.mean(),clo_adeno_act.mean(),
        melatonin_act.mean(),mel_adeno_act.mean()]

sem = [sem_dmso,sem_clonidine,sem_adeno,sem_clo_adeno,sem_melatonin,sem_mel_adeno]
x_sem = [0,1,2,3,4,5]
y_sem =[dmso.mean(),clonidine.mean(),
        adeno.mean(),clo_adeno.mean(),
        melatonin.mean(),mel_adeno.mean()]
        

sem_roc = [sem_dmso_roc,sem_clonidine_roc,sem_adeno_roc,sem_clo_adeno_roc,sem_melatonin_roc,sem_mel_adeno_roc]
y_sem_roc =[dmso_roc.mean(),clonidine_roc.mean(),
        adeno_roc.mean(),clo_adeno_roc.mean(),
        melatonin_roc.mean(),mel_adeno_roc.mean()]


row = 3
column = 1
fig, ax = plt.subplots(row,column, sharex=True)


sns.despine(bottom=True, left=True)
# Show each observation with a scatterplot
ax = plt.subplot(row,column,1)
sns.stripplot(
    data=data_combined, y="average_activity", x="Condition",
    dodge=False, alpha=.5, zorder=1,palette=colors_pal
)
x1,y1 = [-0.2,0.2],[dmso_act.mean(),dmso_act.mean()] #dmso
x2,y2 = [0.8,1.2],[clonidine_act.mean(),clonidine_act.mean()] #clonidine
x3,y3 = [1.8,2.2],[adeno_act.mean(),adeno_act.mean()]# adeno
x4,y4 = [2.8,3.2],[clo_adeno_act.mean(),clo_adeno_act.mean()] #clo adeno
x5,y5 = [3.8,4.2],[melatonin_act.mean(),melatonin_act.mean()]# mel
x6,y6 = [4.8,5.2],[mel_adeno_act.mean(),mel_adeno_act.mean()]# mel adeno

plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.plot(x4,y4,'k-',alpha=1)
plt.plot(x5,y5,'k-',alpha=1)
plt.plot(x6,y6,'k-',alpha=1)
plt.errorbar(x_sem_sleep, y_sem_sleep, yerr=sem_sleep, linestyle='None', capsize = 5, elinewidth=0.8,markeredgewidth=1.5,color='k')
plt.xlim(-0.5,5.8)
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


# Show each observation with a scatterplot
sns.stripplot(
    data=data_combined, y="average_sleepbout", x="Condition",
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
plt.xlim(-0.5,5.8)
# ax.set(yticklabels=[])
sns.despine()

# sns.despine(offset=5,trim=True)
simpleaxis(ax)

ax = plt.subplot(row,column,3)

x1,y1 = [-0.2,0.2],[dmso_roc.mean(),dmso_roc.mean()] #dmso
x2,y2 = [0.8,1.2],[clonidine_roc.mean(),clonidine_roc.mean()] #clonidine
x3,y3 = [1.8,2.2],[adeno_roc.mean(),adeno_roc.mean()]# adeno 
x4,y4 = [2.8,3.2],[clo_adeno_roc.mean(),clo_adeno_roc.mean()] #clo adeno
x5,y5 = [3.8,4.2],[melatonin_roc.mean(),melatonin_roc.mean()]# mel
x6,y6 = [4.8,5.2],[mel_adeno_roc.mean(),mel_adeno_roc.mean()]# mel adeno

blx1,bly1 = [0,0],[-0.5,6.5]

# Show each observation with a scatterplot
sns.stripplot(
    data=data_combined, y="RoC", x="Condition",
    dodge=False, alpha=.5, zorder=2,palette=colors_pal
)

plt.plot(x1,y1,'k-',alpha=1)
plt.plot(x2,y2,'k-',alpha=1)
plt.plot(x3,y3,'k-',alpha=1)
plt.plot(x4,y4,'k-',alpha=1)
plt.plot(x5,y5,'k-',alpha=1)
plt.plot(x6,y6,'k-',alpha=1)
plt.errorbar(x_sem, y_sem_roc, yerr=sem_roc, linestyle='None', capsize = 5, elinewidth=0.8,markeredgewidth=1.5,color='k')
plt.plot(bly1,blx1,'k--',alpha=0.25,zorder=1)

ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

# ax.set(ylabel=None)
ax.set(xlabel=None)

# ax.set(yticklabels=[])
sns.despine()
plt.xlim(-0.5,5.8)
# sns.despine(offset=5,trim=True)
simpleaxis(ax)

