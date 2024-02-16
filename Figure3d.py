# -*- coding: utf-8 -*-
"""
Created on Fri Nov 24 13:56:52 2023

@author: Anya
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

new_df = pd.read_excel("Figure3.xlsx",sheet_name='3e')


colors = ['#15607a','#65e0ba']
g = sns.FacetGrid(data=new_df)
g.map(sns.scatterplot, 'punc_hr4h', 'punc_ZT18_0','sleep_type',palette=colors)
g.map(sns.regplot, 'punc_hr4h', 'punc_ZT18_0',scatter=False)

x = new_df[['punc_hr4h','punc_ZT18_0']].to_numpy()
x = x[~np.isnan(x).any(axis=1)]
slope_a_day, intercept_a_day, r_value_a_day, p_value_a_day, std_err_a_day = stats.linregress(x[:,0],x[:,1])

plt.annotate('r²:{:.3f}'.format(r_value_a_day**2),xy=(1,0),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom', fontsize=9)
plt.annotate('p={:.3f}'.format(p_value_a_day),xy=(1,0.05),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)
plt.annotate('R={:.3f}'.format(r_value_a_day),xy=(1,0.1),xycoords='axes fraction',
            horizontalalignment='right',verticalalignment='bottom',fontsize=9)

ax = g.axes[0,0]
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')

plt.xlim(-8,5)
ax.spines['bottom'].set_color('grey')
ax.spines['top'].set_color('grey')
ax.spines['left'].set_color('grey')
ax.spines['right'].set_color('grey')
ax.tick_params(axis='both',colors='black')


plt.xlabel('')
plt.ylabel('')