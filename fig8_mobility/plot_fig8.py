#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jolakoskip
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

corr_vals = np.array(pd.read_excel('fig8_corr_vals.xlsx').iloc[:,1:])
ige_vals = np.array(pd.read_excel('fig8_ige_vals.xlsx').iloc[:,1:])
rs = np.array(pd.read_excel('fig8_rs.xlsx').iloc[:,1:])


#%%


colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"]
labels = ["$\\tau = -0.2$", "$\\tau = -0.05$", "$\\tau = -0.01$", "$\\tau = 0$"]

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(3,6))

for i in range(4):
    
    ax1.plot(rs, corr_vals[:,i], '-o', color=colors[i])
    #ax1.fill_between(rs, corr_vals_min[:,i], corr_vals_max[:,i], alpha=0.2, color=colors[i], label='_nolegend_')

ax1.set_xlabel('Resetting rate, $r$')
ax1.set_ylabel('Spearman rank correlation')

ax1.legend(labels, ncols=1, loc='best') #loc=[0.17, 0.92])

for i in range(4):
    
    ax2.plot(rs, ige_vals[:,i], '-o', color=colors[i])
    #ax2.fill_between(rs, ige_vals_min[:,i], ige_vals_max[:,i], alpha=0.2, color=colors[i], label='_nolegend_')

ax2.set_xlabel('Resetting rate, $r$')
ax2.set_ylabel('IGE')

ax2.legend(labels, ncols=1, loc='best') #loc=[0.17, 0.92])

fig.tight_layout()

plt.savefig("fig8_mobility.pdf", bbox_inches='tight')
