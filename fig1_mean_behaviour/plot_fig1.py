#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jolakoskip
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%

T = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_time.xlsx').iloc[:,1:]).reshape(-1)
Min_Xs = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_min_xs.xlsx').iloc[:,1:])
Max_Xs = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_max_xs.xlsx').iloc[:,1:])
Median_Xs = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_median_xs.xlsx').iloc[:,1:])
Mean_Xs = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_mean_xs.xlsx').iloc[:,1:])
#%%

trajs_a = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_trajs_a.xlsx').iloc[:,1:])
trajs_b = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_trajs_b.xlsx').iloc[:,1:])
trajs_c = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_trajs_c.xlsx').iloc[:,1:])
#%%

T_crits = np.array(pd.read_excel('data_for_figures/fig1_mean_behaviour/fig1_t_crits.xlsx').iloc[:,1:]).reshape(-1)
#%% Plot (Fig 1)

#plt.rcParams.update({'font.size': 13})

colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"] #, "#77AC30"]

fig, axs = plt.subplots(1,3, figsize=(14,5))

axs = axs.ravel()

for i, label in enumerate(('a)','b)','c)')):
    
    each = 100
    each2 = 10
    
    axs[i].annotate(label, xy=(0.05, 0.90), xycoords="axes fraction", size=15)
    
    axs[i].scatter(T[::each], Min_Xs[::each,i], color=colors[0], label='Minimum $\\langle x(t) \\rangle_N$')
    axs[i].scatter(T[::each], Max_Xs[::each,i], color=colors[1], label='Maximum $\\langle x(t) \\rangle_N$')
    axs[i].scatter(T[::each], Median_Xs[::each,i], color=colors[2], label='Median $\\langle x(t) \\rangle_N$')
    axs[i].scatter(T[::each], Mean_Xs[::each,i], color=colors[3], label='Mean $\\langle x(t) \\rangle_N$')
    
    if i==0:
        X_Vec = trajs_a
    elif i==1:
        X_Vec = trajs_b
    else:
        X_Vec = trajs_c
        
    axs[i].plot(T[::each2], np.sign(X_Vec[::each2,:])*np.log(1+np.abs(X_Vec[::each2,:])), color='gray', alpha=0.3)

    axs[i].axvline(T_crits[i],color='black',linestyle='--',label='$t_c$')
    
    axs[i].set_xlabel('$t$')
    axs[i].tick_params(axis='x', labelsize=15)
    axs[i].tick_params(axis='y', labelsize=15)
    
    if i == 0:
        
        axs[i].set_ylabel('$sign(\\langle x(t) \\rangle_N) \\times log(1+|\\langle x(t) \\rangle_N|)$', size=17)
        axs[i].set_title('Frozen regime, $0=r<\mu-\\tau$', size=15)
        fig.legend(ncols=5,loc=[0.12,0.93], fontsize=15)
    
    elif i==1:
        
        axs[i].set_title('Unstable regime, $\mu-\\tau<r<2(\mu-\\tau)+\sigma^2$', size=15)
    
    elif i==2:
        
        axs[i].set_title('Stable regime, $r>2(\mu-\\tau)+\sigma^2$', size=15)

#fig.legend(['Minimum $\\langle x(t) \\rangle_N$','Maximum $\\langle x(t) \\rangle_N$',
#           'Median $\\langle x(t) \\rangle_N$','Mean $\\langle x(t) \\rangle_N$','s','s'],ncols=7,loc=[0.25,0.95])


fig.tight_layout()

plt.savefig("fig1_mean_behaviour.pdf", bbox_inches='tight')

