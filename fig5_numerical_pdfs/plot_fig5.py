#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jolakoskip
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


plt.rcParams.update({'font.size': 14})
#%%

X_Vecs = np.array(pd.read_excel('fig5_pdfs.xlsx').iloc[:,1:])

#%%

labels_main = ['$t=10^0$', '$t=10^1$', '$t=10^2$', '$t=10^3$', '$t=10^4$']
ls = ['dashdot', 'solid', 'solid', 'solid','solid']
# Create the main plot
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30"]
fig, ax1 = plt.subplots(figsize=(5,5))

data_x = np.zeros((200,5))
data_y = np.zeros((200,5))

# Loop through the first two columns for the main plot
for i in range(5):
    #ax1.clear()
    kde = sns.kdeplot(X_Vecs[:, i], ax=ax1)
    kde_line = kde.get_lines()[-1]  # Get the most recent line
    x_values1 = kde_line.get_xdata()
    y_values1 = kde_line.get_ydata()
    
    data_x[:,i] = x_values1
    data_y[:,i] = y_values1
    
    #ax1.semilogx(x_values1, y_values1, color = colors[i], label=labels_main[i], linewidth=3, linestyle=ls[i])
    #ax1.set_xlim(0,4)
    #ax1.set_xscale('log')
#%%

ls = ['solid', 'solid', 'solid', 'solid','solid']

fig, ax1 = plt.subplots(figsize=(5,5))


for i in range(5):
    x_values1 = data_x[:,i]
    y_values1 = data_y[:,i]
    
    ax1.plot(x_values1, y_values1, color = colors[i], label=labels_main[i], linewidth=3, linestyle=ls[i])
    ax1.set_xlim(np.min(data_x),5)
    
ax1.legend()
ax1.set_ylabel('Numerical PDF, $P(x)$')
ax1.set_xlabel('$x$')


fig.savefig("fig5_numerical_pdfs.pdf")