#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jolakoskip
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

N = 10000
mu = 0.021
sigma = np.sqrt(0.01)
tau = -0.01
x0 = np.ones(N).reshape(N,1) * 1

Tc = 1000
dt = 0.1
r_c = mu-tau + 0.02 #2*(mu-tau)+sigma**2
rs = np.linspace(r_c, 2*(mu-tau)+sigma**2 + 0.3, 30)
iters = 100

shares_vec = np.array(pd.read_excel('shares_top1.xlsx').drop(columns='Unnamed: 0'))
fractions_vec = np.array(pd.read_excel('fraction_to_top.xlsx').drop(columns='Unnamed: 0'))

#%% Plot extreme configurations

plt.rcParams.update({'font.size': 12})

fig, (ax1) = plt.subplots(figsize=(5,5))

#ax1.set_rasterization_zorder(0)

#ax1.set_rasterized(True)
left, bottom, width, height = [0.4, 0.35, 0.4, 0.4]
ax2 = fig.add_axes([left, bottom, width, height])

#ax2.set_rasterized(True)

ax1.scatter(rs[::2], np.median(shares_vec,1)[::2], color="#0072BD")
ax1.fill_between(rs,np.min(shares_vec,1), np.max(shares_vec,1), alpha=0.2, color="#0072BD")
ax1.axvline(2*(mu-tau) + sigma**2, color="#D95319", label='$2(\mu-\\tau)+\sigma^2$', linestyle='--', linewidth=2.5)
ax1.set_xlabel('Resetting rate, $r$', size=15)
ax1.set_ylabel('$P_{1\%}$', size=15)
#ax1.tick_params(axis='x',labelsize=13)
ax1.legend()

ax2.scatter(rs[::2], np.median(fractions_vec,1)[::2], color="#0072BD")
ax2.fill_between(rs, np.min(fractions_vec,1), np.max(fractions_vec,1), alpha=0.2, color="#0072BD")
ax2.set_xlabel('Resetting rate, $r$')
ax2.set_ylabel('Fraction to top 1%')
ax2.axvline(2*(mu-tau) + sigma**2, color="#D95319", linestyle='--', linewidth=2.5)
#ax2.legend()


fig.savefig("fig7_p1_condensation.pdf", bbox_inches='tight')