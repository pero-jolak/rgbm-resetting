#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jolakoskip
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltime(mu=0.021,sigma=sqrt(0.02),tau=-0.05,N=10).dat").readlines()]

rs = np.array(content)[:,0].astype(float)
tc_n10 = np.array(content)[:,1].astype(float)

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltime(mu=0.021,sigma=sqrt(0.02),tau=-0.05,N=100).dat").readlines()]

rs100 = np.array(content)[:,0].astype(float)
tc_n100 = np.array(content)[:,1].astype(float)

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltime(mu=0.021,sigma=sqrt(0.02),tau=-0.05,N=1000).dat").readlines()]

rs1000 = np.array(content)[:,0].astype(float)
tc_n1000 = np.array(content)[:,1].astype(float)

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltime(mu=0.021,sigma=sqrt(0.02),tau=-0.05,N=10000).dat").readlines()]

rs10000 = np.array(content)[:,0].astype(float)
tc_n10000 = np.array(content)[:,1].astype(float)

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltime(mu=0.021,sigma=sqrt(0.02),tau=-0.05,N=100000).dat").readlines()]

rs100000 = np.array(content)[:,0].astype(float)
tc_n100000 = np.array(content)[:,1].astype(float)


#%%

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltimetaus(mu=0.021,sigma=sqrt(0.02),tau=-0.01,N=1000).dat").readlines()]

rs01 = np.array(content)[:,0].astype(float)
tc_01 = np.array(content)[:,1].astype(float)

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltimetaus(mu=0.021,sigma=sqrt(0.02),tau=-0.001,N=1000).dat").readlines()]

rs001 = np.array(content)[:,0].astype(float)
tc_001 = np.array(content)[:,1].astype(float)

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltimetaus(mu=0.021,sigma=sqrt(0.02),tau=-0.000001,N=1000).dat").readlines()]

rs000001 = np.array(content)[:,0].astype(float)
tc_000001 = np.array(content)[:,1].astype(float)

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltimetaus(mu=0.021,sigma=sqrt(0.02),tau=-0.02,N=1000).dat").readlines()]

rs02 = np.array(content)[:,0].astype(float)
tc_02 = np.array(content)[:,1].astype(float)

content = [i.strip().split() for i in open("data_for_figures/fig2_critical_times/criticaltimetaus(mu=0.021,sigma=sqrt(0.02),tau=-0.05,N=1000).dat").readlines()]

rs05 = np.array(content)[:,0].astype(float)
tc_05 = np.array(content)[:,1].astype(float)

#%%
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E", "#77AC30"]

labels_taus = ["$\\tau = -10^{-6}$", "$\\tau = -10^{-3}$", "$\\tau = -10^{-2}$", "$\\tau = -2\cdot 10^{-2}$", "$\\tau = -5\cdot 10^{-2}$"]
labels_N = ["$N = 10$", "$N=10^2$", "$N=10^3$", "$N=10^4$", "$N=10^5$"]
lw = 2
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(3.5,7))

ax1.semilogy(rs,tc_n10, label=labels_N[0], color=colors[0], linewidth=lw)
ax1.semilogy(rs100,tc_n100, label=labels_N[1], color=colors[1], linewidth=lw)
ax1.semilogy(rs1000,tc_n1000, label=labels_N[2], color=colors[2], linewidth=lw)
ax1.semilogy(rs10000,tc_n10000, label=labels_N[3], color=colors[3], linewidth=lw)
ax1.semilogy(rs100000,tc_n100000, label=labels_N[4], color=colors[4], linewidth=lw)
ax1.set_xlim(np.min(rs), np.max(rs)+0.01)
ax1.set_xlabel("Resetting rate, $r$")
ax1.set_ylabel("$t_c$")
ax1.set_ylim(0,10**5)
ax1.legend(prop={'size': 8})

ax2.semilogy(rs000001,tc_000001, label=labels_taus[0], color=colors[0], linewidth=lw)
ax2.semilogy(rs001,tc_001, label=labels_taus[1], color=colors[1], linewidth=lw)
ax2.semilogy(rs01,tc_01, label=labels_taus[2], color=colors[2], linewidth=lw)
ax2.semilogy(rs02,tc_02, label=labels_taus[3], color=colors[3], linewidth=lw)
ax2.semilogy(rs05,tc_05, label=labels_taus[4], color=colors[4], linewidth=lw)
ax2.set_xlabel("Resetting rate, $r$")
ax2.set_ylabel("$t_c$")
ax2.set_ylim(0,10**5)
ax2.set_xlim(np.min(rs), np.max(rs)+0.01)
ax2.legend(prop={'size': 6})

fig.tight_layout()

plt.savefig("fig2_critical_time.pdf", bbox_inches='tight')

