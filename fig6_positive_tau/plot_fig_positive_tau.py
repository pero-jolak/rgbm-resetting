#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jolakoskip
"""
import numpy as np
import matplotlib.pyplot as plt
#%%

sim0001 = [i.strip().split() for i in open("sim_pdf(tau=0001).dat").readlines()]
sim0001x = np.array(sim0001)[::,1].astype(float)
sim0001y = np.array(sim0001)[::,0].astype(float)

sim01 = [i.strip().split() for i in open("sim_pdf(tau=01).dat").readlines()]
sim01x = np.array(sim01)[::,1].astype(float)
sim01y = np.array(sim01)[::,0].astype(float)

sim02 = [i.strip().split() for i in open("sim_pdf(tau=02).dat").readlines()]
sim02x = np.array(sim02)[::,1].astype(float)
sim02y = np.array(sim02)[::,0].astype(float)

analytical02 = [i.strip().split() for i in open("numerical_pdf(tau=0.2,sigma=sqrt(0.01),mu=0.021,r=0.1).dat").readlines()]
analytical02x = np.array(analytical02)[::,1].astype(float)
analytical02y = np.array(analytical02)[::,0].astype(float)

analytical01 = [i.strip().split() for i in open("numerical_pdf(tau=0.1,sigma=sqrt(0.01),mu=0.021,r=0.1).dat").readlines()]
analytical01x = np.array(analytical01)[::,1].astype(float)
analytical01y = np.array(analytical01)[::,0].astype(float)

analytical0001 = [i.strip().split() for i in open("numerical_pdf(tau=0.001,sigma=sqrt(0.01),mu=0.021,r=0.1).dat").readlines()]
analytical0001x = np.array(analytical0001)[::,1].astype(float)
analytical0001y = np.array(analytical0001)[::,0].astype(float)



fig, ax = plt.subplots(figsize=(7,7), tight_layout=True)
ax.plot(analytical02y,analytical02x, color="#0072BD", linewidth=5)
ax.plot(analytical01y,analytical01x, color="#D95319", linewidth=5)
ax.plot(analytical0001y,analytical0001x, color="#EDB120", linewidth=5)


ax.scatter(sim02y[::100], sim02x[::100], color='#0072BD', s=100)
ax.scatter(sim01y[::500], sim01x[::500], color='#D95319', s=100)
ax.scatter(sim0001y[::500], sim0001x[::500], color='#EDB120', s=100)
ax.set_xlim([0.3,2])
ax.set_ylim([0,2.2])

ax.set(xlabel='$x$',ylabel="$P(x)$")
ax.xaxis.label.set_size(20)     # change xlabel size
ax.yaxis.label.set_size(20)     # change ylabel size
ax.title.set_size(10)           # change subplot title size


ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)

#ax.set_xticks((0,0.05,0.1,0.15))
ax.tick_params(right=False, left=True, axis='y', color='black', length=10, grid_color='none',width=2)
ax.tick_params(top=False, bottom=True, axis='x', color='black', length=10, grid_color='none',width=2)

labels = ["$\\tau = 0.2$", "$\\tau = 0.1$", "$\\tau = 0.001$"]
ax.legend(labels, ncols=1, loc='upper right', prop={'size': 20})


plt.savefig("fig6_positive_tau.pdf", bbox_inches='tight')

