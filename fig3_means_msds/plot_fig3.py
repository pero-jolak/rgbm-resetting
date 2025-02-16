#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jolakoskip
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%%

msds_rs = np.array(pd.read_excel('data_for_figures/fig3_means_msds/fig3_msds_rs.xlsx').iloc[:,1:])
means_rs = np.array(pd.read_excel('data_for_figures/fig3_means_msds/fig3_means_rs.xlsx').iloc[:,1:])
T = np.array(pd.read_excel('data_for_figures/fig3_means_msds/fig3_times.xlsx').iloc[:,1:]).reshape(-1)
#%%
def even_markers(x, y, num_markers, color, shape, display, msize, ax):

    xmarkers = np.logspace(np.log10(x[0]), np.log10(x[-1]), num_markers)
    
    ymarkers = np.interp(xmarkers, x, y)
    
    ax.loglog(xmarkers, ymarkers, linestyle='none', marker=shape, 
             markersize=msize, markerfacecolor=color, color='k', 
             label=display)
    
    return xmarkers, ymarkers

#%% MSD analytical

x0 = 1
def msd(t, mu, sigma, r, tau, x0):
    
    term1 = ((r * x0**2) / (r - mu)**2) * np.exp(t * (2 * mu - r + sigma**2 - 2 * tau))
    term1_part1 = (2 * np.exp(-t * (mu + sigma**2 - tau)) * mu * tau) / (mu + sigma**2 - tau)
    term1_part2 = (np.exp(t * (r - 2 * mu - sigma**2 + 2 * tau)) * (r**2 - 2 * r * mu + mu**2 + 2 * r * tau)) / (r - 2 * mu - sigma**2 + 2 * tau)
    term1_ = term1 * (term1_part1 + term1_part2)
    
    term2 = np.exp(t * (2 * mu - r + sigma**2 - 2 * tau)) * (x0**2 - (r * x0**2 * ((2*mu*tau)/(mu+sigma**2-tau) + (r**2-2*r*mu+mu**2+2*r*tau)/(r-2*mu-sigma**2+2*tau)) / (r - mu)**2))
    
    
    mean_x2 = term1_ + term2
    
    return mean_x2

#%% Mean analytical

def mean(t, x0, r, mu, tau):
    
    mean_x = (x0 / (r - mu)) * (r - np.exp(-t * (r - mu + tau)) * mu)
    
    return mean_x
#%%

N = 1000
x0 = np.ones(N).reshape(N,1) * 1

Tc = 1000
dt = 0.01

t = np.linspace(0,Tc,int(Tc/dt))

mu = 0.021
sigma = np.sqrt(0.01)

taus = [-0.01, -0.05, -0.1]
rs = [0.12, 0.22, 0.32]


#%% Crtanje na figurata za trudot (fig3)

x0 = 1

fig, (ax1, ax2) = plt.subplots(2,1, figsize=(5,8))

even_markers(T, msds_rs[:,0], 35, "#0072BD", 'o', '$\\tau=-0.01$', 7, ax2)
ax2.loglog(T, msd(T, mu, sigma, rs[0], taus[0], x0), linewidth=2.5)

even_markers(T, msds_rs[:,1], 35, "#D95319", 'o', '$\\tau=-0.05$', 7, ax2)
ax2.loglog(T, msd(T, mu, sigma, rs[1], taus[1], x0), linewidth=2.5)

even_markers(T, msds_rs[:,2], 35, "#77AC30", 'o', '$\\tau=-0.1$', 7, ax2)
ax2.loglog(T, msd(T, mu, sigma, rs[2], taus[2], x0), linewidth=2.5)

ax2.set_xlabel('$t$')
ax2.set_ylabel('$\\langle x^2(t) \\rangle$')
ax2.legend(fontsize="10")

################################################################################################
even_markers(T, means_rs[:,0], 35, "#0072BD", 'o', '$\\tau=-0.01$', 7, ax1)
ax1.loglog(T,mean(T, x0, rs[0], mu, taus[0]), linewidth=2.5)


even_markers(T, means_rs[:,1], 35, "#D95319", 'o', '$\\tau=-0.05$', 7, ax1)
ax1.loglog(T,mean(T, x0, rs[1], mu, taus[1]), linewidth=2.5)

even_markers(T, means_rs[:,2], 35, "#77AC30", 'o', '$\\tau=-0.1$', 7, ax1)
ax1.loglog(T,mean(T, x0, rs[2], mu, taus[2]), linewidth=2.5)

ax1.set_xlabel('$t$')
ax1.set_ylabel('$\\langle x(t) \\rangle$')

ax1.legend(fontsize="10")
################################################################################################

left, bottom, width, height = [0.35, 0.22, 0.23, 0.15]
ax3 = fig.add_axes([left, bottom, width, height])

#even_markers(T, msds_rs[:,0] - means_rs[:,0]**2, 35, "#0072BD", 'o', '$\\tau=-0.01$', 7, ax3)
ax3.loglog(T,msd(T, mu, sigma, rs[0], taus[0], x0) - mean(T, x0, rs[0], mu, taus[0])**2, linewidth=1, color="#0072BD")

#even_markers(T, msds_rs[:,1] - means_rs[:,1]**2, 35, "#D95319", 'o', '$\\tau=-0.05$', 7, ax3)
ax3.loglog(T,msd(T, mu, sigma, rs[1], taus[1], x0) - mean(T, x0, rs[1], mu, taus[1])**2, linewidth=1, color="#D95319")

#even_markers(T, msds_rs[:,2] - means_rs[:,2]**2, 35, "#77AC30", 'o', '$\\tau=-0.1$', 7, ax3)
ax3.loglog(T,msd(T, mu, sigma, rs[2], taus[2], x0) - mean(T, x0, rs[2], mu, taus[2])**2, linewidth=1, color="#77AC30")

ax3.set_xlabel('$t$', fontsize=8)
ax3.set_ylabel('$\\langle x^2(t) \\rangle - \\langle x(t) \\rangle^2$', fontsize=8)

#ax3.set_xticklabels(ax3.get_xticks(), fontsize=3)
#ax3.set_yticklabels(ax3.get_xticks(), fontsize=3)

ax3.set_ylim(10**(-3),np.max(msds_rs-means_rs**2))
ax3.set_xlim(10**(-1),T[-1])

ax3.tick_params(axis='both', which='major', labelsize=7)

#ax3.legend(fontsize="7")


fig.tight_layout()

plt.savefig("fig3_means_msds.pdf", bbox_inches='tight')

