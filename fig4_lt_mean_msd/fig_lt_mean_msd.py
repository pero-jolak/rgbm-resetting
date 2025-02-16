#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: jolakoskip
"""
import numpy as np
import matplotlib.pyplot as plt


mu = 0.021
tau = -0.01
sigma = np.sqrt(0.01)

Tc = 1000
dt = 0.1

t = np.linspace(0,Tc,int(Tc/dt))


#%%
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
x0 = 1

rs = np.linspace(2*(mu-tau)+sigma**2+0.015,2*(mu-tau)+sigma**2+0.3, 100)
tau = -0.01
lt_msd = np.zeros((len(rs),1))
lt_var = np.zeros((len(rs),1))

for i, r in enumerate(rs):
    
    time_series = msd(t, mu, sigma, r, tau, x0)

    invalid_index = np.argmax(~np.isfinite(time_series))

    if invalid_index > 0:
        valid_series = time_series[:invalid_index]
        
        saturation_value = valid_series[-1]
    else:
        saturation_value = time_series[-1]
        
    lt_msd[i,0] = saturation_value
    lt_var[i,0] = saturation_value - mean(t, x0, r, mu, tau)[-1]**2
    
    print(i)

#%%

x0 = 1

rs2 = np.linspace((mu-tau)+0.005,2*(mu-tau)+sigma**2+0.3, 100)
    
lt_mean = np.zeros((len(rs2),1))

for i, r in enumerate(rs2):
        
    lt_mean[i,0] = mean(t, x0, r, mu, tau)[-1]
    
    print(i)

#%%
fig, ax = plt.subplots(figsize=(4, 4))

left, bottom, width, height = [0.5, 0.4, 0.3, 0.3]

ax.plot(rs,lt_msd, label=r'Long time $\langle x^2 \rangle$', color="#0072BD", linewidth=2.5)
ax.plot(rs2,lt_mean, label=r'Long time $\langle x \rangle$', color="#D95319", linewidth=2.5)

ax2 = fig.add_axes([left, bottom, width, height])

ax2.plot(rs,lt_var, label='Long time VAR', color="#EDB120", linewidth=2)
ax2.set_xlabel('$r$', fontsize=10)
ax2.set_ylabel(r'$\langle x^2 \rangle - \langle x \rangle^2$')
ax2.set_xlim(np.min(rs2), np.max(rs2))
ax2.legend(fontsize=6)

#plt.axvline(2*(mu-tau)+sigma**2, color='black', linestyle='--', linewidth=1)
ax.set_xlabel('Resetting rate, $r$')
ax.set_ylabel('Long time value')
ax.legend()

plt.savefig("fig4_lt_mean_msd.pdf", bbox_inches='tight')

