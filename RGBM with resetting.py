"""
@author: jolakoskip
"""

import numpy as np
import matplotlib.pyplot as plt


def rgbm(x0, N, mu, sigma, Tc, tau, dt, xr, r):

    T = np.linspace(dt,Tc,int(Tc/dt))
    X_Vec = np.ones([len(T),N])
    
    x = x0
    
    for t in range(1,len(T)):
        epsilon = np.random.randn(1,N) # noise
        
        dx = x*(mu*dt + sigma * np.dot(np.sqrt(dt), epsilon.T)) - tau * (x - np.mean(x)) * dt  # RGBM part
        x = x + dx
        
        # Reset random trajectories with probability r*dt
                    
        x_r = np.random.binomial(1, r*dt, N)
        
        if np.argwhere(x_r).size > 0:
            x[np.argwhere(x_r)] = xr
        else:
            x = x
        
        X_Vec[t,:] = x.T
        
    return X_Vec, T

#%% Simulate trajectories

N = 100 # number of trajectories
x_0 = 1 # initial position

x0 = np.ones(N).reshape(N,1) * x_0

mu = 0.021 # drift
sigma = np.sqrt(0.01) # noise amplitude
Tc = 100 # time
tau = -0.01 # reallocation parameter
xr = x_0 # resetting position
r = 0 # resetting rate
dt = 0.1 # time interval

X_Vec,T = rgbm(x0, N, mu, sigma, Tc, tau, dt, xr, r)

#%% Plot

plt.plot(T, X_Vec)
plt.xlabel('$t$')
plt.ylabel('$x(t)$')