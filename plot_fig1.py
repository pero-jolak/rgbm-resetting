"""
@author: jolakoskip
"""
import numpy as np
import matplotlib.pyplot as plt
r_mech = 'stochastic' #stochastic or first-passage

N = 10000
x0 = np.ones(N).reshape(N,1) * 1

mu = 0.021
sigma = np.sqrt(0.01)
Tc = 100
tau = -0.1
xr = 1 # reset position
r = 0 # reset rate
ds = 0.01
dt = 0.1

T = np.linspace(dt,Tc,int(Tc/dt))

XX = np.load("data_fig1.npy")
X_trajs_r0 = np.load("data_fig1_trajs_r0.npy")
X_trajs_r1 = np.load("data_fig1_trajs_r1.npy")
X_trajs_r2 = np.load("data_fig1_trajs_r2.npy")

#%%
colors = ["#0072BD", "#D95319", "#EDB120", "#7E2F8E"]

fig, (ax1,ax2,ax3) = plt.subplots(1,3, figsize=(9,3))

ax1.plot(T,X_trajs_r0[:,:10],color='gray', alpha=0.4)
ax1.plot(T,XX[:,0,0],color=colors[1], label='Minimum/Maximum')
ax1.plot(T,XX[:,0,1],color=colors[1])
ax1.plot(T,XX[:,0,2],color=colors[0], label='Mean')
ax1.plot(T,XX[:,0,3],color=colors[2], label='Median')

ax2.plot(T,X_trajs_r1[:,:10],color='gray', alpha=0.4)
ax2.plot(T,XX[:,1,0],color=colors[1])
ax2.plot(T,XX[:,1,1],color=colors[1])
ax2.plot(T,XX[:,1,2],color=colors[0])
ax2.plot(T,XX[:,1,3],color=colors[2])


ax3.plot(T,X_trajs_r2[:,:10],color='gray', alpha=0.4)
ax3.plot(T,XX[:,2,0],color=colors[1])
ax3.plot(T,XX[:,2,1],color=colors[1])
ax3.plot(T,XX[:,2,2],color=colors[0])
ax3.plot(T,XX[:,2,3],color=colors[2])

ax1.set_ylim(-1,6)
ax2.set_ylim(-1,6)
ax3.set_ylim(-1,6)

axs = [ax1,ax2,ax3]
labels = ['$(a)$', '$(b)$', '$(c)$']

for i, ax in enumerate(axs):
    
    axs[i].set_ylabel('Resources')
    axs[i].set_xlabel('$t$')
    axs[i].tick_params(axis='x', labelsize=12)
    axs[i].tick_params(axis='y', labelsize=12)
    
    axs[i].text(-0.1, 1.1, labels[i], transform=ax.transAxes,
            fontsize=10, va='top', ha='right')


fig.legend(ncols=5,loc=[0.25,0.92], fontsize=12)

ax1.set_title('$r=0$')
ax2.set_title('$r=0.2$')
ax3.set_title('$r=0.7$')

fig.tight_layout()

plt.savefig("fig1_mean_behaviour.pdf", bbox_inches='tight')
