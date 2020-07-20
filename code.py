########################################### 
"""
Project: Analytical solutions to the Lindblad equation for optical decoherence of nonlinear optomechanical systems
Description: This code evolves states under the non-linear optomechanical Hamiltonians under noisy evolution and outputs a number of datasets for various quantities
Functions: 
"""
########################################### 

# Import dependencies 
from qutip import *
import yaml
import matplotlib.pyplot as plt
import os
import datetime
import time
import numpy as np
from scipy.special import factorial
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib import rc
#from palettable.scientific.diverging import Berlin_20
import palettable


rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

# construct the state
states = []
alpha = 2
N = 50

# Generate the pure states 
g0 = 0.1
for n in range(0, N):
	states.append(np.exp( - alpha**2/2)*alpha**n*np.exp( 1.j*2.*np.pi*g0**2*n**2)/np.sqrt(factorial(n))*basis(N,n))
psig01 = sum(states)
g0 = 0.3
states = []
for n in range(0, N):
	states.append(np.exp( - alpha**2/2)*alpha**n*np.exp( 1.j*2.*np.pi*g0**2*n**2)/np.sqrt(factorial(n))*basis(N,n))
psig03 = sum(states)
g0 = 0.5
states = []
for n in range(0, N):
	states.append(np.exp( - alpha**2/2)*alpha**n*np.exp( 1.j*2.*np.pi*g0**2*n**2)/np.sqrt(factorial(n))*basis(N,n))
psig05 = sum(states)



# Solve the Master equation and get the final states
# Define the optomechanical Hamiltonian
a = tensor(destroy(N), qeye(N))
b = tensor(qeye(N), destroy(N))

H01 = b.dag()*b - 0.1*a.dag()*a *(b.dag() + b)
H03 = b.dag()*b - 0.3*a.dag()*a *(b.dag() + b)
H05 = b.dag()*b - 0.5*a.dag()*a *(b.dag() + b)

# Define times and the initial states
times = np.arange(0.0, 6.28, 1.56)
print(times)

psi0 = tensor(coherent(N, alpha), basis(N,0))
L = np.sqrt(0.1)*a

resultg01a01 = mesolve(H01, psi0, times,  c_ops = np.sqrt(0.1)*a, e_ops = [], progress_bar = True)
resultg03a01 = mesolve(H03, psi0, times,  c_ops = np.sqrt(0.1)*a, e_ops = [], progress_bar = True)
resultg05a01 = mesolve(H05, psi0, times,  c_ops = np.sqrt(0.1)*a, e_ops = [], progress_bar = True)

resultg01a02 = mesolve(H01, psi0, times,  c_ops = np.sqrt(0.2)*a, e_ops = [], progress_bar = True)
resultg03a02 = mesolve(H03, psi0, times,  c_ops = np.sqrt(0.2)*a, e_ops = [], progress_bar = True)
resultg05a02 = mesolve(H05, psi0, times,  c_ops = np.sqrt(0.2)*a, e_ops = [], progress_bar = True)


data = []

xvec = np.arange(-4.5, 4.5, 0.01)
yvec = np.arange(-4.5, 4.5, 0.01)

W01 = wigner(psig01, xvec, yvec)
W03 = wigner(psig03, xvec, yvec)
W05 = wigner(psig05, xvec, yvec)

print(np.amax(W05))
print(np.amin(W05))

Wnoisyg01a01 = wigner(resultg01a01.states[-1].ptrace(0), xvec, yvec)
Wnoisyg03a01 = wigner(resultg03a01.states[-1].ptrace(0), xvec, yvec)
Wnoisyg05a01 = wigner(resultg05a01.states[-1].ptrace(0), xvec, yvec)

Wnoisyg01a02 = wigner(resultg01a02.states[-1].ptrace(0), xvec, yvec)
Wnoisyg03a02 = wigner(resultg03a02.states[-1].ptrace(0), xvec, yvec)
Wnoisyg05a02 = wigner(resultg05a02.states[-1].ptrace(0), xvec, yvec)

nrm05 = mpl.colors.Normalize(-W05.max(), W05.max())

# Set up figure and image grid
fig = plt.figure(figsize=(15, 5))
plt.rc('text', usetex=True)
plt.rc('font', family='serif')

grid = ImageGrid(fig, 111,          # as in plt.subplot(111)
                 nrows_ncols=(3,3),
                 axes_pad=0.15,
                 share_all=True,
                 cbar_location="right",
                 cbar_mode="single",
                 cbar_size="5%",
                 cbar_pad=0.15,
                 )

#globalcmap = palettable.scientific.diverging.Berlin_20.mpl_colormap
globalcmap = "Spectral"


plt1 = grid[0].contourf(xvec, xvec, W01, 100, cmap=globalcmap, norm=nrm05)

grid[0].set_title(r'$\tilde{g}_0 = 0.1$')
grid[0].set_ylabel(r'$y,\tilde{\kappa}_{\mathrm{c}} = 0$')
grid[1].set_title(r'$\tilde{g}_0 = 0.3$')
grid[2].set_title(r'$\tilde{g}_0 = 0.5$')
#axes[0].xlim(-5,5)
#axes[0].ylim(-5,5)
#cb1 = fig.colorbar(plt1, ax=axes[0])
plt2 = grid[1].contourf(xvec, xvec, W03, 100, cmap=globalcmap, norm=nrm05)  # Apply Wigner colormap
#axes[1].set_title("g0 = 0.3");
plt3 = grid[2].contourf(xvec, xvec, W05, 100, cmap=globalcmap, norm=nrm05) 
#axes[2].set_title("g0 = 0.5");
#cb3 = fig.colorbar(plt3, ax=axes[2])

plt4 = grid[3].contourf(xvec, xvec, Wnoisyg01a01, 100, cmap=globalcmap, norm=nrm05) 
grid[3].set_ylabel(r'$y, \tilde{\kappa}_{\mathrm{c}}= 0.1$')
plt5 = grid[4].contourf(xvec, xvec, Wnoisyg03a01, 100, cmap=globalcmap, norm=nrm05) 
plt6 = grid[5].contourf(xvec, xvec, Wnoisyg05a01, 100, cmap=globalcmap, norm=nrm05) 

plt4 = grid[6].contourf(xvec, xvec, Wnoisyg01a02, 100, cmap=globalcmap, norm=nrm05) 
plt4 = grid[7].contourf(xvec, xvec, Wnoisyg03a02, 100, cmap=globalcmap, norm=nrm05) 
plt4 = grid[8].contourf(xvec, xvec, Wnoisyg05a02, 100, cmap=globalcmap, norm=nrm05) 

grid[6].set_ylabel(r'$y, \tilde{\kappa}_{\mathrm{c}} = 0.2 	$')
grid[6].set_xlabel(r'$x$')
grid[7].set_xlabel(r'$x$')
grid[8].set_xlabel(r'$x$')

# Sort out the minus signs for the ticks
grid[6].set_xticklabels([r' ', r'$\mbox{-}2.5$',r'$0$', r'$2.5$', r' '])
grid[6].set_yticklabels([r' ', r'$\mbox{-}2.5$',r'$0$', r'$2.5$', r' '])

# Add the colourbar
grid[2].cax.colorbar(plt3, ticks = [np.min(W05), 0, np.amax(W05)])
grid[2].cax.set_yticklabels([r'$\mbox{-}\pi^{-1}$',r'$0$', r'$\pi^{-1}$'])
grid[2].cax.toggle_label(True)
#grid[2].set_yticklabels(['-0.2','-0.15', '-0.1', '0', '0.1', '0.15', '0.2']) 



fig.tight_layout()
plt.savefig("cat-states.jpg", dpi = 300)
plt.show()

