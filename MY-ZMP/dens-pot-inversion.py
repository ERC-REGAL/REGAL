'''
module for simple density-potential inversion
numerical test of different inversion schemes
this script is part of the public domain
CC0 1.0 Universal (CC0 1.0): https://creativecommons.org/publicdomain/zero/1.0/
developed in Python 3.7.0
source: https://mage.uber.space/dokuwiki/material/dens-pot-inversion
'''

from fermion_hamiltonian import GraphHamiltonian
from graph_dft import pot_from_dens # method using functional optimization with BFGS
from timer import timer
from math import pi, sqrt
import matplotlib.pyplot as plt # for plt.show()
import numpy as np
import scipy.interpolate
import random

M = 50 # lattice-sites number, fits to parameter choice
N = 1 # particle number, is always N=1 since this is performed for KS system
max_iter = 1000
tol = 1e-6 # tolerance for convergence

def gauge_pot(pot: list):
    # equal gauge for pot
    # should sum to zero (minimal norm)
    pot = pot - sum(pot)/M
    return pot

@timer
def pot_from_dens_BFGS(H0: GraphHamiltonian, dens: list):
    return gauge_pot(pot_from_dens(H0, dens_target, tol=tol))

@timer
def pot_from_dens_Proc1(H0: GraphHamiltonian, dens: list, eps_list = [1.0, 0.7, 0.4, 0.1], mix = 0.05): # eps not too small, mix rather small
    # method using MY-ZMP scheme for L2 spaces
    dens = np.array(dens)
    pot_list = []
    pot = np.zeros(M)
    for eps in eps_list:
        # pot from previous step, so that fewer iterations are needed for self-consistence
        dens_i = np.zeros(M) # init
        for i in range(max_iter):
            # solve for density
            dens_i = H0.copy().add_potential(pot).eigensystem(test_degen = False, verbose = False).gs_dens
            # update potential
            pot_prev = pot
            pot = (1-mix)*pot + mix/eps * (dens_i - dens) # gauge_pot not needed since ZMP method always gives minimal norm
            # test self-consistence
            if np.linalg.norm(pot - pot_prev, ord=np.inf) < tol:
                print("pot_from_dens_Proc1 (eps={:.2f}): Self-consistency after {} steps.".format(eps, i+1))
                break

        if i==max_iter-1: print("pot_from_dens_Proc1: NO self-consistency after {} steps.".format(i+1))
        pot_list.append(pot)
    
    # extrapolation
    pot_extrapol = np.zeros(M)
    pot_list_T = np.transpose(pot_list)
    for j in range(M):
        # quadratic extrapolation mentioned in ZMP paper
        pot_extrapol[j] = scipy.interpolate.interp1d(eps_list, pot_list_T[j], fill_value='extrapolate', kind='quadratic')(0)
    
    return pot_extrapol

@timer
def pot_from_dens_Proc2(H0: GraphHamiltonian, dens: list, alpha = 0.5):
    # simple difference method like in D. Karlsson's paper 
    pot = np.zeros(M)
    dens = np.array(dens)
    for i in range(max_iter):
        # solve for density
        dens_i = H0.copy().add_potential(pot).eigensystem(test_degen = False, verbose = False).gs_dens
        # update potential
        pot_prev = pot
        pot = pot + alpha * (dens_i - dens) # gauge_pot also seemingly not needed here
        # test convergence
        # better not test on correct density since different potentials might give very similar densities
        if np.linalg.norm(pot - pot_prev, ord=np.inf) < tol:
            print("pot_from_dens_Proc2: Convergence after {} steps.".format(i+1))
            return pot

    print("pot_from_dens_Proc2: NO convergence after {} steps.".format(i+1))
    return pot

# edges for ring graph
edges = []
for i in range(1, M):
    edges.append((i,i+1))
edges.append((M,1)) # append last

H0 = GraphHamiltonian(
        particle_number=N,
        node_number=M,
        graph_edges=edges
    )

# define periodic target density
# made from sin/cos functions with different frequencies
amp_max = 0.5
k_max = 2
x = np.linspace(0, 2*pi, M+1)[0:M]
sin_amps = [.2, .1] #np.random.uniform(low=-amp_max, high=amp_max, size=k_max)
cos_amps = [.3, .2] #np.random.uniform(low=-amp_max, high=amp_max, size=k_max)
dens_target = np.ones(M)
for k in range(1, k_max+1):
    dens_target = dens_target + sin_amps[k-1]*np.sin(x*k) + cos_amps[k-1]*np.cos(x*k)
dens_target = dens_target * N / sum(dens_target)

# invert with functional optimization method BFGS
pot_inv_BFGS = pot_from_dens_BFGS(H0, dens_target)
dens_BFGS = H0.copy().add_potential(pot_inv_BFGS).eigensystem().gs_dens

# invert with Procedure 1
pot_inv_Proc1 = pot_from_dens_Proc1(H0, dens_target)
dens_Proc1 = H0.copy().add_potential(pot_inv_Proc1).eigensystem().gs_dens

# invert with Procedure 2
pot_inv_Proc2 = pot_from_dens_Proc2(H0, dens_target)
dens_Proc2 = H0.copy().add_potential(pot_inv_Proc2).eigensystem().gs_dens

# format strings
fmt_target = 'k^-'
fmt_BFGS = 'r*-'
fmt_Proc2 = 'bo-'
fmt_Proc1 = 'gx-'

plt.figure()
ax = plt.subplot(221)
ax.set_title('potentials')
ax.plot(range(1,M+1), pot_inv_BFGS, fmt_BFGS, label='BFGS')
ax.plot(range(1,M+1), pot_inv_Proc2, fmt_Proc2, label='Proc2')
ax.plot(range(1,M+1), pot_inv_Proc1, fmt_Proc1, label='Proc1')

ax = plt.subplot(222)
ax.set_title('potential difference to BFGS')
ax.plot(range(1,M+1), pot_inv_Proc2-pot_inv_BFGS, fmt_Proc2, label='Proc2')
ax.plot(range(1,M+1), pot_inv_Proc1-pot_inv_BFGS, fmt_Proc1, label='Proc1')
ax.set_yscale('symlog', linthreshy=1e-6)

ax = plt.subplot(223)
ax.set_title('densities')
ax.plot(range(1,M+1), dens_target, fmt_target, label='target')
ax.plot(range(1,M+1), dens_BFGS, fmt_BFGS, label='BFGS')
ax.plot(range(1,M+1), dens_Proc2, fmt_Proc2, label='Proc2')
ax.plot(range(1,M+1), dens_Proc1, fmt_Proc1, label='Proc1')
ax.legend()

ax = plt.subplot(224)
ax.set_title('density difference to target')
ax.plot(range(1,M+1), dens_BFGS-dens_target, fmt_BFGS, label='BFGS')
ax.plot(range(1,M+1), dens_Proc2-dens_target, fmt_Proc2, label='Proc2')
ax.plot(range(1,M+1), dens_Proc1-dens_target, fmt_Proc1, label='Proc1')
ax.set_yscale('symlog', linthreshy=1e-6)

plt.show()