'''
Created on 17.08.2021

@author: mage

library for Graph DFT routines
builds on fermion hamiltonian

use internal Hamiltonians H0 for both systems (w/o external potential)
full system -> ful
reference system -> ref 
'''

import numpy as np
import scipy.linalg as linalg
from fermion_hamiltonian import GraphHamiltonian
from scipy.optimize import minimize
from math import sqrt, pi, cos, sin
from cmath import exp
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import pyvista as pv

def test_dens(dens, M, N, raiseError = True):
    # check if density adds up to N and is in [0,1]
    if len(dens) != M:
        if raiseError: raise ValueError('Density does not have the correct size.')
        return False
    if not all(rho >= 0 and rho <= 1 for rho in dens):
        if raiseError: raise ValueError('Density must be in [0,1] at every node.')
        return False
    if abs(sum(dens) - N) > 1e-10:
        if raiseError: raise ValueError('Density does not add up to N={}.'.format(N))
        return False
    return True

def pot_from_dens(H0: GraphHamiltonian, dens: list, tol = 1e-5):
    # inverse problem of finding potential for a density
    # starts from a GraphHamiltonian H0
    test_dens(dens, H0.M, H0.N)
    # define function that is optimized for fixed rho: -(E(v) - <v,rho>)
    def G(pot: list):
        # add pot to base Hamiltonian
        H = H0.copy().add_potential(pot)
        H.eigensystem(test_degen = False)
        return -(H.gs_energy - np.dot(pot, dens))
    # perform optimization
    # other methods give very similar results
    return minimize(G, np.zeros(H0.M), method='BFGS', options={'disp': False, 'gtol': tol})['x']

def KS_iteration(H0ful: GraphHamiltonian, H0ref: GraphHamiltonian, pot_ext: list, dens_initial = None, mixing_dens = 1, mixing_pot = 1, pulay_depth = 0, max_iter = 20, convergence_plot = False):
    if (not isinstance(mixing_dens, (int, float)) or mixing_dens <= 0):
        raise ValueError('The density mixing parameter must be a strictly positive number.')
    if (not isinstance(mixing_pot, (int, float)) or mixing_pot <= 0):
        raise ValueError('The potential mixing parameter must be a strictly positive number.')
    if (pulay_depth >= 2):
        print('Use Pulay DIIS with depth {}'.format(pulay_depth))
        
    if dens_initial is None: # initialize with uniform density
        dens = np.ones(H0ref.M)*H0ref.N/H0ref.M
    else:
        test_dens(dens_initial, H0ref.M, H0ref.N)
        dens = np.array(dens_initial)
    pot_KS = pot_from_dens(H0ref, dens) # starting KS pot
    
    dens_array = []
    dens_diff_array = []
    pot_KS_diff_array = []
    pulay_stack = [] # dens and residual pairs
    
    # iterate
    for i in range(max_iter):
        # (a)
        # next potential
        # v_{i+1} = v_ext + v_Hxc = v_ext + dFful(dens) - dFref(dens) = v_ext + dFful(dens) + v_i
        pot_KS_old = pot_KS
        pot_KS_new = pot_ext - pot_from_dens(H0ful, dens) + pot_KS_old
        # remove gauge from pot_KS
        pot_KS_new = pot_KS_new - sum(pot_KS)/H0ref.M
        pot_KS_diff = np.linalg.norm(pot_KS_new - pot_KS_old) # for convergence test before mixing
        # mixing in pot_KS
        pot_KS = pot_KS_old + (pot_KS_new - pot_KS_old)*mixing_pot
        # (b)
        # get reference system gs density
        # this solves the whole SE and does not depend on a non-interacting system
        H0ref_new = H0ref.copy().add_potential(pot_KS)
        H0ref_new.eigensystem(test_degen = True)
        dens_old = dens
        dens_new = H0ref_new.gs_dens
        dens_diff = np.linalg.norm(dens_new - dens_old) # for convergence test before mixing
        if (pulay_depth >= 2): # pulay is used, so save dens and residual in stack
            pulay_stack.append({'dens': dens_old, 'residual': dens_new-dens_old})
            if len(pulay_stack) > pulay_depth: pulay_stack.pop(0)
        # (c)
        # mixing in dens or Pulay DIIS which also uses dens mixing additionally
        if (pulay_depth >= 2 and len(pulay_stack) >= 2):
            dens = pulay_update(pulay_stack, mixing_dens)
        else:
            dens = dens_old + (dens_new - dens_old)*mixing_dens
        dens_array.append(dens)
        
        print('it {}: dens = {}, KS pot = {}, spectrum ref: {}'.format(i, format_list(dens), format_list(pot_KS), format_list(H0ref_new.fermion_eigval)))

        # plot
        oct_dens = dens_to_oct(dens)
        oct_dens_old = dens_to_oct(dens_old)
        #ax.scatter(*oct_dens, color='b', alpha=.2)
        ax.plot([oct_dens[0], oct_dens_old[0]], [oct_dens[1], oct_dens_old[1]], [oct_dens[2], oct_dens_old[2]], color='b', alpha=.5)
        
        if convergence_plot: # save convergence speed data
            dens_diff_array.append(dens_diff)
            pot_KS_diff_array.append(pot_KS_diff)
            
        if pot_KS_diff < 1e-5 or dens_diff < 1e-5:
            break
    
    if convergence_plot:
        plt.figure()
        plt.plot(dens_diff_array, 'r-', label='density difference')
        plt.plot(pot_KS_diff_array, 'b-', label='KS potential difference')
        plt.legend(loc="upper right")
    
    return pot_KS, dens_array

## does not always work because it can lead to densities outside [0,1] !
## but this seems to be the usual Pulay algorithm
## Erik implemented this one time enforcing non-negativity thus solving this problem
def pulay_update(stack, mixing_dens): # implementation of Pulay DIIS after Algorithm 1 in Woods et al, JPhysCondMat 31 (2019), p. 18
    # if called the stack always has len >= 2
    # build residual matrix
    size = len(stack)
    A = np.ones((size+1, size+1))
    for i in range(size):
        for j in range(i+1):
            A[i,j] = np.dot(stack[i]['residual'], stack[j]['residual'])
            if (i != j): A[j,i] = A[i,j]
    A[size,size] = 0
    # build right column
    b = np.zeros(size+1)
    b[size] = 1
    # solve linear system
    c = linalg.solve(A, b, assume_a='sym')
    # make update
    return sum(c[i]*(stack[i]['dens'] + mixing_dens*stack[i]['residual']) for i in range(size))

def format_list(vec): # show floats in list with only 2 digits after comma
    return '[' + ', '.join(['{:.4f}'.format(x) for x in vec]) + ']'

def format_list_curly(vec): # show floats in list with only 2 digits after comma
    return '{' + ', '.join(['{:.4f}'.format(x) for x in vec]) + '}'

def init_octahedron_plot(fig):
    global M, OP, dens_oct, ax
    
    # fixed for octahedron
    M = 4 # number of nodes
    
    ## plot octahedron
    ax = fig.add_subplot(1, 1, 1, projection = '3d', aspect = 1)
    
    # octahedron
    OP = {}
    OP['A'] = [ 0.5,  0.5,  0]
    OP['B'] = [ 0.5, -0.5,  0]
    OP['C'] = [-0.5, -0.5,  0]
    OP['D'] = [-0.5,  0.5,  0]
    OP['E'] = [ 0.0,  0.0,  1/sqrt(2)]
    OP['F'] = [ 0.0,  0.0, -1/sqrt(2)]
    OCTO = [[OP['A'], OP['B'], OP['E']],
            [OP['B'], OP['C'], OP['E']],
            [OP['C'], OP['D'], OP['E']],
            [OP['D'], OP['A'], OP['E']],
            [OP['A'], OP['B'], OP['F']],
            [OP['B'], OP['C'], OP['F']],
            [OP['C'], OP['D'], OP['F']],
            [OP['D'], OP['A'], OP['F']]
    ]
    # corresponding densities
    dens_oct = {}
    dens_oct['A'] = [1,1,0,0]
    dens_oct['B'] = [0,1,1,0]
    dens_oct['C'] = [0,0,1,1]
    dens_oct['D'] = [1,0,0,1]
    dens_oct['E'] = [1,0,1,0]
    dens_oct['F'] = [0,1,0,1]
    # plot octahedron
    pc_oct = Poly3DCollection(OCTO, linewidth=0.5, edgecolor='k')
    pc_oct.set_alpha(0.2) # must be in this order 
    pc_oct.set_facecolor('w')
    ax.add_collection3d(pc_oct)

## routine for projecting dens to octahedron
def dens_to_oct(dens):
    P = [0,0,0]
    for key in OP:
        P = P + np.dot(dens, dens_oct[key])*np.array(OP[key])/2
    return P

## plot red dot at density point
def plot_dens_dot(fig, dens, marker_size=20, marker_color='r', marker_alpha=1):
    plt.figure(fig.number)
    ax.scatter(*dens_to_oct(dens), color=marker_color, s=marker_size, alpha=marker_alpha)
    
## also plot doubly degenerate regions (currently not more)
## eigensystem for H must already be solved
def plot_gs_region(fig, H, raster_amp=10, raster_phase=10, marker_size=20, marker_color='r', marker_alpha=1):
    if abs(H.fermion_eigval[0]-H.fermion_eigval[1]) > 1e-10: # no degeneracy
        plot_dens_dot(fig, H.gs_dens, marker_size=marker_size, marker_color=marker_color, marker_alpha=marker_alpha)
    else:
        # plot degeneracy region
        points = []
        for th in range(raster_amp+1): # parametrization of Bloch sphere
            c1 = cos(pi*th/2/raster_amp) #n/raster_amp
            c2 = sin(pi*th/2/raster_amp) #sqrt(1-(n/raster_amp)**2)
            for ph in range(raster_phase):
                Psi = c1*H.fermion_eigvec[0] + c2*exp(1j*2*pi*ph/raster_phase)*H.fermion_eigvec[1]
                points.append(dens_to_oct(H.dens(Psi)))
                #plot_dens_dot(fig, H.dens(Psi), marker_size=marker_size, marker_color=marker_color, marker_alpha=marker_alpha)
        # get mesh from points
        cloud = pv.PolyData(points)
        grid = cloud.delaunay_2d(alpha=2.0) # make it a mesh
        plt.figure(fig.number)
        if grid.n_cells > 0:
            for i in range(grid.n_cells):
                cell_points = grid.cell_points(i)
                ax.plot([cell_points[0,0],cell_points[1,0]], [cell_points[0,1],cell_points[1,1]], [cell_points[0,2],cell_points[1,2]], color=marker_color, alpha=marker_alpha)
                ax.plot([cell_points[0,0],cell_points[2,0]], [cell_points[0,1],cell_points[2,1]], [cell_points[0,2],cell_points[2,2]], color=marker_color, alpha=marker_alpha)
                ax.plot([cell_points[1,0],cell_points[2,0]], [cell_points[1,1],cell_points[2,1]], [cell_points[1,2],cell_points[2,2]], color=marker_color, alpha=marker_alpha)
        else: # was not able to find surface -> 1dim region
            for i in range(len(points)-1):
                ax.plot([points[i][0],points[i+1][0]], [points[i][1],points[i+1][1]], [points[i][2],points[i+1][2]], color=marker_color, alpha=marker_alpha)
                print(points[i])