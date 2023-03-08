'''
module for many-particle system in Slater basis on graphs
this script is part of the public domain
CC0 1.0 Universal (CC0 1.0): https://creativecommons.org/publicdomain/zero/1.0/
developed in Python 3.7.0
source: https://mage.uber.space/dokuwiki/material/fermion-graph
'''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from cmath import exp, pi
from copy import copy, deepcopy
from scipy.special import comb

class GraphHamiltonian:
    def __init__(self, particle_number=1, node_number=1, graph_edges=[], scalar_pot=None, peierls_phase=None, interaction_pot=None, interaction_lambda=1, hopping=-1):
        self.N = particle_number
        self.M = node_number
        
        # test if M >= N
        if self.N > self.M:
            print("The particle number exceeds the node number! Particle number will be reduced to {} to be equal to node number.".format(self.M))
            self.N = self.M
        
        # test if M is big enough for largest node in edges
        if len(graph_edges):
            max_node = max(sum(graph_edges, ())) # sum with () to flatten list
            if max_node > self.M:
                print("The largest node in the list of edges is {} while the node number was specified as {}! Node number will be adapted.".format(max_node, self.M))
                self.M = max_node
        
        # default values
        self.fermion_eigval = None
        self.fermion_eigvec = None
        self.scalar_pot = None
        self.interaction_pot = None
        self.peierls_phase = None
        self.gs_dens = None
        self.gs_curr = None
        self.gs_degeneracy = 1 # default
        self.gs_kappa = 0
        self.gs_rankP = 1
        
        # create graph
        self.G = nx.Graph()
        self.G.add_nodes_from(range(1, self.M+1))
        self.G.add_edges_from(graph_edges)

        # create Slater basis (see manuscript)
        # Slater basis <-> position basis (lexical order)
        self.S = int(comb(self.M, self.N)) # number of basis elements
        self.slater_basis = []
        iset = list(range(1,self.N+1)) # index set for current Slater basis element
        pos = self.N-1 # currently manipulated position
        for _ in range(self.S):
            # save in dict
            self.slater_basis.append(copy(iset)) # else is ref, iset must be sorted
            # iterate lexically
            while iset[pos] == self.M-(self.N-pos-1): 
                pos -= 1
            else:
                iset[pos : self.N] = range(iset[pos]+1, iset[pos]+self.N-pos+1)
                pos = self.N-1
                
        # create parts of Hamiltonian (S x S matrices)
        self.hamiltonian_parts = {}
        self.create(scalar_pot=scalar_pot, peierls_phase=peierls_phase, interaction_pot=interaction_pot, interaction_lambda=interaction_lambda, hopping=hopping)

    def create(self, scalar_pot=None, peierls_phase=None, interaction_pot=None, interaction_lambda=1, hopping=-1):
        # create one-particle Hamiltonian but keep parts separated
        
        # translate scalar_pot into dict with all nodes present
        scalar_pot_dict = {}
        for i in range(self.M):
            if scalar_pot is None:
                scalar_pot_dict[i+1] = 0
            elif type(scalar_pot) is dict:
                if i+1 in scalar_pot:
                    scalar_pot_dict[i+1] = scalar_pot[i+1]
                else:
                    scalar_pot_dict[i+1] = 0
            else: # some form of array is assumed
                try:
                    scalar_pot_dict[i+1] = scalar_pot[i] # is array now
                except IndexError:
                    scalar_pot_dict[i+1] = 0
        scalar_pot = scalar_pot_dict
        
        # get and save settings
        self.scalar_pot = scalar_pot
        self.peierls_phase = peierls_phase
        self.interaction_pot = interaction_pot
        self.interaction_lambda = interaction_lambda
                        
        # create graph Laplacian as complex M x M numpy array
        # SciPy sparse matrix transformed to complex dense matrix
        # Laplacian has opposite sign as in my notes
        T1 = -hopping * np.asarray(nx.laplacian_matrix(self.G).toarray(), dtype=np.complex_)
        #print(np.linalg.eigh(H1)[0]) # print eigenvalues of 1-particle hamiltonian
        D1 = np.diag(np.diag(T1)) # diagonal part
        T1 = T1 - D1 # off-diagonal part
        
        # Peierls phase only multiplied to the off-diagonal part
        if self.peierls_phase is not None:
            for edge in copy(self.peierls_phase): # copy because can be changed
                if edge in self.G.edges: # order does not matter here
                    T1[edge[0]-1,edge[1]-1] *= exp(1j*self.peierls_phase[edge])
                    T1[edge[1]-1,edge[0]-1] *= exp(-1j*self.peierls_phase[edge])
                else: # delete from dict
                    del self.peierls_phase[edge]  
        
        self.hamiltonian_parts['D'] = self.onetomany(D1)
        self.hamiltonian_parts['T'] = self.onetomany(T1)

        # create scalar potential operator
        # scalar_pot can be dict or list
        V1 = np.zeros((self.M, self.M))
        if self.scalar_pot is not None:
            for i in copy(self.scalar_pot): # copy bcs can be changed
                if i >= 1 and i <= self.M:
                    V1[i-1,i-1] = self.scalar_pot[i]
                else: # delete from dict
                    del self.scalar_pot[i]
            self.hamiltonian_parts['V'] = self.onetomany(V1)
            
        # also save one particle Hamiltonian
        self.h = D1 + T1 + V1
                
        # e-e interaction part
        if self.interaction_pot is not None and interaction_lambda != 0:
            W = np.zeros((self.S, self.S))
            for a0 in range(self.S): # a0 starts from 0, loop diagonal
                iset = self.slater_basis[a0]
                for k0,ik in enumerate(iset): # k0 starts from 0
                    for _,il in enumerate(iset[k0+1:]): # ik < il (iset is sorted)
                        # 1/dist interaction potential
                        if interaction_pot == 'inverse_distance':
                            try:
                                dist = nx.shortest_path_length(self.G, ik, il) # compute distance on graph
                                ##print("dist({},{})={}".format(ik, il, dist))
                            except nx.exception.NetworkXNoPath:
                                print('Graph is not connected, distance between nodes set to 1000.')
                                dist = 1000                       
                            W[a0,a0] += 1/dist
                        elif type(interaction_pot) is dict:
                            if (ik,il) in self.interaction_pot:
                                W[a0,a0] += self.interaction_pot[(ik,il)]
                            elif (il,ik) in self.interaction_pot: # could have different order
                                W[a0,a0] += self.interaction_pot[(il,ik)]
                        else:
                            raise TypeError('Interaction potential is not of valid type.')

            self.hamiltonian_parts['W'] = interaction_lambda*W
        
    def onetomany(self, A1): # take a one particle operator and promote it to the Slater basis
        A = np.zeros((self.S, self.S), dtype=np.complex_)
        for a0 in range(self.S): # a0 starts from 0
            iset = self.slater_basis[a0]
            for b0 in range(self.S):
                jset = self.slater_basis[b0]
                # loop through Cartesian product of iset x jset
                for k0,ik in enumerate(iset): # k0 starts from 0
                    for l0,jl in enumerate(jset):
                        # remove ik, jl from index sets and compare remaining sets
                        iset_copy = copy(iset)
                        jset_copy = copy(jset)
                        iset_copy.pop(k0)
                        jset_copy.pop(l0)
                        if iset_copy == jset_copy:
                            # add to H
                            A[a0,b0] += (-1)**(k0+l0) * A1[ik-1,jl-1]
                
        return A
    
    def add_potential(self, pot: list):
        # pot is given as a list with M entries here, not as dict
        # create scalar potential operator
        V1 = np.zeros((self.M, self.M))
        for i in range(self.M):
            V1[i,i] = pot[i]
            self.scalar_pot[i+1] += pot[i] # add to stored potential dict
        # add to V part
        self.hamiltonian_parts['V'] += self.onetomany(V1)
        return self # make callable        
    
    def copy(self):
        return deepcopy(self)
    
    def dens(self, Psi): # dens on nodes of Graph (1...M)
        # Psi given in Slater basis
        dens = []
        for i in range(1,self.M+1): # iterate nodes
            rho = 0 # density at i
            for a0 in range(self.S): # iterate Slater basis indices that must fit indices of Psi
                if i in self.slater_basis[a0]: # node included?
                    rho += abs(Psi[a0])**2
            dens.append(rho)
        return dens
    
    def dens2(self, Psi1, Psi2): # transition dens on nodes of Graph (1...M)
        # Psi1, Psi2 given in Slater basis
        dens2 = []
        for i in range(1,self.M+1): # iterate nodes
            rho2 = 0 # density at i
            for a0 in range(self.S): # iterate Slater basis indices that must fit indices of Psi
                if i in self.slater_basis[a0]: # node included?
                    rho2 += 2*np.conj(Psi1[a0])*Psi2[a0] # define with factor 2
            dens2.append(rho2)
        return dens2

    def densMatrix(self, Psi): # density matrix on nodes of Graph (1...M)
        # Psi given in Slater basis
        D = np.zeros((self.M, self.M), dtype=np.complex_)
        for i in range(1,self.M+1): # iterate nodes
            for j in range(1,self.M+1): # iterate nodes
                for a0 in range(self.S): # loop Slater basis
                    iset = self.slater_basis[a0]
                    if i in iset:
                        for b0 in range(self.S): # loop Slater basis
                            jset = self.slater_basis[b0]
                            if j in jset:
                                # remove i, j from index sets and compare remaining sets
                                iset_copy = copy(iset)
                                jset_copy = copy(jset)
                                iset_copy.remove(i)
                                jset_copy.remove(j)
                                if iset_copy == jset_copy:
                                    D[i-1,j-1] += Psi[a0]*np.conj(Psi[b0])
        return D

    def matrixP(self, setPhi): # calculate M x (g+1)g/2 matrix P
        g = len(setPhi)
        P = np.zeros((self.M, int((g+1)*g/2)), dtype=np.complex_)
        # first g columns are densities
        for i in range(0, g):
            P[:, i] = self.dens(setPhi[i])
        # next g(g-1)/2 columns are transition densities
        for k in range(0, g):
            for l in range(k+1, g):
                i = i+1
                P[:, i] = self.dens2(setPhi[k], setPhi[l])
        return P
    
    def curr(self, Psi): # curr on edges of Graph, numbered like ordering of edges in G
        # Psi given in Slater basis
        curr = []
        for edge in self.G.edges: # iterate through edges in lexical ordering
            J = 0 # current along edge
            i1,i2 = edge
            for a0 in range(self.S): # iterate Slater basis indices that must fit indices of Psi
                Phi1 = self.slater_basis[a0] # Slater basis state
                for b0 in range(self.S):
                    Phi2 = self.slater_basis[b0] # Slater basis state
                    if i1 in Phi1 and i2 in Phi2: # nodes included?
                        # find index of nodes in Slater basis state
                        k1 = Phi1.index(i1)
                        k2 = Phi2.index(i2)
                        # remove i1, i2 from index sets and compare remaining sets
                        Phi1_copy = copy(Phi1)
                        Phi2_copy = copy(Phi2)
                        Phi1_copy.pop(k1)
                        Phi2_copy.pop(k2)
                        if Phi1_copy == Phi2_copy:
                            J += 2 * (self.h[i1-1,i2-1] * (-1)**(k1+k2) * Psi[a0].conjugate() * Psi[b0]).imag
            curr.append(J)
        return curr
    
    def hamiltonian(self):
        # add parts of hamiltonian
        return np.sum([self.hamiltonian_parts[key] for key in self.hamiltonian_parts], axis=0)
    
    def eigensystem(self, get_current = False, test_normalization = False, test_degen = True, verbose = True):
        # solve for eigensystem of sum of Hamiltonian parts
        self.fermion_eigval,self.fermion_eigvec = np.linalg.eigh(self.hamiltonian())
        self.fermion_eigvec = np.transpose(self.fermion_eigvec) # first index is eigvec number
        self.gs_energy = self.fermion_eigval[0]
        self.gs_vector = self.fermion_eigvec[0,:] / np.linalg.norm(self.fermion_eigvec[0,:]) # normalize
        self.gs_dens = np.array(self.dens(self.gs_vector))
        if get_current: self.gs_curr = np.array(self.curr(self.gs_vector)) # sorted like edges in lexical ordering

        # test normalization?
        if test_normalization and verbose:
            print("Normalization test: density sums to {0:.2f} while the particle number is {1}.".format(sum(self.gs_dens), self.N))

        # check for degeneracy (within num accuracy)
        if test_degen:
            self.gs_degeneracy = np.count_nonzero(np.abs(self.fermion_eigval - self.gs_energy) < 1e-10)
            if self.gs_degeneracy > 1:
                if verbose: print("Groundstate has {}-fold degeneracy (or almost). Ground-state vector can thus be non-unique.".format(self.gs_degeneracy))
                # calculate nullity of P (kappa)
                P = self.matrixP(self.fermion_eigvec[0:self.gs_degeneracy, :])
                self.gs_rankP = np.linalg.matrix_rank(P)
                self.gs_kappa = int((self.gs_degeneracy+1)*self.gs_degeneracy/2) - self.gs_rankP
                if verbose: print("Rank and nullity of P matrix: rank = {}, kappa = {}".format(self.gs_rankP, self.gs_kappa))
                # equimix for dens and curr
                for g in range(1,self.gs_degeneracy):
                    self.gs_dens += self.dens(self.fermion_eigvec[g])
                    if get_current: self.gs_current += self.curr(self.fermion_eigvec[g])
                self.gs_dens = np.array(self.gs_dens)/self.gs_degeneracy
                if get_current: self.gs_current = np.array(self.gs_current)/self.gs_degeneracy
                
        return self
         
    def draw(self):
        fig = plt.figure() # new fig
        fig.suptitle('Graph $G(h)$', fontsize=14)
        G_pos = nx.fruchterman_reingold_layout(self.G) # position layout, will be reused
        nx.draw(self.G, pos=G_pos, with_labels=True)
        
        # default distance for labels
        dist_vec = 30/plt.gca().transData.transform((1,1))
        xdist = dist_vec[0]
        ydist = dist_vec[1]
        
        # extra labels for nodes with density if gs density is set
        if self.gs_dens is not None:
            for i in G_pos:
                plt.text(G_pos[i][0], G_pos[i][1]+ydist,
                         s="$\\rho_{{{0}}} = {1:.2f}$".format(i, self.gs_dens[i-1]), horizontalalignment='center')

        # label edges with current if gs current is set
        if self.gs_curr is not None:
            edge_labels = {}
            n = 0
            for edge in self.G.edges:
                label = "J_{{{0}}} = {1:.2f}".format(edge, self.gs_curr[n])
                edge_labels[edge] = "$"+label+"$"
                n += 1
            nx.draw_networkx_edge_labels(self.G, pos=G_pos, edge_labels=edge_labels)
            
        # label nodes with scalar potential
        if self.scalar_pot is not None:
            for i in self.scalar_pot:
                plt.text(G_pos[i][0], G_pos[i][1]-1.5*ydist,
                         s="$v_{{{0}}} = {1:.2f}$".format(i, self.scalar_pot[i]), horizontalalignment='center')
                    
        # label edges with Peierls phase
        if self.peierls_phase is not None:
            for edge in self.peierls_phase: # copy because can be changed
                k,i = edge
                # find middle point
                mid_x = (G_pos[i][0] + G_pos[k][0])/2
                mid_y = (G_pos[i][1] + G_pos[k][1])/2
                # find direction
                dir_x = G_pos[i][0] - G_pos[k][0]
                dir_y = G_pos[i][1] - G_pos[k][1]
                if dir_x == 0:
                    angle = 90
                else:
                    angle = np.arctan(dir_y / dir_x) * 180 / pi
                # transform angle (https://matplotlib.org/3.1.0/gallery/text_labels_and_annotations/text_rotation_relative_to_line.html)
                trans_angle = plt.gca().transData.transform_angles(np.array((angle,)), np.array((mid_x,mid_y)).reshape((1,2)))[0]
                # move in x or y dir?
                if abs(dir_x)*10 > abs(dir_y):
                    xshift = 0
                    yshift = -0.5*ydist 
                else:
                    xshift = -0.5*xdist
                    yshift = 0
                # plot text
                plt.text(mid_x+xshift, mid_y+yshift,
                         s="$\\varphi_{{({0},{1})}} = {2:.2f}$".format(k, i, self.peierls_phase[edge]), rotation=trans_angle, horizontalalignment='center')                
        
        # set margins to make labels visible for sure
        plt.margins(.1)
        
    def draw_fermion(self, basis_label = True):
        # draw the fermion graph
        FG = nx.Graph()
        FG.add_nodes_from(range(1, self.S+1))
        H = self.hamiltonian()
        # loop full Hamiltonian for connections
        # only upper triangle
        for i in range(0, self.S):
            for j in range(0, i):
                if H[i,j] != 0:
                    FG.add_edge(i+1, j+1)
        # draw
        fig = plt.figure() # new fig
        fig.suptitle('Fermionic graph $G(H)$ for $M={}, N={}$'.format(self.M, self.N), fontsize=14)
        FG_pos = nx.fruchterman_reingold_layout(FG)
        nx.draw(FG, pos=FG_pos, with_labels=True)
        
        # default distance for labels
        dist_vec = 30/plt.gca().transData.transform((1,1))
        ydist = dist_vec[1]
        
        # extra labels for Slater basis
        for i in FG_pos:
            label = ''
            if basis_label: label = '$e_{' + ('} \wedge e_{'.join(str(a) for a in self.slater_basis[i-1])) + '}$'
            plt.text(FG_pos[i][0], FG_pos[i][1]+ydist,
                s=label, horizontalalignment='center')