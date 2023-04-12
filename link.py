# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:45:48 2022

@author: mforeman

Link class functions

"""

import numpy as np

class LINK:
    def __init__(self, node1=None, node2=None, distance=None, k=None, ni=1.0,link_type='internal',linkdict=None):
        '''
        Constructor/initialisation function for individual edge

        Parameters
        ----------
        node1, node2 : int
            Id numbers of nodes at ends of edge.
        distance : float
            Length of connection
        k : float
            wavenumber for propagation along edge
        ni : float, optional
            Complex refractive index of edge. The default is 1.0.
        link_type : str, optional
            Specifies whether edge is an 'internal' (default) or connects to an 'exit' node.
        linkdict : supply dictionary to set link properties from stored/previous values
            Default is None

        Returns
        -------
        None.

        '''
        if linkdict is not None:
            self.dict_to_link(linkdict)
        else:
            self.node1 = int(node1)
            self.node2 = int(node2)
            self.link_type = link_type
            self.reset_link(distance, k, ni)

    def reset_link(self, distance, k, ni):
        '''
        Reset function to change wavevector, length, refractive index of edge.

        Parameters
        ----------
        distance : float
            Length of connection
        k : float
            wavenumber for propagation along edge
        ni : float, optional
            Complex refractive index of edge.

        Returns
        -------
        None.

        '''
        self.distance = distance
        self.n = ni
        self.k = k

        self.S_mat = np.array([[0,np.exp(1j*self.n*self.k*distance)],
                            [np.exp(1j*self.n*self.k*distance),0]])#propagation matrix

        self.iS_mat = np.array([[0,np.exp(-1j*self.n*self.k*distance)],
                            [np.exp(-1j*self.n*self.k*distance),0]]) # inverse propagation matrix

        self.inwave  = np.array([0+0j,0+0j])
        self.outwave = np.array([0+0j,0+0j])

    def update(self,connected_node,direction='forward'):
        '''
        Update function for recursive method of calculating mode distributions.
        Updates output/input amplitudes at each connecting node according to propagation matrix of edg

        Parameters
        ----------
        connected_node : [int]
            List of ids of connected nodes
        direction : str, optional
            Set to 'forward' or 'backwards' depending on recursive algorithm being used.
            The default is 'forward'.

        Returns
        -------
        None.

        '''
        if direction == 'forward':
            for node in connected_node:
                if node.number == self.node1:
                    self.inwave[0] = node.outwave[self.node2]  #outwave to node2   node1 outwave towards node 2 b21
                elif node.number == self.node2:
                    self.inwave[1] = node.outwave[self.node1]  #outwave to node1   node2 outwave towards node 1 b12

            # calculate outgoing wave amplitude using the scattering matrix
            self.outwave = np.matmul(self.S_mat , self.inwave).T

            for node in connected_node:
                if node.number == self.node1:
                    node.inwave[self.node2] = self.outwave[0] #inwave from node2    node 1 inwave from node 2 a21
                elif node.number == self.node2:
                    node.inwave[self.node1] = self.outwave[1] #inwave from node1    node 2 inwave from node 1 a12
        elif direction == 'backwards':
            for node in connected_node:
                if node.number == self.node1:
                    self.outwave[0] = node.inwave[self.node2]
                elif node.number == self.node2:
                    self.outwave[1] = node.inwave[self.node1]

            # calculate outgoing wave amplitude using the scattering matrix
            self.inwave = np.matmul(self.iS_mat, self.outwave).T

            for node in connected_node:
                if node.number == self.node1:
                    node.outwave[self.node2] = self.inwave[0]
                elif node.number == self.node2:
                    node.outwave[self.node1] = self.inwave[1]
        else:
            raise(ValueError,'Unknown run direction type: must be forwards of backwards')
            pass

    ##########################
    # %% Save/Load Functions
    ##########################
    def link_to_dict(self,):
        varnames = ['node1','node2','link_type','distance',
                    'n','k','inwave','outwave','S_mat','iS_mat',
                    ]

        linkdict = dict((v, eval('self.'+v)) for v in varnames if hasattr(self, v))

        return linkdict

    def dict_to_link(self,linkdict):
        for key,val in linkdict.items():
            setattr(self, key, val)