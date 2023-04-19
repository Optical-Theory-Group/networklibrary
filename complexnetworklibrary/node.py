# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:44:38 2022

@author: Matthew Foreman

Node class functions
"""

import numpy as np


class NODE:

    def __init__(self, number=0, pos=(0, 0), node_type='internal', nodedict=None):
        """
        Constructor/initialisation function for individual node

        Parameters
        ----------
        number : int
            Unique id number for node.
        pos : (x,y) tuple
            Position of nodes.
        node_type : string, optional
            Specifies whether node is an 'internal' or 'exit' node
        nodedict : supply dictionary to set node properties from stored/previous values
            Default is None
        """
        self.inwave = None
        self.outwave = None
        self.inwave_np = None
        self.outwave_np = None
        self.S_mat = None
        self.iS_mat = None

        if nodedict is not None:
            self.dict_to_node(nodedict)
        else:
            self.number = int(number)
            self.position = np.array(pos)
            self.node_type = node_type
            self.n_connect = 0
            self.sorted_connected_nodes = []

    def degree(self):
        """
        Returns degree of node, i.e. the number of connecting edges

        """
        return self.n_connect

    # def init_Smat(self,Smat_type,scat_loss,**kwargs):
    #     """
    #     Initialise scattering matrix of node

    #     Parameters
    #     ----------
    #     scat_mat_type : str
    #         Specifies type of scattering matrix to use.
    #             'identity': SM is set to identity matrix - complete reflection at each input
    #             'permute_identity' : permuted identity matrix - rerouting to next edge
    #             'random': random SM. Each elemen takes a value in [0,1)
    #             'isotropic_unitary': unitary isotropic SM, implemented through DFT matrix of correct dimension
    #             'random_unitary': random unitary SM
    #             'unitary_cyclic': unitary cyclic SM constructed through specifying phases of eigenvalues using 'delta'
    #                               kwarg
    #             'to_the_lowest_index': reroutes all energy to connected node of lowest index
    #             'custom' : Set a custom scattering matrix. Requires kwarg 'Smat' to be set

    #     scat_loss : float
    #         Specify scattering loss parameter for node, i.e. fraction of power lost
    #     **kwargs : Keyword arguments
    #         Extra keyword arguments required for specified type of scattering matrix:
    #             For scat_mat_type == 'custom':
    #                 kwargs['Smat'] defines custom scattering matrix
    #             For scat_mat_type == 'unitary_cyclic':
    #                 kwargs['delta'] is a vector define phase of eigenvalues of scattering matrix

    #     Returns
    #     -------
    #     None.

    #     """
    #     self.scat_mat_type = Smat_type
    #     self.scat_loss = scat_loss
    #     self.Smat_params = kwargs

    #     if scat_loss != 0 and self.node_type == 'internal':
    #         self.inwave_np  = np.array([0 + 0j]*(2*self.n_connect))
    #         self.outwave_np = np.array([0 + 0j]*(2*self.n_connect))
    #     else:
    #         self.inwave_np  = np.array([0 + 0j]*self.n_connect)
    #         self.outwave_np = np.array([0 + 0j]*self.n_connect)

    #     # scattering matrix for exit node is identity
    #     if self.node_type =='exit':
    #         self.S_mat  = np.identity(self.n_connect,dtype=np.complex_)
    #         self.iS_mat = np.identity(self.n_connect,dtype=np.complex_)
    #         return

    #     # scattering matrix for internal node
    #     if self.scat_mat_type == 'identity':
    #         self.S_mat = np.identity(self.n_connect,dtype=np.complex_)  #identity matrix
    #     elif self.scat_mat_type == 'random':
    #         self.S_mat = np.random.rand(self.n_connect,self.n_connect) #RANDOM SCATTEING MATRIX (nXn)
    #     elif self.scat_mat_type =='isotropic_unitary':
    #         self.S_mat  = (1/self.n_connect)**0.5*dft(self.n_connect)
    #     elif self.scat_mat_type == 'random_unitary':
    #         x = stats.unitary_group.rvs(self.n_connect)
    #         self.S_mat = np.dot(x, x.conj().T)
    #     elif self.scat_mat_type == 'permute_identity':
    #         mat = np.identity(self.n_connect,dtype=np.complex_)
    #         inds = [(i-1) % self.n_connect for i in range(0,self.n_connect)]
    #         self.S_mat = mat[:,inds]
    #     elif self.scat_mat_type == 'custom':
    #         mat = kwargs['Smat']
    #         # dimension checking
    #         if mat.shape  != (self.n_connect,self.n_connect):
    #             raise RuntimeError("Supplied scattering matrix is of incorrect dimensions: {} supplied,
    #                                  {} expected".format(mat.shape,(self.n_connect,self.n_connect)))
    #         else:
    #             self.S_mat = mat
    #     elif self.scat_mat_type == 'unitary_cyclic':
    #         l = np.exp(1j*kwargs['delta'][0:self.n_connect])
    #         s = np.matmul((1/self.n_connect)*dft(self.n_connect),l)
    #         self.S_mat = np.zeros(shape=(self.n_connect,self.n_connect),dtype=np.complex_)
    #         for jj in range(0,self.n_connect):
    #             self.S_mat[jj,:] = np.concatenate(
    #                 (s[(self.n_connect - jj ):self.n_connect],
    #                  s[0:self.n_connect - jj]))

    #     # define inverse scattering matrix
    #     self.iS_mat = np.linalg.inv(self.S_mat)

    #     ####  INTRODUCE INCOHERENT SCATTERING LOSS   #########
    #     if scat_loss != 0:
    #         S11 = (np.sqrt(1-scat_loss**2))*self.S_mat
    #         S12 = np.zeros(shape=(self.n_connect,self.n_connect),dtype=np.complex_)
    #         S21 = np.zeros(shape=(self.n_connect,self.n_connect),dtype=np.complex_)
    #         S22 = scat_loss*np.identity(self.n_connect,dtype=np.complex_)

    #         S_mat_top_row = np.concatenate((S11,S12),axis=1)
    #         S_mat_bot_row = np.concatenate((S21,S22),axis=1)
    #         self.S_mat= np.concatenate((S_mat_top_row,S_mat_bot_row),axis=0)

    #         iS11 = self.iS_mat / np.sqrt(1-scat_loss**2)
    #         iS12 = np.zeros(shape=(self.n_connect,self.n_connect),dtype=np.complex_)
    #         iS21 = np.zeros(shape=(self.n_connect,self.n_connect),dtype=np.complex_)
    #         iS22 = scat_loss*self.iS_mat / np.sqrt(1-scat_loss**2)

    #         iS_mat_top_row = np.concatenate((iS11,iS12),axis=1)
    #         iS_mat_bot_row = np.concatenate((iS21,iS22),axis=1)
    #         self.iS_mat= np.concatenate((iS_mat_top_row,iS_mat_bot_row),axis=0)

    def update(self, scat_loss=0, direction='forward'):
        """
        Update function for recursive method of calculating mode distributions.
        Updates output/input amplitudes at each connecting node according to scattering matrix

        Parameters
        ----------
        scat_loss : float, optional
            scattering loss parameter for node. Default = 0.0
        direction : str, optional
            Set to 'forward' or 'backwards' depending on recursive algorithm being used.
            The default is 'forward'.

        Returns
        -------
        None.

        """
        # check scattering matrix has been initialised
        if not hasattr(self, 'S_mat'):
            raise AttributeError('Nodal scattering matrix has not been initialised. '
                                 'Try assigning a scattering matrix first.')

        # check to see if we have created the list of sorted node IDs
        if not self.sorted_connected_nodes:
            self.sorted_connected_nodes = sorted(self.inwave.keys())

        if direction == 'forward':
            # create vector of incoming wave amplitude (in the order from small node number to large)
            for index, value in enumerate(self.sorted_connected_nodes):
                self.inwave_np[index] = self.inwave[value]

            # calculate outgoing wave amplitude using the scattering matrix
            outwave = np.matmul(self.S_mat, self.inwave_np).T

            # UPDATE EACH NODE OUTWAVE AMPLITUDE
            if self.node_type == 'internal':  # ONLY UPDATES INTERNAL NODES, NOT EXIT NODES,
                # WAVE IS NEVER REFLECTED BACK INTO SYSTEM AT EXIT NODE

                for [counter, u] in enumerate(self.sorted_connected_nodes):
                    # self.outwave.update({u:outwave[counter]})
                    self.outwave[u] = outwave[counter]
                    if scat_loss != 0:
                        # self.outwave.update({'loss%u'%u:outwave[counter + self.n_connect]})
                        self.outwave['loss%u' % u] = outwave[counter + self.n_connect]
        elif direction == 'backwards':
            # create vector of outgoing wave amplitude (in the order from small node number to large)
            for index, value in enumerate(self.sorted_connected_nodes):
                self.outwave_np[index] = self.outwave[value]

            # calculate incoming wave amplitude using the inverse scattering matrix
            inwave = np.matmul(self.iS_mat, self.outwave_np).T

            # UPDATE EACH NODE INPUT AMPLITUDE
            if self.node_type == 'internal':  # ONLY UPDATES INTERNAL NODES, NOT EXIT NODES,
                # WAVE IS NEVER REFLECTED BACK INTO SYSTEM AT EXIT NODE

                for [counter, u] in enumerate(self.sorted_connected_nodes):
                    # self.inwave.update({u:inwave[counter]})
                    self.inwave[u] = inwave[counter]
                    if scat_loss != 0:
                        # self.outwave.update({'loss%u'%u:inwave[counter + self.n_connect]})
                        self.outwave['loss%u' % u] = inwave[counter + self.n_connect]

        else:
            raise (ValueError, 'Unknown run direction type: must be "forwards" or "backwards"')

    ##########################
    # %% Save/Load Functions
    ##########################
    def node_to_dict(self, ):
        varnames = ['number', 'position', 'node_type', 'n_connect',
                    'sorted_connected_nodes', 'scat_mat_type', 'scat_loss',
                    'Smat_params', 'inwave', 'outwave', 'S_mat', 'iS_mat',
                    ]

        nodedict = dict((v, eval('self.' + v)) for v in varnames
                        if (hasattr(self, v) and eval('self.' + v) is not None))

        return nodedict

    def dict_to_node(self, nodedict):
        for key, val in nodedict.items():
            setattr(self, key, val)
