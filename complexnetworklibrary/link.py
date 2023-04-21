# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:45:48 2022

@author: Matthew Foreman

Link class functions

"""

import numpy as np

# setup code logging
import logging
import logconfig

logconfig.setup_logging()
logger = logging.getLogger(__name__)

class LinkType:
    def __init__(self, default):
        self.allowed = ['internal', 'exit']
        self.value = default

    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        if value not in self.allowed:
            raise ValueError("Value must be one of {}".format(self.allowed))
        self.value = value


class LINK:
    link_type = LinkType('internal')

    def __init__(self, node1=0, node2=1, distance=0, k=1.0, ni=1.0, linktype='internal', linkdict=None):
        """
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
        linktype : str, optional
            Specifies whether edge is an 'internal' (default) or connects to an 'exit' node.
        linkdict : supply dictionary to set link properties from stored/previous values
            Default is None

        Returns
        -------
        None.

        """
        # initialise all class properties
        for v, d in self.get_default_properties().items():
            setattr(self, v, d)

        if linkdict is not None:
            self.dict_to_link(linkdict)
        else:
            self.node1 = int(node1) if node1 is not None else None
            self.node2 = int(node2) if node2 is not None else None
            self.link_type = linktype
            self.reset_link(distance, k, ni)

    def reset_link(self, distance, k, ni):
        """
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

        """
        self.distance = distance
        self.n = ni
        self.k = k

        self.S_mat = np.array([[0, np.exp(1j * self.n * self.k * distance)],
                               [np.exp(1j * self.n * self.k * distance), 0]])  # propagation matrix

        self.iS_mat = np.array([[0, np.exp(-1j * self.n * self.k * distance)],
                                [np.exp(-1j * self.n * self.k * distance), 0]])  # inverse propagation matrix

        self.inwave = np.array([0 + 0j, 0 + 0j])
        self.outwave = np.array([0 + 0j, 0 + 0j])

    def update(self, connected_node, direction='forward'):
        """
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

        """
        if direction == 'forward':
            for node in connected_node:
                if node.number == self.node1:
                    self.inwave[0] = node.outwave[self.node2]  # outwave to node2   node1 outwave towards node 2 b21
                elif node.number == self.node2:
                    self.inwave[1] = node.outwave[self.node1]  # outwave to node1   node2 outwave towards node 1 b12

            # calculate outgoing wave amplitude using the scattering matrix
            self.outwave = np.matmul(self.S_mat, self.inwave).T

            for node in connected_node:
                if node.number == self.node1:
                    node.inwave[self.node2] = self.outwave[0]  # inwave from node2    node 1 inwave from node 2 a21
                elif node.number == self.node2:
                    node.inwave[self.node1] = self.outwave[1]  # inwave from node1    node 2 inwave from node 1 a12
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
            raise (ValueError, 'Unknown run direction type: must be forwards of backwards')
            pass

    ##########################
    # %% Save/Load Functions
    ##########################
    def link_to_dict(self, ):
        # note we choose not to use __dict__ method to have greater more transparent control of what we are saving.
        varnames = self.get_default_properties().keys()

        linkdict = dict((v, eval('self.' + v)) for v in varnames
                        if hasattr(self, v))

        return linkdict

    def dict_to_link(self, linkdict):
        varnames = self.get_default_properties().keys()

        keys = linkdict.keys()
        for v in varnames:
            if v in keys:
                setattr(self, v, linkdict[v])
            else:
                setattr(self, v, None)

    @staticmethod
    def get_default_properties():
        return {'node1': 0,
                'node2': 1,
                'link_type': 'internal',
                'distance': 0,
                'n': 1.0,
                'k': 1,
                'inwave': np.array([0 + 0j, 0 + 0j]),
                'outwave': np.array([0 + 0j, 0 + 0j]),
                'S_mat': np.array([[0, 1 + 0j],
                                   [1 + 0j, 0]]),
                'iS_mat': np.array([[0, 1 + 0j],
                                    [1 + 0j, 0]])
                }
