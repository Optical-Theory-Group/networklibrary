# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:45:48 2022

@author: Matthew Foreman

Link class functions

"""

import numpy as np
from typing import Union
from .node import NODE

# setup code logging
import logging
import logconfig

logconfig.setup_logging()
logger = logging.getLogger(__name__)


class LINK:
    def __init__(self, node1: int = 0, node2: int = 1, distance: float = 0, k: Union[complex, float] = 1.0,
                 ni: Union[complex, float] = 1.0, linktype: str = 'internal', linkdict: Union[dict, None] = None):
        """
        Constructor/initialisation function for individual edge

        Parameters
        ----------
        node1, node2 : int
            ID numbers of nodes at ends of edge.
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
        self.link_type: str # string identify for link type
        self.node1: int  # unique numerical id for 1st connected node
        self.node2: int  # unique numerical id for 2nd connected node
        self.link_type: str  # 'internal' or 'exit' specifying nature of link
        self.distance: float  # length of link
        self.n: Union[float, complex]  # effective refractive index of link
        self.k: Union[float, complex]  # vacuum wavenumber of wave propagating in link
        self.inwave: np.ndarray[np.complex64]  # array of inwave amplitudes
        self.outwave: np.ndarray[np.complex64]  # array of outwave amplitudes
        self.S_mat: np.ndarray[np.complex64]  # propagation matrix
        self.iS_mat: np.ndarray[np.complex64]  # inverse propagation matrix

        for v, d in self.get_default_properties().items():
            setattr(self, v, d)

        if linkdict is not None:
            self.dict_to_link(linkdict)
        else:
            if node1 is None:
                raise TypeError("Node 1 ID must be supplied")
            if node2 is None:
                raise TypeError("Node 2 ID must be supplied")
            self.node1 = int(node1)
            self.node2 = int(node2)
            self.link_type = linktype
            self.reset_link(distance, k, ni)

    def __setattr__(self, name, value):
        # use some value checking on possible assigned values of class attributes
        if name == 'link_type':
            if value not in ['internal','exit']:
                raise AttributeError("Tried to set an invalid value for the link_type attribute")
        super().__setattr__(name, value)

    def reset_link(self, distance: float, k: Union[complex, float], ni: Union[complex, float]) -> None:
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

    def update(self, connected_node: list[NODE], direction: str = 'forward') -> None:
        """
        Update function for recursive method of calculating mode distributions.
        Updates output/input amplitudes at each connecting node according to propagation matrix of edg

        Parameters
        ----------
        connected_node : [NODE,...]
            List of connected node objects
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
    def link_to_dict(self, ) -> dict:
        # note we choose not to use __dict__ method to have greater more transparent control of what we are saving.
        varnames = self.get_default_properties().keys()

        linkdict = dict((v, eval('self.' + v)) for v in varnames
                        if hasattr(self, v))

        return linkdict

    def dict_to_link(self, linkdict: dict) -> None:
        varnames = self.get_default_properties().keys()
        keys = linkdict.keys()

        required_keys = ['node1', 'node2', 'link_type']

        for req in required_keys:
            if req not in keys:
                raise ValueError("Required node property '{}' not found in specification dictionary.".format(req))

        for v in varnames:
            if v in keys:
                setattr(self, v, linkdict[v])
            else:
                setattr(self, v, None)

    @staticmethod
    def get_default_properties() -> dict:
        default_values = {'node1': None,  # unique numerical id for 1st connected node
                          'node2': None,  # unique numerical id for 2nd connected node
                          'link_type': 'internal',  # 'internal' or 'exit' specifying nature of link
                          'distance': 0,  # length of link
                          'n': 1.0,  # effective refractive index of link
                          'k': 1,  # vacuum wavenumber of wave propagating in link
                          'inwave': np.array([0 + 0j, 0 + 0j]),  # array of inwave amplitudes
                          'outwave': np.array([0 + 0j, 0 + 0j]),  # array of outwave amplitudes
                          'S_mat': np.array([[0, 1 + 0j],  # propagation matrix
                                             [1 + 0j, 0]]),
                          'iS_mat': np.array([[0, 1 + 0j],  # inverse propagation matrix
                                              [1 + 0j, 0]])
                          }

        return default_values
