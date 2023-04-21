# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:44:38 2022

@author: Matthew Foreman

Node class functions
"""

import numpy as np

# setup code logging
import logging
import logconfig

logconfig.setup_logging()
logger = logging.getLogger(__name__)

# restrict allowed values of type
class NodeType:

    def __init__(self, default):
        self.allowed = ['internal', 'exit']
        self.value = default

    def __set_name__(self, owner, name):
        self.name = "_" + name

    def __get__(self, instance, owner):
        return self.value

    def __set__(self, instance, value):
        if value not in self.allowed:
            msg = "Node type must be one of {}".format(self.allowed)
            logging.error(msg)
            raise ValueError(msg)
        self.value = value


class NODE:
    node_type = NodeType('internal')

    def __init__(self, number=0, pos=(0, 0), nodetype='internal', nodedict=None):
        """
        Constructor/initialisation function for individual node

        Parameters
        ----------
        number : int
            Unique id number for node.
        pos : (x,y) tuple
            Position of nodes.
        nodetype : string, optional
            Specifies whether node is an 'internal' or 'exit' node
        nodedict : supply dictionary to set node properties from stored/previous values
            Default is None
        """
        logging.debug("Initialising {} node #{} @ {}.".format(nodetype,number,pos))
        # initialise all class properties
        logging.debug("Setting default node properties.")
        for v, d in self.get_default_properties().items():
            setattr(self, v, d)

        if nodedict is not None:
            logging.debug("Setting node properties from supplied dictionary. \n{}".format(nodedict))
            self.dict_to_node(nodedict)
        else:
            self.number = int(number)
            self.position = np.array(pos)
            self.node_type = nodetype

    def degree(self):
        """
        Returns degree of node, i.e. the number of connecting edges
        """
        return self.n_connect

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
        # note we choose not to use __dict__ method to have greater more transparent control of what we are saving.
        varnames = self.get_default_properties().keys()

        nodedict = dict((v, eval('self.' + v)) for v in varnames
                        if (hasattr(self, v)))

        return nodedict

    def dict_to_node(self, nodedict):
        keys = nodedict.keys()
        for v,d in  self.get_default_properties().items():
            if v in keys:
                setattr(self, v, nodedict[v])
            else:
                setattr(self, v, d)

    @staticmethod
    def get_default_properties():
        return {'number': 0,                    # unique numerical id for node
                'position': (0,0),              # position of node (2d assumed)
                'node_type': 'internal',        # 'internal' or 'exit' specifying nature of node
                'n_connect': 0,                 # number of connecting links/edges
                'sorted_connected_nodes': [],   # list of ids of nodes that are connected. order matches that of S_mat
                'Smat_type': None,              # string identifier for nature of scattering matrix
                'scat_loss': None,              # parameter describing fractional scattering loss
                'Smat_params': {},              # dictionary of any additional parameters used to generate S_mat
                'inwave': {},                   # dictionary of inwave amplitudes {node id: amplitude, ... }
                'outwave': {},                  # dictionary of outwave amplitudes {node id: amplitude, ...,
                                                #                                   'loss node id': loss amplitude, ...}
                'inwave_np': None,              # array of inwave amplitudes
                'outwave_np': None,             # array of outwave amplitudes
                'S_mat': None,                  # numpy array specifying scattering matrix
                'iS_mat': None,                 # numpy array specifying inverse scattering matrix
                }
