# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:44:38 2022

@author: Matthew Foreman

Node class functions
"""

import numpy as np

from typing import Union, TypedDict

# setup code logging
import logging
import logconfig

logconfig.setup_logging()
logger = logging.getLogger(__name__)


class NODE:
    def __init__(
        self,
        number: int = None,
        pos: tuple[float, float] = None,
        nodetype: str = "internal",
        nodedict: Union[dict, None] = None,
    ) -> None:
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
        logging.info("Initialising node...")
        # initialise class properties
        self.node_type: str  # string identify for type of node
        self.number: int  # unique numerical id for node
        self.position: tuple[float, float]  # position of node (2d assumed)
        self.node_type: str  # 'internal' or 'exit' specifying nature of node
        self.n_connect: int  # number of connecting links/edges
        self.sorted_connected_nodes: list[
            int
        ]  # list of ids of nodes that are connected. order matches that of S_mat
        self.Smat_type: Union[
            None, str
        ]  # string identifier for nature of scattering matrix
        self.scat_loss: float  # parameter describing fractional scattering loss
        self.Smat_params: Union[
            None, dict
        ]  # dictionary of any additional parameters used to generate S_mat
        self.inwave: Union[
            None, dict[int, Union[float, complex]]
        ]  # dictionary of inwave amplitudes
        #                                                               {node id: amplitude, ... }
        self.outwave: Union[
            None, dict[Union[str, int], Union[float, complex]]
        ]  # dictionary of outwave amplitudes
        #                                           {node id: amplitude, ..., 'loss node id: loss amplitude, ...}
        self.inwave_np: Union[
            None, np.ndarray[np.complex64]
        ]  # array of inwave amplitudes
        self.outwave_np: Union[
            None, np.ndarray[np.complex64]
        ]  # array of outwave amplitudes
        self.S_mat: Union[
            None, np.ndarray[np.complex64]
        ]  # numpy array specifying scattering matrix
        self.iS_mat: Union[
            None, np.ndarray[np.complex64]
        ]  # numpy array specifying inverse scattering matrix

        for v, d in self.get_default_properties().items():
            setattr(self, v, d)

        if nodedict is not None:
            logging.info("...from supplied dictionary. \n{}".format(nodedict))
            self.dict_to_node(nodedict)
        else:
            if number is None:
                raise TypeError(
                    "Node ID number has not been supplied, but is required"
                )
            if pos is None:
                raise TypeError(
                    "Node position has not been supplied, but is required"
                )

            logging.info(
                "...from input arguments: {} node #{} @ {}.".format(
                    nodetype, number, pos
                )
            )
            self.number = int(number)
            self.position = np.array(pos)
            self.node_type = nodetype

    def __setattr__(self, name, value):
        # use some value checking on possible assigned values of class attributes
        if name == "node_type":
            if value not in ["internal", "exit"]:
                raise AttributeError(
                    "Tried to set an invalid value for the node_type attribute"
                )
        if name == "scat_loss":
            if value < 0 or value > 1:
                raise AttributeError(
                    "Tried to set an invalid value for the scat_loss attribute. Most lie between 0 and 1"
                )
        super().__setattr__(name, value)

    def degree(self) -> int:
        """
        Returns degree of node, i.e. the number of connecting edges
        """
        return self.n_connect

    def update(self, scat_loss: float = 0, direction: str = "forward") -> None:
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
        logging.info("Updating node using {} algorithm".format(direction))
        # check to see if we have created the list of sorted node IDs
        if not self.sorted_connected_nodes:
            self.sorted_connected_nodes = sorted(self.inwave.keys())

        if direction == "forward":
            # create vector of incoming wave amplitude (in the order from small node number to large)
            for index, value in enumerate(self.sorted_connected_nodes):
                self.inwave_np[index] = self.inwave[value]

            # calculate outgoing wave amplitude using the scattering matrix
            outwave = np.matmul(self.S_mat, self.inwave_np).T

            # UPDATE EACH NODE OUTWAVE AMPLITUDE
            if (
                self.node_type == "internal"
            ):  # ONLY UPDATES INTERNAL NODES, NOT EXIT NODES,
                # WAVE IS NEVER REFLECTED BACK INTO SYSTEM AT EXIT NODE

                for [counter, u] in enumerate(self.sorted_connected_nodes):
                    # self.outwave.update({u:outwave[counter]})
                    self.outwave[u] = outwave[counter]
                    if scat_loss != 0:
                        # self.outwave.update({'loss%u'%u:outwave[counter + self.n_connect]})
                        self.outwave["loss%u" % u] = outwave[
                            counter + self.n_connect
                        ]
        elif direction == "backward":
            # create vector of outgoing wave amplitude (in the order from small node number to large)
            for index, value in enumerate(self.sorted_connected_nodes):
                self.outwave_np[index] = self.outwave[value]

            # calculate incoming wave amplitude using the inverse scattering matrix
            inwave = np.matmul(self.iS_mat, self.outwave_np).T

            # UPDATE EACH NODE INPUT AMPLITUDE
            if (
                self.node_type == "internal"
            ):  # ONLY UPDATES INTERNAL NODES, NOT EXIT NODES,
                # WAVE IS NEVER REFLECTED BACK INTO SYSTEM AT EXIT NODE

                for [counter, u] in enumerate(self.sorted_connected_nodes):
                    # self.inwave.update({u:inwave[counter]})
                    self.inwave[u] = inwave[counter]
                    if scat_loss != 0:
                        # self.outwave.update({'loss%u'%u:inwave[counter + self.n_connect]})
                        self.outwave["loss%u" % u] = inwave[
                            counter + self.n_connect
                        ]

        else:
            raise (
                ValueError,
                'Unknown run direction type: must be "forward" or "backward"',
            )

    ##########################
    # %% Save/Load Functions
    ##########################
    def node_to_dict(self) -> dict:
        # note we choose not to use __dict__ method to have greater more transparent control of what we are saving.
        varnames = self.get_default_properties().keys()

        nodedict = dict(
            (v, eval("self." + v)) for v in varnames if (hasattr(self, v))
        )

        return nodedict

    def dict_to_node(self, nodedict: dict) -> None:
        # check for required keys
        keys = nodedict.keys()
        required_keys = ["number", "position", "node_type"]

        for req in required_keys:
            if req not in keys:
                raise ValueError(
                    "Required node property '{}' not found in specification dictionary.".format(
                        req
                    )
                )

        for v, d in self.get_default_properties().items():
            if v in keys:
                setattr(self, v, nodedict[v])
            else:
                setattr(self, v, d)

    @staticmethod
    def get_default_properties():
        default_values = {
            "number": None,  # unique numerical id for node
            "position": None,  # position of node (2d assumed)
            "node_type": "internal",  # 'internal' or 'exit' specifying nature of node
            "n_connect": 0,  # number of connecting links/edges
            "sorted_connected_nodes": [],
            # list of ids of nodes that are connected. order matches that of S_mat
            "Smat_type": None,  # string identifier for nature of scattering matrix
            "scat_loss": 0,  # parameter describing fractional scattering loss
            "Smat_params": {},  # dictionary of any additional parameters used to generate S_mat
            "inwave": {},  # dictionary of inwave amplitudes {node id: amplitude, ... }
            "outwave": {},  # dictionary of outwave amplitudes {node id: amplitude, ...,
            #                                   'loss node id': loss amplitude, ...}
            "inwave_np": None,  # array of inwave amplitudes
            "outwave_np": None,  # array of outwave amplitudes
            "S_mat": None,  # numpy array specifying scattering matrix
            "iS_mat": None,  # numpy array specifying inverse scattering matrix
        }

        return default_values
