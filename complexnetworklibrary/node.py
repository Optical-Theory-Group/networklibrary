# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:44:38 2022

@author: Matthew Foreman

Node class functions
"""

# setup code logging
import logging

import numpy as np

import logconfig

logconfig.setup_logging()
logger = logging.getLogger(__name__)


class Node:
    """
    Class that defines nodes in the network

    Attributes:
    ----------
    number:
        Unique numerical id for node
    position:
        Coordinates of node position
    node_type:
        Nature of node ("internal" or "exit")
    n_connect:
        number of connecting links/edges
    sorted_connected_nodes:
        list of ids of nodes that are connected.
        order matches that of S_mat
    S_mat_type:
        string identifier for nature of scattering matrix
    scat_loss:
        parameter describing fractional scattering loss
    S_mat_params:
        dictionary of any additional parameters used to generate S_mat
    inwave:
        dictionary of inwave amplitudes
        {node id: amplitude, ... }
    outwave:
        dictionary of outwave amplitudes
        {node id: amplitude, ..., 'loss node id: loss amplitude, ...}
    inwave_np:
        array of inwave amplitudes
    outwave_np:
        array of outwave amplitudes
    S_mat:
        numpy array specifying scattering matrix
    iS_mat:
        numpy array specifying inverse scattering matrix
    """

    _ATTR_NAMES = [
        "number",
        "position",
        "node_type",
        "n_connect",
        "sorted_connected_nodes",
        "S_mat_type",
        "scat_loss",
        "S_mat_params",
        "inwave",
        "outwave",
        "inwave_np",
        "outwave_np",
        "S_mat",
        "iS_mat",
    ]

    def __init__(
        self,
        number: int | None = None,
        position: tuple[float, float] | None = None,
        node_type: str = "internal",
    ) -> None:
        logging.info("Initialising node...")

        # Check required args
        if number is None:
            raise ValueError(
                "Node ID number has not been supplied, but is required."
            )
        if position is None:
            raise ValueError(
                "Node position has not been supplied, but is required."
            )
        if node_type is None:
            raise ValueError(
                "Node type has not been supplied, but is required."
            )

        # Tell logger mode was made from input vars
        logging.info(
            f"...from input arguments: {node_type} node #{number} @ {position}"
        )

        # Set default values
        # These are overriden if from_spec factory method is used
        self.number: int = int(number)
        self.position: tuple[float, float] = np.array(position)
        self._node_type: str = node_type
        self.n_connect: int = 0
        self.sorted_connected_nodes: list[int] = []
        self.S_mat_type: None | str = None
        self._scat_loss: float = 0.0
        self.S_mat_params: None | dict = {}
        self.inwave: None | dict[str | int, float | complex] = {}
        self.outwave: None | dict[str | int, float | complex] = {}
        self.inwave_np: None | np.ndarray[np.complex64] = None
        self.outwave_np: None | np.ndarray[np.complex64] = None
        self.S_mat: None | np.ndarray[np.complex64] = None
        self.iS_mat: None | np.ndarray[np.complex64] = None

    @property
    def node_type(self):
        return self._node_type

    @node_type.setter
    def node_type(self, value):
        if value not in ["internal", "exit"]:
            raise ValueError(
                "Invalid value for node_type. Must be 'internal' or 'exit'."
            )
        self._node_type = value

    @property
    def scat_loss(self):
        return self._scat_loss

    @scat_loss.setter
    def scat_loss(self, value):
        if value < 0 or value > 1:
            raise ValueError("scat_loss must be between 0 and 1.")
        self._scat_loss = value

    @property
    def degree(self) -> int:
        """Alias for n_connect"""
        return self.n_connect

    @classmethod
    def from_spec(cls, node_spec: dict):
        """Factory method where attributes are given via a dictionary"""
        # Create node with default attributes
        new_node = cls(
            number=node_spec.get("number"),
            position=node_spec.get("position"),
            node_type=node_spec.get("node_type"),
        )

        # Add on other properties in the node_spec
        for key, val in node_spec.items():
            setattr(new_node, key, val)

        return new_node

    def update(self, direction: str = "forward") -> None:
        """
        Update function for recursive method of calculating mode distributions.
        Updates output/input amplitudes at each connecting node according to
        scattering matrix

        Parameters
        ----------
        direction : str, optional
            Set to 'forward' or 'backwards' depending on recursive algorithm
            being used. The default is 'forward'.
        """

        logging.info("Updating node using {} algorithm".format(direction))

        # Don't update exit nodes!
        if self.node_type == "exit":
            return

        # check to see if we have created the list of sorted node IDs
        if not self.sorted_connected_nodes:
            self.sorted_connected_nodes = sorted(self.inwave.keys())

        if direction == "forward":
            outwave_np = np.matmul(self.S_mat, self.inwave_np).T
            outwave = {
                node_id: val
                for node_id, val in zip(
                    self.sorted_connected_nodes, outwave_np
                )
            }
            self.outwave = outwave
            self.outwave_np = outwave_np
        elif direction == "backward":
            inwave_np = np.matmul(self.iS_mat, self.outwave_np).T
            inwave = {
                node_id: val
                for node_id, val in zip(self.sorted_connected_nodes, inwave_np)
            }

            self.inwave = inwave
            self.inwave_np = inwave_np
        else:
            raise ValueError(
                'Unknown run direction type: must be "forward" or "backward"'
            )

    def to_dict(self) -> dict:
        """Return a dictionary of the node attributes"""
        return {
            v: getattr(self, v) for v in self._ATTR_NAMES if hasattr(self, v)
        }

    def __str__(self):
        attr_values = [
            f"{attr}: {getattr(self, attr)}"
            for attr in self._ATTR_NAMES
            if hasattr(self, attr)
        ]
        return ",\n".join(attr_values)
