# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:44:38 2022

@author: Matthew Foreman, Niall Byrnes

Node class functions
"""

# setup code logging
import logging

import numpy as np

import logconfig
from complexnetworklibrary.spec import NodeSpec

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
    num_connect:
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

    def __init__(
        self,
        number: int,
        position: tuple[float, float],
        node_type: str,
        num_connect: int = 0,
        
    ) -> None:
        # Tell logger mode was made from input vars
        logging.info("Initialising node...")
        logging.info(
            f"...from input arguments: {node_type} node #{number} @ {position}"
        )

        # Enforce mandatory arguments
        self._validate_args(number, position, node_type)
        self.number = number
        self.position = np.array(position)
        self._node_type = node_type


        self.links: list[int] = []

    def _validate_args(
        self,
        number: int | None,
        position: tuple[float, float] | None,
        node_type: str | None,
    ) -> None:
        """Check that args are given"""
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
        """Alias for num_connect"""
        return self.num_connect

    @property
    def required_attr_names(self) -> list[str]:
        return ["number", "position", "node_type"]

    @property
    def attr_names(self) -> list[str]:
        """List of attributes for Node objects"""
        return self.required_attr_names + NodeSpec.attr_names

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
            # Use S matrix to find outgoing waves at node
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
            # Use inverse S matrix to find incoming waves at node
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
            v: getattr(self, v) for v in self.attr_names if hasattr(self, v)
        }

    def __str__(self):
        """String representation of the object. Prints all attributes."""
        attr_values = [
            f"{attr}: {getattr(self, attr)}"
            for attr in self.attr_names
            if hasattr(self, attr)
        ]
        return ",\n".join(attr_values)
