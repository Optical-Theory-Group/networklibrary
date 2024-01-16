# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:44:38 2022

@author: Matthew Foreman, Niall Byrnes

Node class functions
"""

# setup code logging
import logging
import matplotlib.pyplot as plt
import numpy as np
from typing import Any
import logconfig
from complex_network.components.component import Component

logconfig.setup_logging()
logger = logging.getLogger(__name__)


class Node(Component):
    """
    Class that defines nodes in the network

    Attributes:
    ----------
    index:
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
        index: int,
        node_type: str,
        position: tuple[float, float],
        data: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(index=index, nature=node_type, data=data)
        self.position = np.array(position)

    @property
    def node_type(self):
        """Alias for nature"""
        return self.nature

    @node_type.setter
    def node_type(self, value):
        self.nature = value

    @node_type.setter
    def node_type(self, value):
        self.nature = value

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
    def x(self) -> float:
        return self.position[0]

    @property
    def y(self) -> float:
        return self.position[1]

    @staticmethod
    def get_default_values() -> dict[str, Any]:
        """Default values for the node"""
        default_values: dict[str, Any] = {
            "index": 0,
            "position": None,
            "node_type": "internal",
            "num_connect": 0,
            "sorted_connected_nodes": [],
            "sorted_connected_links": [],
            "inwave": {},
            "outwave": {},
            "inwave_np": np.zeros(0, dtype=np.complex128),
            "outwave_np": np.zeros(0, dtype=np.complex128),
            "S_mat_type": "COE",
            "S_mat_params": {},
            "S_mat": np.zeros(0, dtype=np.complex128),
            "iS_mat": np.zeros(0, dtype=np.complex128),
        }
        return default_values

    def draw(
        self,
        ax: plt.Axes,
        show_index: bool = False,
        show_exit_index: bool = False,
        color: str | None = None,
    ) -> None:
        """Draw node on figure"""
        if show_index:
            ax.text(self.x, self.y, self.index)

        if show_exit_index and self.node_type == "exit":
            ax.text(self.x, self.y, self.index)

        if color is not None:
            ax.plot(self.x, self.y, "o", color=color)
        elif self.node_type == "internal":
            ax.plot(self.x, self.y, "o", color="#9678B4")
        elif self.node_type == "exit":
            ax.plot(self.x, self.y, "o", color="#85C27F")
