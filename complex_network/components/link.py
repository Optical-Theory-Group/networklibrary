# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:45:48 2022

@author: Matthew Foreman

Link class functions

"""

# setup code logging
import logging
from typing import Any
import numpy as np
import matplotlib.pyplot as plt
import logconfig

from complex_network.components.component import Component

logconfig.setup_logging()
logger = logging.getLogger(__name__)


class Link(Component):
    """
    Class that defines links/fibres in the network

    Attributes:
    ----------
    index:
        Unique integer id for the link
    node_indices:
        Tuple of indices of the nodes connected by the link
    link_type:
        string identify for link type, "internal" or "exit
    length:
        length of link
    n:
        effective refractive index of link
    k0:
        vacuum wavenumber of wave propagating in link
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
        propagation matrix
    iS_mat:
        inverse propagation matrix
    """

    def __init__(
        self,
        index: int,
        link_type: str,
        node_indices: tuple[int, int],
        data: dict[str, Any] | None = None,
    ):
        super().__init__(index=index, nature=link_type, data=data)
        self.node_indices = node_indices

    @property
    def link_type(self):
        """Alias for nature"""
        return self.nature

    @link_type.setter
    def link_type(self, value):
        self.nature = value

    @property
    def power_diff(self):
        """Difference in power flowing in both directions"""
        return np.abs(
            np.abs(self.inwave_np[0]) ** 2 - np.abs(self.outwave_np[0]) ** 2
        )

    @property
    def power_direction(self):
        """Direction in which net power flows within the link"""
        return np.sign(
            np.abs(self.inwave_np[0]) ** 2 - np.abs(self.outwave_np[0]) ** 2
        )

    @staticmethod
    def get_default_values() -> dict[str, Any]:
        default_values: dict[str, Any] = {
            "index": 0,
            "node_indices": None,
            "link_type": "internal",
            "length": 0.0,
            "n": 1.0,
            "k0": 1.0,
            "sorted_connected_nodes": [],
            "inwave": {},
            "outwave": {},
            "inwave_np": np.array([0 + 0j, 0 + 0j]),
            "outwave_np": np.array([0 + 0j, 0 + 0j]),
            "S_mat": np.array([[0, 1 + 0j], [1 + 0j, 0]]),
            "iS_mat": np.array([[0, 1 + 0j], [1 + 0j, 0]]),
        }
        return default_values

    def update_wave_parameters(
        self, n: float | complex, k0: float | complex
    ) -> None:
        self.n = n
        self.k0 = k0

    def update_S_matrices(self) -> None:
        """Function to set the scattering matrix of the link"""

        self.S_mat = np.array(
            [
                [0, np.exp(1j * self.n * self.k0 * self.length)],
                [np.exp(1j * self.n * self.k0 * self.length), 0],
            ]
        )
        self.iS_mat = np.array(
            [
                [0, np.exp(-1j * self.n * self.k0 * self.length)],
                [np.exp(-1j * self.n * self.k0 * self.length), 0],
            ]
        )

    def draw(
        self,
        ax: plt.Axes,
        node_1_pos: np.ndarray,
        node_2_pos: np.ndarray,
        show_index: bool = False,
        color: None = None,
    ) -> None:
        """Draw link on figure"""
        node_1_x, node_1_y = node_1_pos[0], node_1_pos[1]
        node_2_x, node_2_y = node_2_pos[0], node_2_pos[1]

        if show_index:
            ax.text(
                (node_1_x + node_2_x) / 2,
                (node_1_y + node_2_y) / 2,
                self.index,
                color="red",
            )
        if color is not None:
            linecol = color
        else:
            if self.link_type == "exit":
                linecol = "#85C27F"
            else:
                linecol = "#9678B4"
        ax.plot(
            [node_1_x, node_2_x], [node_1_y, node_2_y], color=linecol, lw=1.0
        )
