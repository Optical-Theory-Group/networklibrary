"""Class module for network nodes."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from complex_network.components.component import Component


class Node(Component):
    """Class that defines nodes in the network

    Attributes:
    ----------
    index:
        Unique numerical id for node
    position:
        Coordinates of node position
    node_type:
        Nature of node ("internal" or "external")
    num_connect:
        number of connecting links/edges
    sorted_connected_nodes:
        list of ids of nodes that are connected.
        order matches that of S
    S_mat_type:
        string identifier for nature of scattering matrix
    scat_loss:
        parameter describing fractional scattering loss
    S_mat_params:
        dictionary of any additional parameters used to generate S
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
    S:
        numpy array specifying scattering matrix
    S_inv:
        numpy array specifying inverse scattering matrix
    dS:
        derivative of S with respect to k0"""

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
    def node_type(self) -> str:
        """Alias for nature."""
        return self.nature
    
    @node_type.setter
    def node_type(self, value) -> None:
        self.nature = value

    @property
    def isinternal(self) -> bool:
        """Return True if node is internal."""
        return self.nature == "internal"
    
    @property
    def isexternal(self) -> bool:
        """Return True if node is internal."""
        return self.nature == "external"
    
    @property
    def scat_loss(self) -> float:
        """Return the scattering loss at the node."""
        return self._scat_loss

    @scat_loss.setter
    def scat_loss(self, value) -> None:
        if value < 0 or value > 1:
            raise ValueError("scat_loss must be between 0 and 1.")
        self._scat_loss = value

    @property
    def degree(self) -> int:
        """Alias for num_connect."""
        return self.num_connect

    @property
    def x(self) -> float:
        """x coordinate of the node."""
        return self.position[0]

    @property
    def y(self) -> float:
        """y coordinate of the node."""
        return self.position[1]
    @staticmethod
    def get_default_S(self, k0: complex) -> np.ndarray:
        """Default scattering matrix for the node."""
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    @staticmethod
    def get_default_S_inv(self, k0: complex) -> np.ndarray:
        """Default inverse scattering matrix for the node."""
        return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    @staticmethod
    def get_default_dS(self, k0: complex) -> np.ndarray:
        """Default derivative of the scattering matrix for the node."""
        return np.zeros((2, 2), dtype=np.complex128)

    @staticmethod
    def get_default_values() -> dict[str, Any]:
        """Default values for the node."""
        default_values: dict[str, Any] = {
            "index": 0,
            "is_perturbed": False,
            "perturbation_data": {},
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
            "get_S": Node.get_default_S,
            "get_S_inv": Node.get_default_S_inv,
            "get_dS": Node.get_default_dS,
        }
        return default_values

    def draw(
        self,
        ax: plt.Axes,
        show_index: bool = False,
        show_external_index: bool = False,
        show_internal_index: bool = False,
        color: str | None = None,
        markersize: float = 6.0,
    ) -> None:
        """Draw node on figure.

        Should be provided with a figure axis. Primarily called by the network
        class's draw method."""
        if show_index:
            ax.text(self.x, self.y, self.index)

        if show_external_index and self.node_type == "external":
            ax.text(self.x, self.y, self.index)

        if show_internal_index and self.node_type == "internal":
            ax.text(self.x, self.y, self.index)

        if color is not None:
            ax.plot(self.x, self.y, "o", color=color, markersize=markersize)
        elif self.node_type == "internal":
            ax.plot(
                self.x, self.y, "o", color="#9678B4", markersize=markersize
            )
        elif self.node_type == "external":
            ax.plot(
                self.x, self.y, "o", color="#85C27F", markersize=markersize
            )
