"""Class module for network links."""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from complex_network.components.component import Component


class Link(Component):
    """Class that defines links/fibres in the network

    Attributes:
    ----------
    index:
        Unique integer id for the link
    node_indices:
        Tuple of indices of the nodes connected by the link
    link_type:
        string identify for link type, "internal" or "external
    length:
        length of link
    n:
        effective refractive index of link
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
    get_S:
        function that returns the propagation matrix
    get_S_inv:
        function that returns the inverse propagation matrix
    get_dS:
        function that returns the derivative of the propagation matrix"""

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
    def link_type(self) -> str:
        """Alias for nature"""
        return self.nature

    @link_type.setter
    def link_type(self, value) -> None:
        self.nature = value

    @property
    def power_diff(self) -> float:
        """Difference in power flowing in both directions."""
        return np.abs(
            np.abs(self.inwave_np[0]) ** 2 - np.abs(self.outwave_np[0]) ** 2
        )

    @property
    def power_direction(self) -> float:
        """Direction in which net power flows within the link.

        Output is 1 or -1"""
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
            "n": lambda k0: 1.0,
            "dn": lambda k0: 0.0,
            "Dn": 0.0,
            "material": None,
            "sorted_connected_nodes": [],
            "inwave": {},
            "outwave": {},
            "inwave_np": np.array([0 + 0j, 0 + 0j]),
            "outwave_np": np.array([0 + 0j, 0 + 0j]),
            "get_S": lambda k0: np.array([[0, 1 + 0j], [1 + 0j, 0]]),
            "get_S_inv": lambda k0: np.array([[0, 1 + 0j], [1 + 0j, 0]]),
            "get_dS": lambda k0: np.array([[0, 0], [0, 0]]),
        }
        return default_values

    def draw(
        self,
        ax: plt.Axes,
        node_1_pos: np.ndarray,
        node_2_pos: np.ndarray,
        show_index: bool = False,
        color: None = None,
    ) -> None:
        """Draw link on figure."""
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
            if self.link_type == "external":
                linecol = "#85C27F"
            else:
                linecol = "#9678B4"
        ax.plot(
            [node_1_x, node_2_x], [node_1_y, node_2_y], color=linecol, lw=1.0
        )
