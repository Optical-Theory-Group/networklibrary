# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:45:48 2022

@author: Matthew Foreman

Link class functions

"""

# setup code logging
import logging
from typing import Union

import numpy as np

import logconfig

from .node import Node

logconfig.setup_logging()
logger = logging.getLogger(__name__)


class Link:
    """
    Class that defines links/fibres in the network

    Attributes:
    ----------
    link_type:
        string identify for link type
    node1:
        unique numerical id for 1st connected node
    node2:
        unique numerical id for 2nd connected node
    length:
        length of link
    n:
        effective refractive index of link
    k0:
        vacuum wavenumber of wave propagating in link
    inwave_np:
        array of inwave amplitudes
    outwave_np:
        array of outwave amplitudes
    S_mat:
        propagation matrix
    iS_mat:
        inverse propagation matrix
    """

    _ATTR_NAMES = [
        "node1",
        "node2",
        "link_type",
        "length",
        "n",
        "k0",
        "inwave",
        "outwave",
        "S_mat",
        "iS_mat",
    ]

    def __init__(
        self,
        node1: None | int = 0,
        node2: None | int = 1,
        length: float = 0.0,
        k0: Union[complex, float] = 1.0,
        n: Union[complex, float] = 1.0,
        link_type: str = "internal",
        linkdict: Union[dict, None] = None,
    ):
        # Validate node IDs
        if node1 is None:
            raise ValueError("Node 1 ID must be supplied")
        if node2 is None:
            raise ValueError("Node 2 ID must be supplied")

        # Set default values
        # This will be overriden if from_spec method is used
        self._link_type: str = link_type
        self.node1: int = node1
        self.node2: int = node2
        self.length: float = length
        self.n: float | complex = n
        self.k0: float | complex = k0
        self.inwave_np: np.ndarray[np.complex64] = np.array([0 + 0j, 0 + 0j])
        self.outwave_np: np.ndarray[np.complex64] = np.array([0 + 0j, 0 + 0j])
        self.S_mat: np.ndarray[np.complex64] = np.array(
            [
                [0, np.exp(1j * n * k0 * length)],
                [np.exp(1j * n * k0 * length), 0],
            ]
        )
        self.iS_mat: np.ndarray[np.complex64] = np.array(
            [
                [0, np.exp(-1j * n * k0 * length)],
                [np.exp(-1j * n * k0 * length), 0],
            ]
        )

    @property
    def link_type(self):
        return self._link_type

    @link_type.setter
    def link_type(self, value):
        if value not in ["internal", "exit"]:
            raise ValueError(
                "Invalid value for link_type. Must be 'internal' or 'exit'."
            )
        self._link_type = value

    @classmethod
    def from_spec(cls, link_spec: dict):
        """Factory method where attributes are given via a dictionary"""
        # Create node with default attributes
        new_link = cls(
            node1=link_spec.get("node1"),
            node2=link_spec.get("node2"),
            length=link_spec.get("length"),
            k0=link_spec.get("k0"),
            n=link_spec.get("n"),
            link_type=link_spec.get("link_type"),
        )

        # Add on other properties in the node_spec
        for key, val in link_spec.items():
            setattr(new_link, key, val)

        return new_link

    def reset_link(
        self,
        length: float,
        k0: Union[complex, float],
        n: Union[complex, float],
    ) -> None:
        """Reset function to change wavevector, length, refractive index of
        edge. In and outwaves are also reset."""

        self.length = length
        self.n = n
        self.k0 = k0
        self.S_mat = np.array(
            [
                [0, np.exp(1j * n * k0 * length)],
                [np.exp(1j * n * k0 * length), 0],
            ]
        )
        self.iS_mat = np.array(
            [
                [0, np.exp(-1j * n * k0 * length)],
                [np.exp(-1j * n * k0 * length), 0],
            ]
        )
        self.inwave_np = np.array([0 + 0j, 0 + 0j])
        self.outwave_np = np.array([0 + 0j, 0 + 0j])

    def update(
        self, connected_nodes: list[Node], direction: str = "forward"
    ) -> None:
        """
        Update function for recursive method of calculating mode distributions.
        Updates output/input amplitudes at each connecting node according
        to propagation matrix of edg

        Parameters
        ----------
        connected_nodes : [NODE,...]
            List of connected node objects
        direction : str, optional
            Set to 'forward' or 'backwards' depending on recursive algorithm
            being used. The default is 'forward'.
        """
        if direction == "forward":
            self.outwave_np = np.matmul(self.S_mat, self.inwave_np).T
        elif direction == "backwards":
            self.inwave_np = np.matmul(self.iS_mat, self.outwave_np).T
        else:
            raise ValueError(
                "Unknown run direction type: must be forwards of backwards",
            )

    def to_dict(self) -> dict:
        """Return a dictionary of the link attributes"""
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
