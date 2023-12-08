# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:45:48 2022

@author: Matthew Foreman

Link class functions

"""

# setup code logging
import logging
from typing import Union
from complexnetworklibrary.spec import LinkSpec
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

    def __init__(
        self,
        node1: None | int = 0,
        node2: None | int = 1,
        link_type: str = "internal",
        link_spec: LinkSpec | None = None,
    ):
        self._validate_args(node1, node2, link_type)
        self.node1 = node1
        self.node2 = node2
        self._link_type = link_type

        # Set attributes from NodeSpec object
        if link_spec is None:
            link_spec = LinkSpec()
        for attr_name in LinkSpec.attr_names:
            setattr(self, attr_name, getattr(link_spec, attr_name))

    def _validate_args(
        self,
        node1: int | None,
        node2: int | None,
        link_type: str | None,
    ) -> None:
        """Check that args are given"""
        if node1 is None:
            raise ValueError(
                "Node 1 ID has not been supplied, but is required."
            )
        if node2 is None:
            raise ValueError(
                "Node 2 ID has not been supplied, but is required."
            )
        if link_type is None:
            raise ValueError(
                "Link type has not been supplied, but is required."
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

    @property
    def required_attr_names(self) -> list[str]:
        return ["node1", "node2", "link_type"]

    @property
    def attr_names(self) -> list[str]:
        """List of attributes for Node objects"""
        return self.required_attr_names + LinkSpec.attr_names

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
            v: getattr(self, v) for v in self.attr_names if hasattr(self, v)
        }

    def __str__(self):
        attr_values = [
            f"{attr}: {getattr(self, attr)}"
            for attr in self.attr_names
            if hasattr(self, attr)
        ]
        return ",\n".join(attr_values)
