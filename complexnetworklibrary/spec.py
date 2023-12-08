"""Module defining spec objects that used to specify properties of different
components of the networks.
"""

import numpy as np
import dataclasses
from dataclasses import dataclass, field, fields


@dataclass
class Spec:
    """Base spec class.

    Has a few useful methods.
    """

    @property
    def attr_names(self) -> list[str]:
        return [field.name for field in fields(self)]


@dataclass
class NodeSpec(Spec):
    """Parameters for creating node objects.

    See node.py documentation for more information.
    """

    n_connect: int = 0
    sorted_connected_nodes: list[int] = field(default_factory=list)
    S_mat_type: None | str = None
    scat_loss: float = 0.0
    S_mat_params: dict = field(default_factory=dict)
    inwave: dict[str | int, float | complex] = field(default_factory=dict)
    outwave: dict[str | int, float | complex] = field(default_factory=dict)
    inwave_np: None | np.ndarray[np.complex64] = None
    outwave_np: None | np.ndarray[np.complex64] = None
    S_mat: None | np.ndarray[np.complex64] = None
    iS_mat: None | np.ndarray[np.complex64] = None


@dataclass
class LinkSpec(Spec):
    """Parameters for creating link objects.

    See link.py documentation for more information.
    """

    length: float = 0.0
    n: float | complex = 1.0
    k0: float | complex = 1.0
    inwave_np: np.ndarray[np.complex64] = np.array(
        [[0.0 + 0.0 * 1j, 0.0 + 0.0 * 1j]]
    )
    outwave_np: np.ndarray[np.complex64] = np.array(
        [[0.0 + 0.0 * 1j, 0.0 + 0.0 * 1j]]
    )
    S_mat: np.ndarray[np.complex64] | None = None
    iS_mat: np.ndarray[np.complex64] | None = None

    def __post_init__(self):
        """Calculate S matrices from the other paramteres if not given"""
        if self.S_mat is None:
            self.S_mat = np.array(
                [
                    [0, np.exp(1j * self.n * self.k0 * self.length)],
                    [np.exp(1j * self.n * self.k0 * self.length), 0],
                ]
            )
        if self.iS_mat is None:
            self.iS_mat = np.array(
                [
                    [0, np.exp(-1j * self.n * self.k0 * self.length)],
                    [np.exp(-1j * self.n * self.k0 * self.length), 0],
                ]
            )


@dataclass
class NetworkSpec(Spec):
    """Parameters for creating network objects.

    See network.py documentation for more information.
    """

    n_connect: int = 0
    sorted_connected_nodes: list[int] = field(default_factory=list)
    S_mat_type: None | str = None
    scat_loss: float = 0.0
    S_mat_params: dict = field(default_factory=dict)
    inwave: dict[str | int, float | complex] = field(default_factory=dict)
    outwave: dict[str | int, float | complex] = field(default_factory=dict)
    inwave_np: None | np.ndarray[np.complex64] = None
    outwave_np: None | np.ndarray[np.complex64] = None
    S_mat: None | np.ndarray[np.complex64] = None
    iS_mat: None | np.ndarray[np.complex64] = None
