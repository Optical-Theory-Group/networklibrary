"""Module defining spec objects that used to specify properties of different
components of the networks.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any
from complex_network.materials.material import Material
from complex_network.materials.dielectric import Dielectric


@dataclass
class NetworkSpec:
    """Parameters associated with the construction of the network. Note that
    values can

    Attributes:
    ----------
    network_type: str



    node_S_mat_params:
        S_mat_type: str
            Type of scattering matrix used. See get_S_mat in network_factory for
            details.
        scat_loss: float
            Specify scattering loss parameter for node, i.e. fraction of power
            lost from the network.
        subunitary_factor: float
            Used for CUE and COE cases.
        S_mat:
            Used in custom case.
        delta:
            Used in unitary_cyclic case

    """

    network_type: str
    num_internal_nodes: int
    num_external_nodes: int
    num_seed_nodes: int
    network_shape: str
    network_size: float | tuple[float, float]
    external_size: float
    external_offset: float
    node_S_mat_type: str
    node_S_mat_params: dict[str, Any]
    material: Material | None = None

    def __post_init__(self) -> None:
        if self.material is None:
            self.material = Dielectric("glass")
