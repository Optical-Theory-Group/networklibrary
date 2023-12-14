"""Module defining spec objects that used to specify properties of different
components of the networks.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Any


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
    num_exit_nodes: int
    network_shape: str
    network_size: float | tuple[float, float]
    exit_size: float
    node_S_mat_type: str
    node_S_mat_params: dict[str, Any]
