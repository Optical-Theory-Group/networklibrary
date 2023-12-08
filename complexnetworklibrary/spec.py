"""Module defining spec objects that used to specify properties of different
components of the networks.
"""

import numpy as np
from dataclasses import dataclass, fields


@dataclass
class NetworkSpec:
    """Parameters associated with the construction of the network. Note that
    values can 

    Attributes:
    ----------

    node_S_mat_type:
        Specifies type of scattering matrix to use. Options are:

        'identity':
            identity matrix - complete reflection at each input
        'permute_identity' :
            permuted identity matrix - rerouting to next edge
        'uniform':
            each element takes a value in [0,1)
        'isotropic_unitary':
            unitary isotropic SM, implemented through DFT matrix of correct
            dimension
        'COE' :
            drawn from circular orthogonal ensemble
        'CUE' :
            drawn from circular unitary ensemble
        'unitary_cyclic':
            unitary cyclic SM constructed through specifying phases of
            eigenvalues using 'delta'
        'to_the_lowest_index':
            reroutes all energy to connected node of lowest index
        'custom' :
            Set a custom scattering matrix. Requires kwarg 'S_mat' to be set

    node_scat_loss:
        Specify scattering loss parameter for node, i.e. fraction of power
        lost from the network.

    """

    # Scattering at the nodes
    node_S_mat_type: str = "COE"
    node_scat_loss: float = 0.0

    geometry_type: str = "delaunay"

    @property
    def attr_names(self) -> list[str]:
        return [f.name for f in fields(self)]
