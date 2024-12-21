import copy
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm
import matplotlib.colors as mcolors

from complex_network.materials.dielectric import Dielectric
from complex_network.networks import network_factory, pole_calculator
from complex_network.networks.network_perturbator import NetworkPerturbator
from complex_network.networks.network_spec import NetworkSpec

# Generate the random network
np.random.seed(1)
spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    external_offset=0.0,
    num_internal_nodes=15,
    num_external_nodes=5,
    network_size=100e-6,
    external_size=120e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.translate_node(10, np.array([0.00002, -0.00001]))
network.translate_node(6, np.array([0.0000, 0.00001]))
network.add_segment_to_link(9, [0.4, 0.6])
network.draw(show_indices=True)
# network.draw(equal_aspect=True, show_indices=True)

# Get poles
poles = np.array(
    [
        (11420952.63744201 - 256.54478745j),
        (11421630.55338267 - 92.24831323j),
        (11422629.93980187 - 480.80052417j),
        (11422817.89861903 - 61.67705531j),
        (11422900.58183313 - 1772.93556847j),
        (11423917.22089518 - 237.09156394j),
        (11424869.53847224 - 409.86933413j),
        (11425979.88014366 - 301.38740088j),
        (11427460.63433062 - 849.3044861j),
        (11428729.762673 - 364.45590138j),
    ]
)
poles = sorted(poles, key=lambda x: -x.imag)
# loops = [1,3,5,6]

# Loop poles
for i, pole in enumerate(np.array(poles)[[0, 2, 5]]):
    incident_field = np.array([1, 0, 0, 0, 0])
    k0 = pole
    network.scatter_direct(incident_field, k0)
    # title = f"{i}: Loop" if i in loops else f"{i}: Non-loop"
    network.plot_internal(
        k0,
        title=f"{i}",
        lw=2.0,
    )

# l = network.get_link(32)
# phase = poles[2] * l.n(poles[2]) * l.length
