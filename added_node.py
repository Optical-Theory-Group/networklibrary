import os
import scipy
import copy

import matplotlib.pyplot as plt
import numpy as np

from complex_network.networks import network_factory
from complex_network.networks.network_spec import NetworkSpec
from complex_network.perturbations.network_perturbator import (
    NetworkPerturbator,
)
from complex_network.perturbations import pole_finder

# Generate the random network
np.random.seed(1)
spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    external_offset=0.0,
    num_internal_nodes=15,
    num_external_nodes=5,
    network_size=500e-6,
    external_size=550e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.draw(show_indices=True)

old_to_new_nodes, old_to_new_links = network.add_node_to_link(9, 0.5, 1.0)

S_ee_after = network.get_S_ee(k0)
print(S_ee_after - S_ee_before)
