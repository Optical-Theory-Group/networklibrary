import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy

from complex_network.networks import network_factory, pole_finder
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
    network_size=500e-6,
    external_size=550e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.add_node_to_link(9, 0.1, 0.25)


perturbator = NetworkPerturbator(network)
perturbator.perturb_pseudonode_s(15, 0.5)

print(perturbator.unperturbed_network.get_node(15))
print(perturbator.unperturbed_network.get_link(33))
print(perturbator.unperturbed_network.get_link(34))


print(perturbator.perturbed_network.get_node(15))
print(perturbator.perturbed_network.get_link(33))
print(perturbator.perturbed_network.get_link(34))

s = network.get_wigner_smith_s(2*np.pi/(500e-9), 15)