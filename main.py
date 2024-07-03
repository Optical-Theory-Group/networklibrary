import functools
import numpy as np
import copy
from complex_network.networks import network_factory, network_perturbator
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks import pole_calculator

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
link_index = 9
network.add_segment_to_link(9, [0.4, 0.6])
network.draw(show_indices=True)
link_index = 34

# Perturb the link and set up matrices
perturbator = network_perturbator.NetworkPerturbator(network)
perturbator.perturb_segment_n(34, 0.1)
network = perturbator.perturbed_network

pole_guess = 1.142e7 - 600j
pole = pole_calculator.find_pole(network, pole_guess)
S = network.get_S_ee(pole)


print(network.get_S_ee(1))