import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

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
    external_size=110e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)


link_index = 34
network.add_segment_to_link(9, [0.4, 0.6])
network.draw(show_indices=True)

pole_guesses = 1.142e7 + np.array(
    [
        1750 - 175j,
        2200 - 50j,
        3800 - 275j,
        6000 - 150j,
        6500 - 225j,
        7800 - 125j,
        8200 - 250j,
    ]
)
poles = np.array(
    [pole_calculator.find_pole(network, guess) for guess in pole_guesses]
)

# perturbator = NetworkPerturbator(network)

# Dn_values = np.linspace(1e-6, 1e-4, 10)

# poles_dict, pole_shifts_dict = perturbator.track_pole_segment_n(
#     poles, link_index, Dn_values
# )

# Get background for zoomed in plot
# Broad sweep to find some of the poles
Dk0 = 1.1427e7 + 20
k0_centre = 1.1427e7 + 800

k0_min = k0_centre - Dk0 - 180j
k0_max = k0_centre + Dk0 - 130j
num_points = 1 * 10**2

zoom_x, zoom_y, zoom_data = pole_calculator.sweep(k0_min, k0_max, num_points, network)