import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from complex_network.materials.dielectric import Dielectric
from complex_network.networks import network_factory
from complex_network.networks.network_perturbator import NetworkPerturbator
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.pole_finder import (contour_integral,
                                                  contour_integral_segment,
                                                  find_pole, sweep)

np.random.seed(1)

spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    num_internal_nodes=15,
    num_external_nodes=5,
    network_size=500e-6,
    external_size=550e-6,
    external_offset=0.0,
    node_S_mat_type="COE",
    node_S_mat_params={},
    material=Dielectric("glass"),
)

network = network_factory.generate_network(spec)
network.draw(show_indices=True)
perturbator = NetworkPerturbator(network)

lam = 500e-9
k0 = 2 * np.pi / lam

# Perturb link
base_n = 1.5
link_index = 9
perturbator.perturb_link_n(link_index, 0.001)

print(perturbator.perturbed_network.get_link(10).n(k0))
print(perturbator.perturbed_network.get_link(9).n(k0))

perturbator.perturb_link_n(link_index, 0.001)

print(perturbator.perturbed_network.get_link(10).n(k0))
print(perturbator.perturbed_network.get_link(9).n(k0))