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
from complex_network.networks.pole_calculator import (
    contour_integral,
    contour_integral_segment,
    find_pole,
    sweep,
    get_residue,
)

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
pole = 12532230.332102112 - 11.136143180724291j

k0 = 2 * np.pi / (500e-9)
S1 = network.get_S_ee(k0)


network.add_node_to_link(9, 0.5)
network.draw(show_indices=True)
S2 = network.get_S_ee(k0)
