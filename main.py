# -*- coding: utf-8 -*-
"""
Created on Mon 18 Oct 2021

@author: Matthew Foreman

Draw an example of different networks and save them to the output folder

"""

import os


os.chdir("..")
import matplotlib.pyplot as plt
import numpy as np

from complexnetworklibrary import network_factory
from complexnetworklibrary.spec import NetworkSpec


np.random.seed(1)

spec = NetworkSpec(
    network_type="delaunay",
    num_internal_nodes=30,
    num_exit_nodes=3,
    network_shape="circular",
    network_size=1.0,
    exit_size=1.2,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.draw(show_indices=True)


incident_field = np.zeros(network.num_exit_nodes, dtype=np.complex128)
incident_field[0] = 1.0
out = network.scatter_iterative(incident_field, verbose=True)


network.plot_fields(title="Original network")

# Perturb network
perturbed_node = 8

variance = 0.01
num_changes = 5
max_val = 0.225
for i in range(num_changes):
    network.random_scaled_node_perturbation(perturbed_node, variance=variance)
    out = network.scatter_iterative(incident_field, verbose=True)
    network.plot_fields(
        highlight_nodes=[perturbed_node],
        title=f"Node {perturbed_node} perturbed, i = {i}",
    )

# Voronoi

spec = NetworkSpec(
    network_type="voronoi",
    num_internal_nodes=30,
    num_seed_nodes=30,
    num_exit_nodes=3,
    network_shape="circular",
    network_size=1.0,
    exit_size=1.1,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
