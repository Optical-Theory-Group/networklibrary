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
    num_internal_nodes=20,
    num_seed_nodes=20,
    num_exit_nodes=10,
    network_shape="circular",
    network_size=1.0,
    exit_offset=1.0,
    exit_size=1.2,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.draw(show_indices=True)
step = network.get_network_step_matrix()
plt.figure()
plt.imshow(np.abs(step))

S = network.get_network_matrix()
plt.figure()
plt.imshow(np.abs(S))



# spec = NetworkSpec(
#     network_type="delaunay",
#     num_internal_nodes=50,
#     num_exit_nodes=(5,5),
#     num_seed_nodes=20,
#     network_shape="slab",
#     network_size=(1.0, 1.0),
#     exit_size=5.0,
#     exit_offset=0.1,
#     node_S_mat_type="COE",
#     node_S_mat_params={},
# )
# network = network_factory.generate_network(spec)
# network.draw(show_indices=False)

# spec = NetworkSpec(
#     network_type="voronoi",
#     num_internal_nodes=100,
#     num_exit_nodes=5,
#     num_seed_nodes=25,
#     network_shape="circular",
#     network_size=1.0,
#     exit_offset=1.0,
#     exit_size=1.1,
#     node_S_mat_type="COE",
#     node_S_mat_params={},
# )
# np.random.seed(1)
# network = network_factory.generate_network(spec)
# network.draw(show_indices=False)
