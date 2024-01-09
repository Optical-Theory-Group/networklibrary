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
    num_internal_nodes=100,
    num_exit_nodes=20,
    network_shape="circular",
    network_size=1.0,
    exit_size=1.2,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.draw(show_indices=False)

spec = NetworkSpec(
    network_type="delaunay",
    num_internal_nodes=100,
    num_exit_nodes=10,
    network_shape="slab",
    network_size=(30.0, 20.0),
    exit_size=1.0,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.draw(show_indices=False)
