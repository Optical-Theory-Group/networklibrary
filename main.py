# -*- coding: utf-8 -*-
"""
Created on Mon 18 Oct 2021

@author: Matthew Foreman

Draw an example of different networks and save them to the output folder

"""

import os

import matplotlib.pyplot as plt
import numpy as np

from complexnetworklibrary import network_factory
from complexnetworklibrary.spec import NetworkSpec


np.random.seed(0)

spec = NetworkSpec(
    network_type="delaunay",
    num_internal_nodes=3,
    num_exit_nodes=3,
    network_shape="circular",
    network_size=1.0,
    exit_size=1.2,
    node_S_mat_params={"S_mat_type": "COE"},
)
network = network_factory.generate_network(spec)
network.draw()

n = network.get_node(3)
n.inwave["-1"] = 1.0 + 0j
n.inwave_np = np.array([1.0 + 0j, 0.0])

for _ in range(10**4):
    network.scatter_step()
print(network.get_node(3).outwave_np)
print(network.get_node(4).outwave_np)
print(network.get_node(5).outwave_np)

for _ in range(10**4):
    network.scatter_step()
print(network.get_node(3).outwave_np)
print(network.get_node(4).outwave_np)
print(network.get_node(5).outwave_np)

a = np.vstack(
    (
        network.get_node(3).outwave_np[0],
        network.get_node(4).outwave_np[0],
        network.get_node(5).outwave_np[0],
    )
)
