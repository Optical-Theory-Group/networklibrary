# -*- coding: utf-8 -*-
"""
Created on Mon 18 Oct 2021

@author: Matthew Foreman

Draw an example of different networks and save them to the output folder

"""

import os
import scipy

import matplotlib.pyplot as plt
import numpy as np

from complex_network import network_factory
from complex_network.spec import NetworkSpec
from tqdm import tqdm

np.random.seed(1)

spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    exit_offset=0.0,
    num_internal_nodes=5,
    num_exit_nodes=3,
    network_size=500e-6,
    exit_size=550e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)

network = network_factory.generate_network(spec)
network.draw(equal_aspect=True, show_indices=True)

n = 1.5
k0 = 500e-9

S = network.get_S_matrix_direct(n, k0)
network.scatter_direct(np.array([1.0, 0.0, 0.0]))

# We now examine the following loop:
# N1 -> L0 -> N3 -> L3 -> N4 -> L5 -> N1
L_0 = network.get_link(0).length
L_3 = network.get_link(3).length
L_5 = network.get_link(5).length
N_1 = network.get_node(1).outwave["3"] / network.get_node(1).inwave["4"]
N_3 = network.get_node(3).outwave["4"] / network.get_node(3).inwave["1"]
N_4 = network.get_node(4).outwave["1"] / network.get_node(4).inwave["3"]
print(L_0)
print(L_3)
print(L_5)
print(f"Node 1 factor: {N_1}") 
print(f"Node 3 factor: {N_3}")
print(f"Node 4 factor: {N_4}")

print(network.get_node(1).S_mat[2,3])


# A = -np.angle(S_N3 * S_N1 * S_N4) / (n * (L0 + L3 + L5))

# k0 = A
# in_S, in_P, P = network.get_pole_matrix(n, k0)
# N_1 = network.get_network_step_matrix(n, k0)
# N = network.get_network_matrix(n, k0)

# ims = np.linspace(-1.01, -1.03, 10**4)

# for im in ims:
#     k0 = A + 1j * im

#     S = network.get_S_matrix_direct(n, k0)
#     network.plot_fields()
#     prod = np.exp(1j * k0 * n * (L0 + L3 + L5)) * S_N1 * S_N3 * S_N4
#     if np.real(prod) > 1.0:
#         break
    # print(prod)


# # Print P
# for power in [1, 2, 3, 4, 5, 10, 50, 100]:
#     m = np.linalg.matrix_power(N_1, power)
#     max = np.max(np.abs(m))
#     print(f"Power = {power}, Max = {max}")

#     plt.figure()
#     plt.imshow(np.abs(m))

# # Get S from large power of N_1
# large_power = np.linalg.matrix_power(N_1, 10000)
# num_exit_nodes = network.num_exit_nodes
# S = large_power[0:num_exit_nodes, num_exit_nodes : 2 * num_exit_nodes]

# # From zeroing eigs
# S2 = network.get_S_matrix_direct(n, k0)
