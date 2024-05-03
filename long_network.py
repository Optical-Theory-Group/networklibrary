# -*- coding: utf-8 -*-
"""
Created on Mon 18 Oct 2021

@author: Matthew Foreman

Draw an example of different networks and save them to the output folder

"""

import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from complex_network.networks import network_factory
from complex_network.networks.network_spec import NetworkSpec

np.random.seed(1)

# Simulation parameters
num_realisations = 10**0

# Network parameters
node_density = 10 / (100e-6) ** 2
network_height = 100e-6
network_lengths = 10e-6 * np.array([i for i in range(1, 50 + 1)])
network_length = network_lengths[-1]
num_exit_nodes = 10

lam = 500e-9
n = 1.5
k0 = 2 * np.pi / lam

# Num nodes based on network size
network_area = network_height * network_length
num_internal_nodes = 2 * num_exit_nodes + int(node_density * network_area)
network_offset = 10e-6

spec = NetworkSpec(
    network_type="delaunay",
    network_shape="slab",
    num_seed_nodes=0,
    exit_offset=network_offset,
    num_internal_nodes=num_internal_nodes,
    num_exit_nodes=num_exit_nodes,
    network_size=(network_length, network_height),
    exit_size=1.0,
    node_S_mat_type="COE",
    node_S_mat_params={},
)

for _ in tqdm(range(num_realisations)):
    network = network_factory.generate_network(spec)
    S = network.get_S_matrix_direct(n, k0)
    t = S[num_exit_nodes : 2 * num_exit_nodes, 0:num_exit_nodes]
    u, s, vh = scipy.linalg.svd(t)

    v = np.conj(vh.T)
    max_vec = v[:,0]
    augmented_vec = np.zeros((20),dtype=np.complex128)
    augmented_vec[0:10] = max_vec

    # network.scatter_iterative(augmented_vec)
    network.scatter_direct(augmented_vec)
    network.plot_fields(title="Eigenchannel")

    vec = np.random.randn(5)
    vec = vec / scipy.linalg.norm(vec)

    augmented_vec = np.zeros((20),dtype=np.complex128)
    augmented_vec[0:10] = vec

    network.scatter_direct(augmented_vec)
    network.plot_fields(title="Random input")

