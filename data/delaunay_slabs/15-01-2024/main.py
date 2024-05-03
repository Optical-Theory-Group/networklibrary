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

# np.random.seed(1)

# Simulation parameters
num_realisations = 10**3

# Network parameters
node_density = 10 / (100e-6) ** 2
network_height = 100e-6
network_lengths = 10e-6 * np.array([i for i in range(1, 50 + 1)])
num_exit_nodes = 10

lam = 500e-9
n = 1.5
k0 = 2 * np.pi / lam

data = np.empty((0, num_realisations * num_exit_nodes))
mfps = np.empty((0, num_realisations))
normalised_lengths = np.empty((0, num_realisations))

for i, network_length in enumerate(network_lengths):
    print(network_length)
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

    tau_array = np.array([])
    normalised_lengths_array = np.array([])
    mfps_array = np.array([])

    for _ in tqdm(range(num_realisations)):
        network = network_factory.generate_network(spec)
        S = network.get_S_matrix_direct(n, k0)

        mfp = np.mean([l.length for l in network.internal_links])
        mfps_array = np.hstack((mfps_array, mfp))

        normalised_length = network_length / mfp
        normalised_lengths_array = np.hstack(
            (normalised_lengths_array, normalised_length)
        )

        t = S[num_exit_nodes : 2 * num_exit_nodes, 0:num_exit_nodes]
        u, s, vh = scipy.linalg.svd(t)
        tau = s * s
        tau_array = np.hstack((tau_array, tau))

    data = np.vstack((data, tau_array))
    normalised_lengths = np.vstack(
        (normalised_lengths, normalised_lengths_array)
    )
    mfps = np.vstack((mfps, mfps_array))

# Save
np.save("data.npy", data)
np.save("mfps.npy", mfps)
np.save("lengths.npy", normalised_lengths)
