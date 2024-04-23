import os
import scipy
import copy

import matplotlib.pyplot as plt
import numpy as np

from complex_network.networks import network_factory
from complex_network.networks.network_spec import NetworkSpec
from complex_network.perturbations.network_perturbator import (
    NetworkPerturbator,
)
from complex_network.perturbations import pole_finder

# Generate the random network
np.random.seed(1)
spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    external_offset=0.0,
    num_internal_nodes=3,
    num_external_nodes=2,
    network_size=500e-6,
    external_size=550e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.draw(show_indices=True)

lam = 500e-9
k0 = 2 * np.pi / lam
S_ee_before = network.get_S_ee(k0)
P_ei_before = network.get_P_ei(k0)
P_ii_before = network.get_P_ii(k0)
S_ii_before = network.get_S_ii()
internal_map_before = network.internal_scattering_map
external_map_before = network.external_scattering_map
scattering_slices_before = network.internal_scattering_slices

# Scattering
# n_mat_before = network.get_network_matrix(k0)
# incident_field = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# num_externals = network.num_external_nodes
# incident_vector = np.zeros((len(n_mat_before)), dtype=np.complex128)
# incident_vector[num_externals : 2 * num_externals] = incident_field
# out_vec = n_mat_before @ incident_vector
# network.set_network_fields(out_vec)
# network.plot_fields()
# print(network.get_node(9))

old_to_new_nodes, old_to_new_links = network.add_node_to_link(
    1, s=0.5, reflection_coefficient=0
)
new_to_old_nodes = {value: key for key, value in old_to_new_nodes.items()}
new_to_old_links = {value: key for key, value in old_to_new_links.items()}


network.draw(show_indices=True)
S_ee_after = network.get_S_ee(k0)
P_ei_after = network.get_P_ei(k0)
P_ii_after = network.get_P_ii(k0)
S_ii_after = network.get_S_ii()
internal_map_after = network.internal_scattering_map
external_map_after = network.external_scattering_map
scattering_slices_after = network.internal_scattering_slices

# # Check all nodes are same
# for node in network.internal_nodes:
#     if node.index == 15:
#         continue
#     print(node.index)
#     index = node.index
#     old_slice = scattering_slices_before[str(new_to_old_nodes[index])]
#     old_S_mat = S_ii_before[old_slice, old_slice]

#     new_slice = scattering_slices_after[str(index)]
#     new_S_mat = S_ii_after[new_slice, new_slice]
#     # print(np.allclose(old_S_mat, new_S_mat))

# # Check all internal links are same
# for link in network.internal_links:
#     if link.index == 33 or link.index == 34:
#         continue
#     index = link.index
#     node_one_new, node_two_new = link.sorted_connected_nodes
#     node_one_old = new_to_old_nodes[node_one_new]
#     node_two_old = new_to_old_nodes[node_two_new]

#     old_S_mat = P_ii_before[
#         internal_map_before[f"{node_one_old},{node_two_old}"],
#         internal_map_before[f"{node_two_old},{node_one_old}"],
#     ]
#     new_S_mat = P_ii_after[
#         internal_map_after[f"{node_one_new},{node_two_new}"],
#         internal_map_after[f"{node_two_new},{node_one_new}"],
#     ]
#     # print(np.allclose(old_S_mat, new_S_mat))

# # Check all internal links are same
# for link in network.external_links:

#     index = link.index
#     node_one_new, node_two_new = link.sorted_connected_nodes
#     node_one_old = new_to_old_nodes[node_one_new]
#     node_two_old = new_to_old_nodes[node_two_new]

#     old_S_mat = P_ii_before[
#         external_map_before[f"{node_two_old}"],
#         internal_map_before[f"{node_one_old},{node_two_old}"],
#     ]
#     new_S_mat = P_ii_after[
#         external_map_after[f"{node_two_new}"],
#         internal_map_after[f"{node_one_new},{node_two_new}"],
#     ]
#     # print(np.allclose(old_S_mat, new_S_mat))

# # Scattering
# n_mat_after = network.get_network_matrix(k0)
# incident_field = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
# num_externals = network.num_external_nodes
# incident_vector = np.zeros((len(n_mat_after)), dtype=np.complex128)
# incident_vector[num_externals : 2 * num_externals] = incident_field
# out_vec = n_mat_after @ incident_vector
# network.set_network_fields(out_vec)
# network.plot_fields()
# print(network.get_node(9))