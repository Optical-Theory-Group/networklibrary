import os


os.chdir("..")
import matplotlib.pyplot as plt
import numpy as np

from complex_network.networks import network_factory
from complex_network.networks.network_spec import NetworkSpec

np.random.seed(1)

spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    exit_offset=0.0,
    num_internal_nodes=15,
    num_exit_nodes=5,
    network_size=500e-6,
    exit_size=550e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)

n = 1.5
network = network_factory.generate_network(spec)
network.draw()

# Passive case
k = 2*np.pi/(500e-9)
network_matrix = network.get_network_matrix(1.5, k)
l, w = np.linalg.eig(network_matrix)

incident_field = np.zeros(len(network_matrix))
incident_field[9] = 1.0
outgoing_field = network_matrix@incident_field
network.reset_fields()
network.set_network_fields(outgoing_field)
network.plot_fields()

network.set_network_fields(w[:,-1])
network.plot_fields()

# Check orthogonality with eigenvectors
for i in range(len(network_matrix)):
    prod = np.dot(incident_field, w[:,i])
    print(f"{i}: {prod}")


# At the pole
pole = 12532234.746804317 - 18.467637263524797j
network_matrix = network.get_network_matrix(1.5, pole)
l, w = np.linalg.eig(network_matrix)
w_inv = np.linalg.inv(w)

incident_field = np.zeros(len(network_matrix))
incident_field[8] = 1.0
outgoing_field = network_matrix@incident_field
network.reset_fields()
network.set_network_fields(outgoing_field)
network.plot_fields()


# Check orthogonality with eigenvectors
for i in range(len(network_matrix)):
    prod = np.dot(incident_field, w[:,i])
    print(f"{i}: {prod}")

network.set_network_fields(w[:,-1])
network.plot_fields()



# incident_field = np.zeros(len(network_matrix))
# incident_field[5 + 0] = 1.0
# for i in range(len(w)):
#     prod = np.dot(incident_field, np.conj(w[:,i]))
#     print(f"Eigenvector {i} product with incident field is :{prod}")

# network.set_network_fields(w[:,151])
# network.plot_fields()


# Figure out what k should be
# link_number = 0
# link = network.get_link(link_number)
# node_one_index, node_two_index = link.node_indices
# node_one, node_two = network.get_node(node_one_index), network.get_node(
#     node_two_index
# )
# node_one_sorted = node_one.sorted_connected_nodes
# node_two_sorted = node_two.sorted_connected_nodes

# r_one = node_one.S_mat[
#     node_one_sorted.index(node_two_index),
#     node_one_sorted.index(node_two_index),
# ]
# r_two = node_two.S_mat[
#     node_two_sorted.index(node_one_index),
#     node_two_sorted.index(node_one_index),
# ]

# k = -np.log(r_one * r_two) / (2 * 1j * link.n * link.length)

