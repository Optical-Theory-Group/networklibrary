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
from complex_network.networks.pole_calculator import (contour_integral,
                                                      contour_integral_segment,
                                                      find_pole, get_residue,
                                                      sweep)
from complex_network.scattering_matrices import node_matrix

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
# pole = 12532230.332102112 - 11.136143180724291j

# k0 = 2 * np.pi / (500e-9)
# S1 = network.get_S_ee(k0)


# network.add_node_to_link(9, 0.5)
# network.draw(show_indices=True)

k0 = 2 * np.pi / (500e-9)
fractional_positions = [0.4, 0.6]
link_index = 9

S1 = network.get_S_ee(k0)


fractional_positions = np.sort(fractional_positions)
s1, s2 = fractional_positions

original_link = network.get_link(link_index)
node_one_index, node_two_index = original_link.node_indices
node_one = network.get_node(node_one_index)
node_two = network.get_node(node_two_index)
node_one_position = node_one.position
node_two_position = node_two.position

# Work out the new node positions
# This is so we can find them after all the index relabeling
first_node_position = node_one.position + s1 * (
    node_two.position - node_one.position
)
second_node_position = node_one.position + s2 * (
    node_two.position - node_one.position
)

# Add first node and find its index
network.add_node_to_link(link_index, s1)
first_node_index = network.get_node_by_position(first_node_position).index

# Find the link that connects the new node and node_two
# Note: node_two_index may have changed!
node_two_index = network.get_node_by_position(node_two_position).index
second_link_index = network.get_link_by_node_indices(
    (first_node_index, node_two_index)
).index

# Add second node and find its index
# Ratio is its fractional position along the new link
ratio = (s2 - s1) / (1 - s1)
network.add_node_to_link(second_link_index, ratio)
second_node_index = network.get_node_by_position(second_node_position).index

# Get all the new nodes and links
node_one = network.get_node_by_position(node_one_position)
first_node = network.get_node_by_position(first_node_position)
second_node = network.get_node_by_position(second_node_position)
node_two = network.get_node_by_position(node_two_position)

link_one = network.get_link_by_node_indices((node_one.index, first_node.index))
link_two = network.get_link_by_node_indices(
    (second_node.index, node_two.index)
)
middle_link = network.get_link_by_node_indices(
    (first_node_index, second_node_index)
)

# Set up segment node scattering matrices
# We need to ensure that the scattering matrix is done in the right
# order. Its order is determined by the numerical order of the
# sorted connected nodes. First and second here are with respect to
# the scattering matrix order
sorted_connected_nodes = first_node.sorted_connected_nodes
first_link = (
    link_one if node_one.index == sorted_connected_nodes[0] else middle_link
)
second_link = middle_link if first_link.index == link_one.index else link_one
perturbed_link_number = 1 if first_link.index == middle_link.index else 2

first_node.get_S = node_matrix.get_S_fresnel_closure(first_link, second_link)
first_node.get_S_inv = node_matrix.get_S_fresnel_inverse_closure(
    first_link, second_link
)
first_node.get_dS = node_matrix.get_S_fresnel_derivative_closure(
    first_link, second_link, perturbed_link_number
)

# Do the other one
sorted_connected_nodes = second_node.sorted_connected_nodes
first_link = (
    link_two if node_two.index == sorted_connected_nodes[0] else middle_link
)
second_link = middle_link if first_link.index == link_two.index else link_two
perturbed_link_number = 1 if first_link.index == middle_link.index else 2

second_node.get_S = node_matrix.get_S_fresnel_closure(first_link, second_link)
second_node.get_S_inv = node_matrix.get_S_fresnel_inverse_closure(
    first_link, second_link
)
second_node.get_dS = node_matrix.get_S_fresnel_derivative_closure(
    first_link, second_link, perturbed_link_number
)

S2 = network.get_S_ee(k0)
# print(S2-S1)

# Change link n, see if things are affected
print(first_node.get_S(k0))
print(second_node.get_S(k0))
print(middle_link.get_S(k0))

middle_link.n = lambda k0: 1.6
middle_link.dn = lambda k0: 0.0

print("---")
print(first_node.get_S(k0))
print(second_node.get_S(k0))
print(middle_link.get_S(k0))

S3 = network.get_S_ee(k0)
