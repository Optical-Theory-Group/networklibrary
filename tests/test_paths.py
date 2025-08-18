# Write a code that tests if all the paths find_paths_to_target_distance is valid
# We will generate a network where we add the node to the existing network and calculate the path length

import matplotlib.pyplot as plt
import numpy as np

from complex_network.networks.network_perturbator import NetworkPerturbator
from complex_network.networks.network_factory import generate_network
from complex_network.networks.network_spec import NetworkSpec
from complex_network.interferometry import OLCR
from multiprocessing import Pool, cpu_count
from collections import deque
from complex_network.networks.network_paths import find_paths_to_target_distance


def check_unique_paths(path_results):
    """Check that all (detailed_path, position) pairs are unique.

    Returns (is_unique, duplicates_list). Each duplicate is a tuple (path, position).
    """
    seen = set()
    duplicates = []
    for p, pos in path_results:
        # normalize path to a tuple; round position to avoid tiny FP differences
        key = (tuple(p), round(float(pos), 12))
        if key in seen:
            duplicates.append((p, pos))
        else:
            seen.add(key)
    return (len(duplicates) == 0, duplicates)


spec = NetworkSpec(
        num_external_nodes=1,
        num_internal_nodes=10,
        network_type='delaunay',
        network_shape='slab',
        network_size=(200e-6, 200e-6),
        external_offset=10e-6,
        random_seed=10,
        fully_connected=True,
        node_S_mat_type='neumann'
    )
network = generate_network(spec)
source_node = 10
max_hops = 10
target_opls = [400e-6]
target_link = (2,8)
target_link_length = network.get_link_by_node_indices(target_link).length
target_link_index = network.get_link_by_node_indices(target_link).index

# get the paths and positions
# for target_opl in target_opls:
#     path_results = find_paths_to_target_distance(
#         network=network,
#         source_node_idx=source_node,
#         target_link=target_link,
#         target_distance=target_opl,
#         max_hops=max_hops,
#         tolerance=1e-6,
#         max_bounces=3
#     )

#     all_paths = []
#     all_positions = []
#     for result in path_results:
#         position = result[1]
#         path = result[0]

#         all_paths.append(path)
#         all_positions.append(position)

#         # Make a perturbed network with the node added to that position
#         # By default, the node is added from pos from the shorter index node

#         ratio = position/ target_link_length
#         perturbator = NetworkPerturbator(network)
#         perturbator.add_perturbation_node(link_index=target_link_index, fractional_position=ratio)
#         after = perturbator.perturbed_network

#         # now, the path, R will be replaced by 10. The first and last element of the path needs to be added by 1
#         path = [10 if x == 'R' else x for x in path]
#         path = [10 if x=='T' else x for x in path]
#         path[0] += 1
#         path[-1] += 1

        # print(path)

        # path_sum = np.sum(after.get_lengths_along_path(path))
        # print(path_sum*1e6)

# check if the paths are unique
path_results = find_paths_to_target_distance(
        network=network,
        source_node_idx=source_node,
        target_link=target_link,
        target_distance=target_opls[0],
        max_hops=max_hops,
        tolerance=1e-6,
        max_bounces=3
    )
is_unique, duplicates = check_unique_paths(path_results)
if not is_unique:
    print("Found duplicate paths:")
    for dup in duplicates:
        print(dup)

if is_unique:
    print("All paths are unique.")