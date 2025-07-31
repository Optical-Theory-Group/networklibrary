"""The goal of this module is to implement a graph based searching and perturbation recovery algorithm.
   The code uses peaks obtained from a OLCR scan and then tries to recover the original position of 
   the perturbation within the network that best explains the observed peaks."""


from importlib.resources import path
import matplotlib.pyplot as plt
import numpy as np
from complex_network.networks.network_factory import generate_network #type: ignore
from complex_network.networks.network_spec import NetworkSpec #type: ignore
from collections import deque
from typing import List, Tuple, Dict, Union, Any, Set
from complex_network.networks.network import Network #type: ignore
import logging
from itertools import product
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

""" We will define a function that finds spatial candidates in the network that add upto
    a particular optical path length.
    This condition alone will lead to infinite number of candidates. So, we introduce the constraint
    that all the paths are within a number of hops from the source node. (finite scattering events)
    
    We are trying to find the fractional poistion along a link between two nodes where the perturbation
    is located. So, we will estimate the paths to the nodes at both ends of the link from the source node.
    These paths will already have a fixed length. We try to reverse estimate the perturbation using this length
    and the target optical path length we have to get to"""

def find_paths_to_node(
        network: Network,
        start_node: int,
        end_node: int,
        num_max_scatter: int
) -> List[List[int]]:
    """Find paths from start_node to end_node within num_max_scatter."""
    paths = []
    queue = deque([(start_node, [start_node])])

    while queue:
        current_node, path = queue.popleft()

        # The first node does not scatter
        num_scatter = len(path) - 1
        if num_scatter >  num_max_scatter:
            continue

        if current_node == end_node:
            paths.append(path)
            continue

        for neighbor in network.get_connecting_nodes(current_node):
            if neighbor.index != start_node: # We will skip the start node
                queue.append((neighbor.index, path + [neighbor.index]))

    return paths

def find_paths_from_node(
        network: Network,
        start_node: int,
        end_node: int,
        num_max_scatter: int
) -> List[List[int]]:
    """Find paths from start_node to end_node within num_max_scatter."""
    paths = []
    queue = deque([(start_node, [start_node])])

    while queue:
        current_node, path = queue.popleft()

        # The last node does not scatter
        num_scatter = len(path) - 1
        if num_scatter >= num_max_scatter:
            continue

        if current_node == end_node:
            paths.append(path)
        
        else:
            for neighbor in network.get_connecting_nodes(current_node):
                queue.append((neighbor.index, path + [neighbor.index]))

    return paths

def insert_transmission_markers(
        path: List[Union[int, str]],
        reflection_node: int,
        other_node: int
)-> List[Union[int, str]]:
    """Insert 'T' markers whenever the path traverses through the link containing the perturbation."""
    result = []
    i = 0

    while i< len(path):
        result.append(path[i])

        # Check if we need to insert a transmission marker between the current and next node
        if i < len(path) - 1:
            current_node = path[i]
            next_node = path[i + 1]

            # Only check if both elements are node indices 
            if isinstance(current_node,int) and isinstance(next_node,int):
                # Check if this is the traversal of the perturbation link in either direction.
                if (current_node == reflection_node and next_node == other_node) or \
                   (current_node == other_node and next_node == reflection_node):
                    result.append('T')

        i += 1
    return result

def analyze_path_for_reflection(
        network: Network,
        path: List[int],
        reflection_node: int,
        other_node: int,
        target_opl: float,
        num_max_scatter: int
)-> List[Dict[str, Union[str,int, float, List]]]:
    possible_locations = []
    target_link = network.get_link_by_node_indices(node_indices=[reflection_node, other_node])
    link_length = target_link.length

    # assuming k=1e7, fix when accounting for dispersion effects
    n_index = target_link.n(1e7)
    # Calculating the optical path length to one end of the link
    opl_prime = network.get_optical_path_length(path)
    if opl_prime > target_opl:
        return []    
    remaining_opl = target_opl - opl_prime
    n=2
    while True:
        num_scatter = len(path) - 1
        reflection_in_link = n
        total_scatter = num_scatter + reflection_in_link

        if total_scatter > num_max_scatter:
            break
        l = remaining_opl / (reflection_in_link*n_index)

        if 0< l < link_length:
            try:
                idx = path.index(reflection_node)
                path_to_node = path[:idx+1]
                path_from_node = path[idx:]

                bounce_sequence = []
                for _ in range(n//2):
                    bounce_sequence.extend(['R', reflection_node])

                # We will construct the path without the transmission markers
                formatted_path =  path_to_node + bounce_sequence + path_from_node[1:]

                # Now, we will insert the transmission markers
                formatted_path = insert_transmission_markers(formatted_path,reflection_node, other_node)

            except ValueError:
                formatted_path = path

            if reflection_node < other_node:
                location_from_smaller = l
                location_ratio = l / link_length
            else:
                location_from_smaller = link_length - l
                location_ratio = 1 - (l / link_length)

            result = {
                'status': 'accepted',
                'type': 'reflection',
                'num_scatter': total_scatter,
                'location': location_from_smaller,
                'location_ratio': location_ratio,
                'path': formatted_path,
                'reflection_node': reflection_node,
                'n_bounce': n
            }
            possible_locations.append(result)
        n += 2

    return possible_locations

def find_perturbation_candidates(
        network: Network,
        source_node: int,
        target_link_nodes: Tuple[int, int],
        target_opl: float,
        num_max_scatter: int,
        tolerance: float = 1e-6
        ) -> List[Dict[str, Union[str, int, float, List]]]:
    """Find the perturbation location in the network based on the target optical path length."""
    results = []
    node_a, node_b  = sorted(target_link_nodes)
    target_link = network.get_link_by_node_indices(node_indices=sorted(target_link_nodes))
    link_length = target_link.length
    n_index = target_link.n(1e7)  # Assuming k=1e7, fix when accounting for dispersion effects

    if source_node in (node_a, node_b):
        return [] # We skip analysis for the first link

    # Reflection candidates
    for reflection_end_node, other_end_node in [(node_a, node_b), (node_b, node_a)]:
        # Find paths to the reflection end node
        paths_to_reflection = find_paths_to_node(network, source_node, reflection_end_node, num_max_scatter//2)
        paths_from_reflection = find_paths_from_node(network, reflection_end_node, source_node, num_max_scatter//2)

        for path_in in paths_to_reflection:
            for path_out in paths_from_reflection:
                # Combine the paths
                combined_path = path_in + path_out[1:]

                reflection_results = analyze_path_for_reflection(
                    network,
                    combined_path,
                    reflection_end_node,
                    other_end_node,
                    target_opl,
                    num_max_scatter
                )
                results.extend(reflection_results)


    # Transmission candidates
    paths_through_link = []
    paths_to_node_a = find_paths_to_node(network, source_node, node_a, num_max_scatter//2)
    paths_to_node_b = find_paths_to_node(network, source_node, node_b, num_max_scatter//2)

    for path_to_a in paths_to_node_a:
        paths_from_b = find_paths_from_node(network, node_b, source_node, num_max_scatter-len(path_to_a))
        for path_from_b in paths_from_b:
            # Combine the paths
            combined_path = path_to_a + path_from_b
            combined_path = insert_transmission_markers(combined_path, node_a, node_b)
            paths_through_link.append(combined_path)

    for path_to_b in paths_to_node_b:
        paths_from_a = find_paths_from_node(network, node_a, source_node, num_max_scatter-len(path_to_b))
        for path_from_a in paths_from_a:
            # Combine the paths
            combined_path = path_to_b + path_from_a
            combined_path = insert_transmission_markers(combined_path, node_a, node_b)
            paths_through_link.append(combined_path)

    for path in paths_through_link:
        num_scatter = len(path) - 1
        if num_scatter > num_max_scatter:
            continue
        path_without_marker = [node for node in path if node != 'T']
        total_opl = network.get_optical_path_length(path_without_marker)

        if abs(total_opl - target_opl) < tolerance:
            results.append({
                'status': 'accepted',
                'type': 'transmission',
                'num_scatter': num_scatter,
                'location': 'N/A',
                'location_ratio': 'N/A',
                'path': path})


    return results

if __name__ == "__main__":

    # Generate network
    seed = 9
    ne = 1
    ni = 5

    wavelength = 1000e-9
    k = 2 * np.pi / wavelength

    spec = NetworkSpec(
        num_external_nodes=ne,
        num_internal_nodes=ni,
        network_type="delaunay",
        network_shape="slab",
        network_size=(200e-6, 200e-6),
        external_offset=10e-6,
        random_seed=seed,
        fully_connected=True,
        node_S_mat_type='neumann'
    )
    network = generate_network(spec)

    # # Example usage
    # source_node = 6
    # links = [link.sorted_connected_nodes for link in network.internal_links]
    # opls1 = np.array([483.60967219,  589.91179824,  622.21244425,  658.71317426,
    #     695.61391228,  730.41460829,  759.1151823 ,  784.61569231,
    #     # 799.21598432,  807.21614432,  819.61639233,  830.61661233,
    #     # 852.21704434,  859.91719834,  869.01738035,  879.51759035,
    #     # 892.41784836,  904.51809036,  923.71847437,  938.81877638,
    #     # 959.21918438,  980.31960639,  993.3198664 , 1006.0201204
    #     ]) * 1e-6
    # for opl in opls1:
    #     for target_link_nodes in links:
    #         measured_opl = opl  # Use the current OPL from the array
    #         max_hops = 12  # Example maximum hops

    #         results = find_perturbation_candidates(
    #             network, source_node, target_link_nodes, measured_opl, num_max_scatter=max_hops
    #         )
    #         for result in results:
    #             if result['status'] == 'accepted':
    #                 # If any of the path is a transmission, we skip this OPL
    #                 if result['type'] == 'transmission':
    #                     valid_opl1 = False
    #                     print(f"Skipping OPL {measured_opl} for source {source_node} due to transmission path.")

    opls1 = np.array([302.50605012,  402.10804216,  407.40814816,
            471.30942619,  480.10960219,  508.0101602 ,
            # 555.71111422,  577.21154423,  592.61185224,  613.21226425,
            # 624.21248425,  639.81279626,  657.11314226,  686.91373827,
            # 697.11394228,  719.61439229,  742.9148583 ,  764.41528831,
            # 780.91561831,  793.11586232,  798.81597632,  819.61639223,
            # 828.81657633,  848.91697834,  858.41716834,  869.21738435,
            # 882.61765235,  898.11796236,  917.01834037,  928.41856837,
            # 958.41916838,  977.71955439,  988.5197704 ,  997.5199504
            ]) * 1e-6  # Example measured optical path lengths

    opls2 = np.array([483.60967219,
            695.61391228,  759.1151823 ,
            # 799.21598432,  807.21614432,  819.61639233,  830.61661233,
            # 852.21704434,  859.91719834,  869.01738035,  879.51759035,
            # 892.41784836,  904.51809036,  923.71847437,  938.81877638,
            # 959.21918438,  980.31960639,  993.3198664 , 1006.0201204
            ]) * 1e-6  # Example measured optical path lengths

    links = [link.sorted_connected_nodes for link in network.internal_links]
    source_node1 = 5
    source_node2 = 6  # Another source node for the second OPL
    max_hops = 12  # Example maximum hops
    results1 = find_perturbation_candidates(network, source_node1, (0,2), 302.50605012e-6, num_max_scatter=max_hops)
    results2 = find_perturbation_candidates(network, source_node2, (0,2), 695.61391228e-6, num_max_scatter=max_hops)

    for result in results1:
        if result['status'] == 'accepted':
            print(f"Source Node: {source_node1}, OPL: {result['location']*1e6:.2f}µm, Path: {result['path']}")

    for result in results2:
        if result['status'] == 'accepted':
            print(f"Source Node: {source_node2}, OPL: {result['location']*1e6:.2f}µm, Path: {result['path']}")

    # for o1 in opls1:
    #     for o2 in opls2:
    #         LINK1 = []
    #         LOC1 = []

    #         LINK = []
    #         LOC = []
    #         valid_opl1 = True
    #         for target_link_nodes in links:    
    #             results = find_perturbation_candidates(network, source_node1, target_link_nodes, o1, num_max_scatter=max_hops)
    #             print(f"Analyzing OPLs {o1*1e6:.2f}µm and {o2*1e6:.2f}µm")
    #             print(f"Target link nodes: {target_link_nodes}")
    #             # print(results)
    #             for result in results:
    #                 if result['status'] == 'accepted':
    #                     # If any of the path is a transmission, we skip this OPL
    #                     if result['type'] == 'transmission':
    #                         valid_opl1 = False
    #                         break
    #             if not valid_opl1:
    #                 break

    #         valid_opl2 = True
    #         for target_link_nodes in links:
    #             results = find_perturbation_candidates(network, source_node2, target_link_nodes, o2, num_max_scatter=max_hops)
    #             for result in results:
    #                 if result['status'] == 'accepted':
    #                     # If any of the path is a transmission, we skip this OPL
    #                     if result['type'] == 'transmission':
    #                         valid_opl2 = False
    #                         break
    #             if not valid_opl2:
    #                 break

            
    #         if valid_opl1:
    #             for target_link_nodes in links:    
    #                 results = find_perturbation_candidates(network, source_node1, target_link_nodes, o1, num_max_scatter=max_hops)

    #                 # Make a temporary list to hold results for this link
    #                 temp_link = []
    #                 temp_loc = []
    #                 for result in results:
    #                     if result['status'] == 'accepted':
    #                         loc = result['location']
    #                         if not isinstance(loc, str):
    #                             temp_link.append(target_link_nodes)
    #                             temp_loc.append(result['location'])
    
    #                 LINK.extend(temp_link)
    #                 LOC.extend(temp_loc)

    #         if valid_opl2:
    #             for target_link_nodes in links:    
    #                 results = find_perturbation_candidates(network, source_node2, target_link_nodes, o2, num_max_scatter=max_hops)

    #                 # Make a temporary list to hold results for this link
    #                 temp_link1 = []
    #                 temp_loc1 = []
    #                 for result in results:
    #                     if result['status'] == 'accepted':
    #                         loc = result['location']
    #                         if not isinstance(loc, str):
    #                             temp_link1.append(target_link_nodes)
    #                             temp_loc1.append(result['location'])

    #                 # Before appending, discard the whole results if there was a transmission
    #                 if any(result['type'] == 'transmission' for result in results):
    #                     continue
    #                 LINK1.extend(temp_link1)
    #                 LOC1.extend(temp_loc1)
            
    #         if valid_opl1 and valid_opl2:
    #             XLOC = []
    #             YLOC = []
    #             for i, link_nodes in enumerate(LINK):
    #                 location = LOC[i]

    #                 # I need to get the spatial coordinates for the location
    #                 node_a, node_b = link_nodes

    #                 node_a_coords = network.get_node(node_a).position
    #                 node_b_coords = network.get_node(node_b).position

    #                 # The location is the distance from node_a to the perturbation
    #                 # Calculate the angle of the link with respect to the x-axis
    #                 link_vector = np.array(node_b_coords) - np.array(node_a_coords)
    #                 link_angle = np.arctan2(link_vector[1], link_vector[0])

    #                 # resolve the location into x and y coordinates
    #                 location_x = node_a_coords[0] + location * np.cos(link_angle)
    #                 location_y = node_a_coords[1] + location * np.sin(link_angle)

    #                 XLOC.append(location_x)
    #                 YLOC.append(location_y)

    #             XLOC1 = []
    #             YLOC1 = []
    #             for i, link_nodes in enumerate(LINK1):
    #                 location = LOC1[i]

    #                 # I need to get the spatial coordinates for the location
    #                 node_a, node_b = link_nodes

    #                 node_a_coords = network.get_node(node_a).position
    #                 node_b_coords = network.get_node(node_b).position

    #                 # The location is the distance from node_a to the perturbation
    #                 # Calculate the angle of the link with respect to the x-axis
    #                 link_vector = np.array(node_b_coords) - np.array(node_a_coords)
    #                 link_angle = np.arctan2(link_vector[1], link_vector[0])

    #                 # resolve the location into x and y coordinates
    #                 location_x = node_a_coords[0] + location * np.cos(link_angle)
    #                 location_y = node_a_coords[1] + location * np.sin(link_angle)

    #                 XLOC1.append(location_x)
    #                 YLOC1.append(location_y)
                
    #             # Now we can plot the perturbation location
    #             plt.figure(figsize=(10, 10),dpi=1000)
    #             network.draw(show_indices=True)
    #             plt.title(f"Perturbation Locations for OPLs {o1*1e6:.2f}µm and {o2*1e6:.2f}µm")
    #             plt.scatter(XLOC, YLOC, color='red',marker='x', alpha=1)
    #             # Have no fill in the circles
    #             plt.scatter(XLOC1, YLOC1, color='blue', marker='o', alpha=1, edgecolor='blue', facecolor='none')
    #             plt.savefig(f"perturbation_locations_{o1*1e6:.2f}_{o2*1e6:.2f}.png")

