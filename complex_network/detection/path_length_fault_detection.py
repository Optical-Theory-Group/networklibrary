"""This module implements the graph searching sensing algorithms, that takes peaks from an OLCR scan and then returns the possible
    position on the network that can explain that peak
    
    Assumptions.
        The perturbation is modelled as an additional node in between the link that scatters.
        Currently, we only support one perturbation within the network (i.e. one additional node), the idea is that faults are rare events
        The perturbation doesn't add any additional phase that can increase the optical path length of measured peaks
        We haven't considered the effects of dispersion which shifts OLCR peaks which is quite important"""


import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque, defaultdict


from complex_network.networks.network_path_search import compute_and_cache_all_path_lengths
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_path_search import _spec_fingerprint, load_path_cache, load_hop_counts_cache
from complex_network.networks.network_factory import generate_network
from complex_network.networks.network import Network


def _centered_bin_edges(data: np.ndarray, bin_size: float) -> np.ndarray:
    """Compute histogram bin edges of width bin_size centered so bin centers fall on
    multiples of bin_size and each edge is offset by bin_size/2.

    Example: bin_size=0.5e-6 -> centers at ..., -0.5e-6, 0, 0.5e-6, ... and edges at center +/- 0.25e-6.
    """
    data = np.asarray(data)
    if data.size == 0:
        # single bin centered at zero
        return np.array([-bin_size / 2, bin_size / 2])
    mn = float(np.min(data))
    mx = float(np.max(data))
    # start and end aligned to bin grid, then offset by half-bin so bin centers are multiples of bin_size
    start = (np.floor(mn / bin_size) * bin_size) - (bin_size / 2)
    end = (np.ceil(mx / bin_size) * bin_size) + (bin_size / 2)
    # create edges from start to end inclusive
    edges = np.arange(start, end + bin_size * 0.5, bin_size)
    return edges

def _compute_weighted_combinations(centers_a, centers_b, hist_a, hist_b, T_min, T_max, bin_size):
    """
    Compute weighted combinations of distance centers using scattering-based weighting.
    
    Args:
        centers_a, centers_b: Distance bin centers
        hist_a, hist_b: 2D histograms [dist_bin, hop_count] 
        T_min, T_max: Range for valid sums
        bin_size: Size for final binning
    
    Returns:
        centers, strengths, bin_details: Final histogram centers, strengths, and detailed path information
    """
    if len(centers_a) == 0 or len(centers_b) == 0:
        return None, None, None
    
    # Scattering parameters
    alpha = 0.43  # Signal reduction factor per scattering event
    
    # First pass: find all valid combinations and collect all hop counts to find global minimum
    all_hop_counts = set()
    valid_combinations_data = []
    
    for i, center_a in enumerate(centers_a):
        for j, center_b in enumerate(centers_b):
            sum_dist = center_a + center_b
            if T_min <= sum_dist <= T_max:
                path_details = {}
                
                # Get all hop counts that have non-zero paths
                for h1 in range(1, hist_a.shape[1]):
                    for h2 in range(1, hist_b.shape[1]):
                        num_paths = hist_a[i, h1] * hist_b[j, h2]
                        if num_paths > 0:
                            total_hops = h1 + h2
                            if total_hops not in path_details:
                                path_details[total_hops] = 0
                            path_details[total_hops] += num_paths
                            all_hop_counts.add(total_hops)
                
                if path_details:
                    valid_combinations_data.append((sum_dist, path_details))
    
    if not valid_combinations_data or not all_hop_counts:
        return None, None, None
    
    # Second pass: calculate strengths without normalization
    valid_combinations = []
    combination_strengths = []
    combination_details = []
    
    for sum_dist, path_details in valid_combinations_data:
        # Calculate scattering-based strength: m1*(alpha)^k1+m2*(alpha)^k2...
        strength = 0.0
        for hops, num_paths in path_details.items():
            hop_weight = alpha ** hops
            strength += num_paths * hop_weight

        if strength > 0:
            valid_combinations.append(sum_dist)
            combination_strengths.append(strength)
            combination_details.append((sum_dist, strength, path_details))
    
    if not valid_combinations:
        return None, None, None
    
    # Bin the combinations with their strengths
    combinations = np.array(valid_combinations)
    strengths = np.array(combination_strengths)

    edges = _centered_bin_edges(combinations, bin_size)
    final_strengths, bin_indices = np.histogram(combinations, bins=edges, weights=strengths)
    final_centers = (edges[:-1] + edges[1:]) / 2
    
    # Create detailed information for each bin
    bin_details = {}
    for idx, (sum_dist, strength, path_details) in enumerate(combination_details):
        bin_idx = np.digitize(sum_dist, edges) - 1
        if 0 <= bin_idx < len(final_centers):
            center = final_centers[bin_idx]
            if center not in bin_details:
                bin_details[center] = {'total_strength': 0, 'paths': {}}
            bin_details[center]['total_strength'] += strength
            for hops, num_paths in path_details.items():
                if hops not in bin_details[center]['paths']:
                    bin_details[center]['paths'][hops] = 0
                bin_details[center]['paths'][hops] += num_paths
    

    # Filter out bins with zero strength
    non_zero_mask = final_strengths > 0
    final_centers = final_centers[non_zero_mask]
    final_strengths = final_strengths[non_zero_mask]
    
    # Print detailed information for non-zero bins
    print("\nDetailed path analysis:")
    for i, center in enumerate(final_centers):
        if center in bin_details:
            details = bin_details[center]
            strength = final_strengths[i]

            # Recalculate strength based on aggregated paths without normalization
            if details['paths']:
                calculated_strength = 0.0
                for hops, num_paths in details['paths'].items():
                    hop_weight = alpha ** hops
                    calculated_strength += num_paths * hop_weight

                print(f"center: {center*1e6:.2f} µm, strength={calculated_strength:.2e}", end="")
            else:
                print(f"center: {center*1e6:.2f} µm, strength={strength:.2e}", end="")
            
            # Sort paths by hop count for consistent output
            sorted_paths = sorted(details['paths'].items())
            path_strings = []
            for hops, num_paths in sorted_paths:
                path_strings.append(f"{int(num_paths)} paths: {hops} scattering")
            
            if path_strings:
                print(f", {', '.join(path_strings)}")
            else:
                print()
    
    # Check if any bins have strength
    if len(final_centers) == 0 or np.sum(final_strengths) == 0:
        return None, None, None

    return final_centers, final_strengths, bin_details
    
def _create_fixed_position_bins(link_length: float, bin_size: float) -> Tuple[np.ndarray, np.ndarray]:
    """Create fixed bins from 0 to link_length with given bin_size.
    
    Args:
        link_length: Length of the link (L)
        bin_size: Size of each bin
    
    Returns:
        bin_edges, bin_centers: Arrays of bin edges and centers
    """
    # Create bins from 0 to link_length
    num_bins = int(np.ceil(link_length / bin_size))
    bin_edges = np.linspace(0, num_bins * bin_size, num_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    return bin_edges, bin_centers

def _bin_x_positions_with_strengths(x_positions: np.ndarray, strengths: np.ndarray, 
                                  bin_edges: np.ndarray, bin_centers: np.ndarray,
                                  link_length: float) -> Tuple[np.ndarray, np.ndarray]:
    """Bin x positions into fixed bins and aggregate their strengths.
    
    Args:
        x_positions: Array of x positions to bin
        strengths: Corresponding strengths for each x position
        bin_edges: Fixed bin edges from 0 to L
        bin_centers: Fixed bin centers
        link_length: Length of the link for filtering valid positions
    
    Returns:
        non_zero_centers, non_zero_strengths: Centers and strengths of non-zero bins
    """
    if len(x_positions) == 0 or len(strengths) == 0:
        return np.array([]), np.array([])
    
    # Filter x positions to be within [0, link_length]
    valid_mask = (x_positions >= 0) & (x_positions <= link_length)
    valid_x = x_positions[valid_mask]
    valid_strengths = strengths[valid_mask]
    
    if len(valid_x) == 0:
        return np.array([]), np.array([])
    
    # Bin the valid x positions with their strengths
    binned_strengths, _ = np.histogram(valid_x, bins=bin_edges, weights=valid_strengths)
    
    # Filter out bins with zero strength
    non_zero_mask = binned_strengths > 0
    non_zero_centers = bin_centers[non_zero_mask]
    non_zero_strengths = binned_strengths[non_zero_mask]
    
    return non_zero_centers, non_zero_strengths

def _plot_solutions(network:Network, target_link:Tuple[int, int], solutions:np.ndarray, color:str = 'r',marker="x"):
    """This plots the suspected positions on the link that might correpsond to the actual point on the network"""
    # plt.figure(figsize=(10, 10), dpi=200)
    # network.draw(show_internal_indices=True, show_external_indices=True)

    # get the nodes in the link we suspect
    (node_a, node_b) = target_link
    node_a_x, node_a_y = network.get_node(node_a).position[0], network.get_node(node_a).position[1]
    node_b_x, node_b_y = network.get_node(node_b).position[0], network.get_node(node_b).position[1]

    # get the angle of the link
    angle = np.arctan2(node_b_y - node_a_y, node_b_x - node_a_x)

    # plot the solutions along this distance along the angle.
    x_solutions = node_a_x + solutions * np.cos(angle)
    y_solutions = node_a_y + solutions * np.sin(angle)

    plt.scatter(x_solutions, y_solutions, color=color, marker=marker, s=5, alpha=0.5)
    plt.show()

class OptimizedSolver:

    def __init__(self, network: Network, network_spec: NetworkSpec, sources: List[int], max_hops: int):
        self.network_spec = network_spec
        self.sources = sources
        self.max_hops = max_hops
        self.network = network

        external_node_indices = [node.index for node in self.network.external_nodes]
        if not all(source in external_node_indices for source in sources):
            raise ValueError(f"Sources must be external nodes")

        # Generate the fingerprint of network_spec
        self.network_spec_fingerprint = _spec_fingerprint(network_spec)

        # Check if all the path files have been computed for the source nodes and cached
        # Will compute the missing path files
        for source in self.sources:
            compute_and_cache_all_path_lengths(self.network, source, self.network_spec_fingerprint, self.max_hops)

    
    def _single_link_solver(self, target_opls, source_idx, target_link, bin_size: float = 0.5e-6):
        """Find the positions in this link that will correspond to the target optical path lengths."""

        # Get the node indices of the target link
        (node_a, node_b) = target_link
        target_link_length = self.network.get_link_by_node_indices((node_a, node_b)).length

        # Create fixed bins from 0 to L
        bin_edges, bin_centers = _create_fixed_position_bins(target_link_length, bin_size)
        print(f"\nFixed bins from 0 to {target_link_length*1e6:.2f} µm with bin size {bin_size*1e6:.2f} µm")
        print(f"Number of bins: {len(bin_centers)}")

        # Load all the distances and hop counts
        path_distances_to_a = load_path_cache(self.network_spec_fingerprint, source_idx, node_a, self.max_hops)
        path_distances_to_b = load_path_cache(self.network_spec_fingerprint, source_idx, node_b, self.max_hops)
        hop_counts_to_a = load_hop_counts_cache(self.network_spec_fingerprint, source_idx, node_a, self.max_hops)
        hop_counts_to_b = load_hop_counts_cache(self.network_spec_fingerprint, source_idx, node_b, self.max_hops)

        # path_distances_to_a and path_distances_to_b is the path lengths of all paths going into node a and node b
        # We have 4 possibilities
        # 1. goes in through A and comes back through A
        #      path_distance_to_a + path_distance_to_a' + 2x = target_opls
        #       (a, a' belongs to A)
        # 2. goes in through A and comes back through B
        #   2 cases:
        #      reflection first then transmission | transmission first and reflection
        #       path_distance_to_a + 2x + L + path_distance_to_b = target_opls
        #       (a belongs to A, b belongs to B)
        #       path_distance_to_a + 3L -2x + path_distance_to_b = target_opls
        #       (a belongs to A, b belongs to B)
        # 3. goes in through B and comes back through A
        #   2 cases:
        #      reflection first then transmission | transmission first and reflection
        #       path_distance_to_b + 3L - 2x + path_distance_to_a = target_opls
        #       (b belongs to B, a belongs to A)
        #       path_distance_to_b + L + 2x + path_distance_to_a = target_opls
        #       (b belongs to B, a belongs to A)
        # 4. goes in through B and comes back through B
        #       path_distance_to_b + path_distance_to_b' + 2(L-x) = target_opls
        #       (b,b' belongs to B)

        # Use centered bins so values are centered and bin centers differ by bin_size,
        # which results in values offset by bin_size/2 on both sides (e.g., +/- 0.25um for 0.5um bins).
        edges_a = _centered_bin_edges(np.asarray(path_distances_to_a), bin_size/2)
        edges_b = _centered_bin_edges(np.asarray(path_distances_to_b), bin_size/2)

        # Calculate bin centers (histogram centers)
        centers_a = (edges_a[:-1] + edges_a[1:]) / 2
        centers_b = (edges_b[:-1] + edges_b[1:]) / 2
        
        # Create 2D histograms: hist[dist_bin, hop_count] = number of paths
        max_hops = self.max_hops
        
        # Initialize 2D histograms for distance bins vs hop counts
        hist_a = np.zeros((len(centers_a), max_hops + 1))
        hist_b = np.zeros((len(centers_b), max_hops + 1))
        
        # Populate histograms
        for dist, hops in zip(path_distances_to_a, hop_counts_to_a):
            # Find which distance bin this path belongs to
            dist_bin_idx = np.digitize(dist, edges_a) - 1
            if 0 <= dist_bin_idx < len(centers_a) and 1 <= hops <= max_hops:
                hist_a[dist_bin_idx, hops] += 1
                
        for dist, hops in zip(path_distances_to_b, hop_counts_to_b):
            # Find which distance bin this path belongs to  
            dist_bin_idx = np.digitize(dist, edges_b) - 1
            if 0 <= dist_bin_idx < len(centers_b) and 1 <= hops <= max_hops:
                hist_b[dist_bin_idx, hops] += 1
        
        # Use the new scattering-based weighted combination approach for all three cases
        # Case 1: A to A paths - path_distance_to_a + path_distance_to_a' + 2x = target_opls
        # Range: target_opls - 2*target_link_length <= sum <= target_opls
        print("\n=== Case 1: A to A paths ===")
        centers_aa, strengths_aa, details_aa = _compute_weighted_combinations(
            centers_a, centers_a, hist_a, hist_a,
            T_min=target_opls - 2*target_link_length, 
            T_max=target_opls,
            bin_size=bin_size
        )
        
        # Case 4: B to B paths - path_distance_to_b + path_distance_to_b' + 2(L-x) = target_opls
        # Range: target_opls - 2*target_link_length <= sum <= target_opls  
        print("\n=== Case 4: B to B paths ===")
        centers_bb, strengths_bb, details_bb = _compute_weighted_combinations(
            centers_b, centers_b, hist_b, hist_b,
            T_min=target_opls - 2*target_link_length, 
            T_max=target_opls,
            bin_size=bin_size
        )
        
        # Cases 2&3: A to B paths - path_distance_to_a + path_distance_to_b + L ± 2x = target_opls
        # Range: target_opls - 3*target_link_length <= sum <= target_opls - target_link_length
        print("\n=== Cases 2&3: A to B paths ===")
        centers_ab, strengths_ab, details_ab = _compute_weighted_combinations(
            centers_a, centers_b, hist_a, hist_b,
            T_min=target_opls - 3*target_link_length, 
            T_max=target_opls - target_link_length,
            bin_size=bin_size
        )

        # Calculate x values, handling None cases
        # For aa-> x = (target_opls - centers_aa) / 2
        # For bb-> x = target_link_length - (target_opls - centers_bb) / 2
        # for ab-> 
        #   reflection_first transmission second:
        #       x = (target_opls-centers_ab-target_link_length) / 2
        #   transmission_first reflection_second:
        #       x = (centers_ab + 3*target_link_length - target_opls) / 2
        x_aa = (target_opls - centers_aa) / 2 if centers_aa is not None else np.array([])
        x_ab = (target_opls - centers_ab - target_link_length) / 2 if centers_ab is not None else np.array([])
        x_ba = (centers_ab + 3*target_link_length - target_opls) / 2 if centers_ab is not None else np.array([])
        x_bb = target_link_length - (target_opls - centers_bb) / 2 if centers_bb is not None else np.array([])

        # Bin x positions into fixed bins and get non-zero bins with strengths
        print("\n=== Binning x positions into fixed bins ===")
        
        # Case AA
        aa_centers, aa_strengths = _bin_x_positions_with_strengths(
            x_aa, strengths_aa if strengths_aa is not None else np.array([]), 
            bin_edges, bin_centers, target_link_length
        )
        
        # Case BB  
        bb_centers, bb_strengths = _bin_x_positions_with_strengths(
            x_bb, strengths_bb if strengths_bb is not None else np.array([]), 
            bin_edges, bin_centers, target_link_length
        )
        
        # Case AB (reflection first)
        ab_centers, ab_strengths = _bin_x_positions_with_strengths(
            x_ab, strengths_ab if strengths_ab is not None else np.array([]), 
            bin_edges, bin_centers, target_link_length
        )
        
        # Case BA (transmission first)  
        ba_centers, ba_strengths = _bin_x_positions_with_strengths(
            x_ba, strengths_ab if strengths_ab is not None else np.array([]), 
            bin_edges, bin_centers, target_link_length
        )

        # Combine all non-zero bins
        all_centers = []
        all_strengths = []
        
        if len(aa_centers) > 0:
            all_centers.extend(aa_centers)
            all_strengths.extend(aa_strengths)
            
        if len(bb_centers) > 0:
            all_centers.extend(bb_centers)
            all_strengths.extend(bb_strengths)
            
        if len(ab_centers) > 0:
            all_centers.extend(ab_centers)
            all_strengths.extend(ab_strengths)
            
        if len(ba_centers) > 0:
            all_centers.extend(ba_centers)
            all_strengths.extend(ba_strengths)

        # Aggregate strengths for bins that appear multiple times
        if all_centers:
            unique_centers = np.unique(all_centers)
            aggregated_strengths = np.zeros(len(unique_centers))
            
            for center, strength in zip(all_centers, all_strengths):
                idx = np.where(unique_centers == center)[0][0]
                aggregated_strengths[idx] += strength
            
            # Filter out zero strengths
            non_zero_mask = aggregated_strengths > 0
            final_centers = unique_centers[non_zero_mask]
            final_strengths = aggregated_strengths[non_zero_mask]
            
            # Sort by position
            sort_idx = np.argsort(final_centers)
            final_centers = final_centers[sort_idx]
            final_strengths = final_strengths[sort_idx]
        else:
            final_centers = np.array([])
            final_strengths = np.array([])

        # Print results
        print(f"\nFinal binned results ({len(final_centers)} non-zero bins):")
        for center, strength in zip(final_centers, final_strengths):
            print(f"  Bin center: {center*1e6:.2f} µm, Strength: {strength:.2e}")

        # Concatenate all the x values for compatibility (optional)
        x_arrays = [arr for arr in [x_aa, x_ab, x_ba, x_bb] if len(arr) > 0]
        x = np.concatenate(x_arrays) if x_arrays else np.array([])

        return (final_centers, final_strengths, x_aa, x_bb, x_ab, x_ba, x)

if __name__ == "__main__":
    seed = 10
    ne = 1
    ni = 10

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
        node_S_mat_type='neumann')

    network = generate_network(spec)

    d = OptimizedSolver(network, spec, [10], 10)
    target_opls = 311.90623812e-6
    target_pls = target_opls / 1.5

    plt.figure(figsize=(10, 10), dpi=200)
    network.draw(show_internal_indices=True, show_external_indices=True)

    (final_centers, final_strengths, x_aa, x_bb, x_ab, x_ba, x) = d._single_link_solver(target_opls=target_pls, source_idx=10, target_link=(2,8), bin_size=1e-6)

    # _plot_solutions(network, (2, 8), x, color='r',marker="o")
    # plt.savefig(os.path.join(os.getcwd(), "test.png"))