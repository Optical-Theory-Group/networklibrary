# We will write a code to test the performance of the naive localization algorithm
import numpy as np
import time
from complex_network.detection.naive_localization import FaultLocalizer
from complex_network.networks.network_path_search import _spec_fingerprint
from complex_network.networks.network_spec import NetworkSpec
import os
from scipy.signal import hilbert, savgol_filter, find_peaks
from complex_network.networks.network_factory import generate_network
import json

output_dir = '/home/baruva/network_cache/olcr_data'
ni = [4,6,8,10,12]
ne = 1
seed_indices = np.arange(1,100,1)
ni = 6
# Global statistics across all networks
global_stats = {
    'correct_predictions': 0,
    'correct_link_wrong_position': 0,
    'wrong_link': 0,
    'total_predictions': 0,
    'position_errors': [],
    'per_network_stats': {}
}

for seed in seed_indices:
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
    links = [link.index for link in network.internal_links]
    sources = [node.index for node in network.external_nodes]
    
    # Per-network statistics
    network_stats = {
        'seed': seed,
        'ni': ni,
        'correct_predictions': 0,
        'correct_link_wrong_position': 0,
        'wrong_link': 0,
        'total_predictions': 0,
        'position_errors': [],
        'per_link_results': {},
        'per_link_accuracy': {}
    }

    for link_index in links:
        link_length = network.get_link(link_index).length
        # Create fault positions every 5μm along the link
        fault_positions = np.arange(5e-6, link_length, 5e-6)
        ratios = fault_positions / link_length
        
        # Round ratios to 2 decimal places and filter
        ratios = np.round(ratios, 2)
        # Make the ratios unique and within (0, 1)
        ratios = np.unique(ratios)
        ratios = ratios[(ratios > 0) & (ratios < 1)]

        for ratio in ratios:
            pert_file_s1 = os.path.join(output_dir, f'{_spec_fingerprint(spec)}_pert_link{link_index}_r{ratio:.2f}_s1.npy')
            pert_file_s2 = os.path.join(output_dir, f'{_spec_fingerprint(spec)}_pert_link{link_index}_r{ratio:.2f}_s2.npy')
            ref_file_s1 = os.path.join(output_dir, f'{_spec_fingerprint(spec)}_ref_s1.npy')
            ref_file_s2 = os.path.join(output_dir, f'{_spec_fingerprint(spec)}_ref_s2.npy')
            opl_file = os.path.join(output_dir, f'{_spec_fingerprint(spec)}_opls.npy')

            pert_s1 = np.load(pert_file_s1)
            pert_s2 = np.load(pert_file_s2)
            ref_s1 = np.load(ref_file_s1)
            ref_s2 = np.load(ref_file_s2)
            opl = np.load(opl_file)

            olcr_ref_dict = {}
            olcr_pert_dict = {}
            dx = np.abs(opl[1]-opl[0])
            spatial_resolution = 1.1e-6/3  # in meters (physical distance)
            coherence_length = 1.1e-6  # in meters (optical path length)
            smooth_window = int(np.ceil(coherence_length / dx))  # Both in optical path length units
            if smooth_window % 2 == 0:
                smooth_window += 1  # Ensure window length is odd for savgol_filter

            # # Lets find the envelope of the interferograms
            # Remove the DC component
            I_perturbed1 = pert_s1 - np.mean(pert_s1)
            I_reference1 = ref_s1 - np.mean(ref_s1)

            env_I_perturbed1 = np.abs(hilbert(I_perturbed1))
            env_I_reference1 = np.abs(hilbert(I_reference1))

            env_I_perturbed_s1 = savgol_filter(env_I_perturbed1, window_length=smooth_window, polyorder=3)
            env_I_reference_s1 = savgol_filter(env_I_reference1, window_length=smooth_window, polyorder=3)

            I_perturbed2 = pert_s2 - np.mean(pert_s2)
            I_reference2 = ref_s2 - np.mean(ref_s2)

            env_I_perturbed2 = np.abs(hilbert(I_perturbed2))
            env_I_reference2 = np.abs(hilbert(I_reference2))

            env_I_perturbed_s2 = savgol_filter(env_I_perturbed2, window_length=smooth_window, polyorder=3)
            env_I_reference_s2 = savgol_filter(env_I_reference2, window_length=smooth_window, polyorder=3)

            olcr_ref_dict[sources[0]] = (opl, env_I_reference_s1)
            olcr_pert_dict[sources[0]] = (opl, env_I_perturbed_s1)

            olcr_ref_dict[sources[1]] = (opl, env_I_reference_s2)
            olcr_pert_dict[sources[1]] = (opl, env_I_perturbed_s2)

            localizer = FaultLocalizer(network=network, source_indices=sources, max_hops=14, n_index=1.5, coherence_length=1.1e-6)
            (best_link, best_position, score) = localizer.localize_fault(olcr_ref_dict, olcr_pert_dict)

            # If the prediction is in the same link and position is within a coherence length, we count it as correct
            true_link = (network.get_link(link_index).sorted_connected_nodes[0], network.get_link(link_index).sorted_connected_nodes[1])
            true_link_length = network.get_link(link_index).length
            threshold = spatial_resolution/true_link_length  # position error threshold in terms of ratio

            # Update statistics
            network_stats['total_predictions'] += 1
            global_stats['total_predictions'] += 1
            
            # Store per-link result
            if link_index not in network_stats['per_link_results']:
                network_stats['per_link_results'][link_index] = {
                    'correct': 0,
                    'correct_link_wrong_pos': 0,
                    'wrong_link': 0,
                    'position_errors': []
                }

            # spatial position of the fault
            if best_link == true_link:
                position_error = abs(best_position - ratio)
                network_stats['per_link_results'][link_index]['position_errors'].append(position_error)
                network_stats['position_errors'].append(position_error)
                global_stats['position_errors'].append(position_error)
                
                if position_error <= threshold:
                    # Category 1: Correct link and position within threshold
                    network_stats['correct_predictions'] += 1
                    global_stats['correct_predictions'] += 1
                    network_stats['per_link_results'][link_index]['correct'] += 1
                    print(f'✓ Correct: link {best_link} at position {best_position:.2e} (true: {ratio:.2e}), error: {position_error:.2e}, score: {score:.4f}')
                else:
                    # Category 2: Correct link but position outside threshold
                    network_stats['correct_link_wrong_position'] += 1
                    global_stats['correct_link_wrong_position'] += 1
                    network_stats['per_link_results'][link_index]['correct_link_wrong_pos'] += 1
                    print(f'~ Correct link but position error: link {best_link}, error {position_error:.2e} exceeds threshold {threshold:.2e}')
            else:
                # Category 3: Wrong link
                network_stats['wrong_link'] += 1
                global_stats['wrong_link'] += 1
                network_stats['per_link_results'][link_index]['wrong_link'] += 1
                print(f'✗ Wrong link: predicted {best_link}, true {true_link}')
            # break
        
        # Calculate per-link accuracy for this link
        if link_index in network_stats['per_link_results']:
            link_data = network_stats['per_link_results'][link_index]
            total_tests = link_data['correct'] + link_data['correct_link_wrong_pos'] + link_data['wrong_link']
            if total_tests > 0:
                accuracy = (link_data['correct'] / total_tests) * 100
                network_stats['per_link_accuracy'][link_index] = {
                    'accuracy': accuracy,
                    'correct': link_data['correct'],
                    'total': total_tests,
                    'mean_position_error': np.mean(link_data['position_errors']) if link_data['position_errors'] else 0
                }
    #     break
    # break

    
    # Save per-network statistics
    global_stats['per_network_stats'][seed] = network_stats
    
    # Calculate per-network accuracy
    if network_stats['total_predictions'] > 0:
        accuracy = network_stats['correct_predictions'] / network_stats['total_predictions'] * 100
        print(f"\n{'='*60}")
        print(f"Network seed {seed} Summary:")
        print(f"  Correct predictions: {network_stats['correct_predictions']}/{network_stats['total_predictions']} ({accuracy:.1f}%)")
        print(f"  Correct link, wrong position: {network_stats['correct_link_wrong_position']}")
        print(f"  Wrong link: {network_stats['wrong_link']}")
        if network_stats['position_errors']:
            print(f"  Mean position error: {np.mean(network_stats['position_errors']):.2e}")
            print(f"  Median position error: {np.median(network_stats['position_errors']):.2e}")
            print(f"  Max position error: {np.max(network_stats['position_errors']):.2e}")
        
        # Print per-link accuracy
        if network_stats['per_link_accuracy']:
            print(f"\n  Per-Link Accuracy:")
            for link_idx in sorted(network_stats['per_link_accuracy'].keys()):
                link_acc = network_stats['per_link_accuracy'][link_idx]
                print(f"    Link {link_idx}: {link_acc['accuracy']:.1f}% ({link_acc['correct']}/{link_acc['total']}) "
                      f"| Mean error: {link_acc['mean_position_error']:.2e}")
        print(f"{'='*60}\n")

# Save global statistics to file
output_stats_dir = '/home/baruva/network_cache/performance_stats'
os.makedirs(output_stats_dir, exist_ok=True)

# Save detailed statistics as JSON
stats_file = os.path.join(output_stats_dir, f'performance_stats_ni{ni}.json')
with open(stats_file, 'w') as f:
    # Convert numpy types to Python types for JSON serialization
    json_stats = {
        'global_stats': {
            'correct_predictions': int(global_stats['correct_predictions']),
            'correct_link_wrong_position': int(global_stats['correct_link_wrong_position']),
            'wrong_link': int(global_stats['wrong_link']),
            'total_predictions': int(global_stats['total_predictions']),
            'position_errors': [float(e) for e in global_stats['position_errors']],
        },
        'per_network_stats': {}
    }
    
    for seed, net_stats in global_stats['per_network_stats'].items():
        json_stats['per_network_stats'][str(seed)] = {
            'seed': int(net_stats['seed']),
            'ni': int(net_stats['ni']),
            'correct_predictions': int(net_stats['correct_predictions']),
            'correct_link_wrong_position': int(net_stats['correct_link_wrong_position']),
            'wrong_link': int(net_stats['wrong_link']),
            'total_predictions': int(net_stats['total_predictions']),
            'position_errors': [float(e) for e in net_stats['position_errors']],
            'per_link_results': {
                str(link_idx): {
                    'correct': int(link_data['correct']),
                    'correct_link_wrong_pos': int(link_data['correct_link_wrong_pos']),
                    'wrong_link': int(link_data['wrong_link']),
                    'position_errors': [float(e) for e in link_data['position_errors']]
                }
                for link_idx, link_data in net_stats['per_link_results'].items()
            }
        }
    
    json.dump(json_stats, f, indent=2)

print(f"\n{'='*60}")
print(f"GLOBAL STATISTICS (ni={ni}):")
print(f"{'='*60}")
if global_stats['total_predictions'] > 0:
    overall_accuracy = global_stats['correct_predictions'] / global_stats['total_predictions'] * 100
    print(f"Total predictions: {global_stats['total_predictions']}")
    print(f"Correct predictions: {global_stats['correct_predictions']} ({overall_accuracy:.1f}%)")
    print(f"Correct link, wrong position: {global_stats['correct_link_wrong_position']}")
    print(f"Wrong link: {global_stats['wrong_link']}")
    
    if global_stats['position_errors']:
        print(f"\nPosition Error Statistics:")
        print(f"  Mean: {np.mean(global_stats['position_errors']):.2e}")
        print(f"  Median: {np.median(global_stats['position_errors']):.2e}")
        print(f"  Std Dev: {np.std(global_stats['position_errors']):.2e}")
        print(f"  Min: {np.min(global_stats['position_errors']):.2e}")
        print(f"  Max: {np.max(global_stats['position_errors']):.2e}")
    
    # Find best and worst performing networks
    if len(global_stats['per_network_stats']) > 0:
        print(f"\nPer-Network Performance:")
        network_accuracies = []
        for seed, net_stats in global_stats['per_network_stats'].items():
            if net_stats['total_predictions'] > 0:
                acc = net_stats['correct_predictions'] / net_stats['total_predictions'] * 100
                network_accuracies.append((seed, acc, net_stats))
        
        if network_accuracies:
            network_accuracies.sort(key=lambda x: x[1], reverse=True)
            best_seed, best_acc, best_stats = network_accuracies[0]
            worst_seed, worst_acc, worst_stats = network_accuracies[-1]
            
            print(f"\n  Best performing network: seed {best_seed} with {best_acc:.1f}% accuracy")
            print(f"    Correct: {best_stats['correct_predictions']}/{best_stats['total_predictions']}")
            
            print(f"\n  Worst performing network: seed {worst_seed} with {worst_acc:.1f}% accuracy")
            print(f"    Correct: {worst_stats['correct_predictions']}/{worst_stats['total_predictions']}")

print(f"\nStatistics saved to: {stats_file}")
print(f"{'='*60}\n")


