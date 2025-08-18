#!/usr/bin/env python3
"""Test the updated validation logic for reflection paths"""

from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_factory import generate_network
from complex_network.networks.network_paths import find_paths_to_target_distance

def test_reflection_validation():
    # Create a test network
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
    
    net = generate_network(spec)
    
    # Test with link (2,8)
    target_link = (2, 8)
    source_node = 10
    target_distance = 400e-6
    
    print(f"Testing with target link: {target_link}")
    print(f"Source node: {source_node}")
    print(f"Target distance: {target_distance*1e6:.1f}μm")
    print()
    
    # Find paths
    path_results = find_paths_to_target_distance(
        network=net,
        source_node_idx=source_node,
        target_link=target_link,
        target_distance=target_distance,
        max_hops=10,
        tolerance=1e-6,
        max_bounces=2
    )
    
    print(f"Found {len(path_results)} paths")
    
    # Check that all paths have at least one reflection
    reflection_count = 0
    transmission_only_count = 0
    
    for i, (path, position) in enumerate(path_results):  # Check first 10 paths
        has_reflection = 'R' in path
        has_only_transmission = 'T' in path and 'R' not in path
        
        if has_reflection:
            reflection_count += 1
        if has_only_transmission:
            transmission_only_count += 1
            
        print(f"Path {i+1}: {path}")
        print(f"  Has reflection: {has_reflection}")
        print(f"  Transmission only: {has_only_transmission}")
        print(f"  Position: {position*1e6:.2f}μm")
        print()
    
    print(f"Summary of first 10 paths:")
    print(f"  Paths with reflections: {reflection_count}")
    print(f"  Transmission-only paths: {transmission_only_count}")
    
    if transmission_only_count > 0:
        print("❌ ERROR: Found transmission-only paths!")
    else:
        print("✅ SUCCESS: All paths have reflection events!")

def test_transmission_before_reflection():
    """Test if any path has transmission events before reflection events."""
    # Create a test network
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
    
    net = generate_network(spec)
    
    # Test with link (2,8)
    target_link = (2, 8)
    source_node = 10
    target_distance = 400e-6
    
    print(f"\n{'='*60}")
    print(f"TESTING: Transmission before Reflection")
    print(f"Target link: {target_link}")
    print(f"Source node: {source_node}")
    print(f"Target distance: {target_distance*1e6:.1f}μm")
    print(f"{'='*60}")
    
    # Find paths
    path_results = find_paths_to_target_distance(
        network=net,
        source_node_idx=source_node,
        target_link=target_link,
        target_distance=target_distance,
        max_hops=10,
        tolerance=1e-6,
        max_bounces=2
    )
    
    print(f"Found {len(path_results)} paths to analyze")
    print()
    
    # Analyze all paths for transmission before reflection
    paths_with_t_before_r = []
    paths_with_r_before_t = []
    paths_with_only_r = []
    
    for i, (path, position) in enumerate(path_results):
        # Find first occurrence of 'T' and 'R'
        first_t_idx = None
        first_r_idx = None
        
        for j, element in enumerate(path):
            if element == 'T' and first_t_idx is None:
                first_t_idx = j
            if element == 'R' and first_r_idx is None:
                first_r_idx = j
        
        # Categorize the path
        if first_t_idx is not None and first_r_idx is not None:
            if first_t_idx < first_r_idx:
                paths_with_t_before_r.append((i, path, position, first_t_idx, first_r_idx))
            else:
                paths_with_r_before_t.append((i, path, position, first_r_idx, first_t_idx))
        elif first_r_idx is not None and first_t_idx is None:
            paths_with_only_r.append((i, path, position, first_r_idx))
    
    # Report results
    print(f"Analysis Results:")
    print(f"  Paths with T before R: {len(paths_with_t_before_r)}")
    print(f"  Paths with R before T: {len(paths_with_r_before_t)}")
    print(f"  Paths with only R (no T): {len(paths_with_only_r)}")
    print()
    
    # Show examples of paths with T before R
    if paths_with_t_before_r:
        print(f"✅ SUCCESS: Found {len(paths_with_t_before_r)} paths with transmission before reflection!")
        print("\nExamples of paths with T before R:")
        for i, (path_idx, path, position, t_idx, r_idx) in enumerate(paths_with_t_before_r[:5]):
            print(f"  Path {path_idx+1}: {path}")
            print(f"    First T at index {t_idx}, First R at index {r_idx}")
            print(f"    Position: {position*1e6:.2f}μm")
            print()
    else:
        print(f"❌ No paths found with transmission before reflection")
        print("This might indicate that the current path generation doesn't allow")
        print("crossing the target link before reflection occurs.")
        
        # Show some example paths for comparison
        print("\nExample paths found:")
        for i, (path_idx, path, position, r_idx, t_idx) in enumerate(paths_with_r_before_t[:3]):
            print(f"  Path {path_idx+1}: {path}")
            print(f"    First R at index {r_idx}, First T at index {t_idx}")
            print(f"    Position: {position*1e6:.2f}μm")
        print()


if __name__ == "__main__":
    test_reflection_validation()
    test_transmission_before_reflection()
