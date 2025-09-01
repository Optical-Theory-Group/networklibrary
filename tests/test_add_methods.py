#!/usr/bin/env python3
"""
Test script for the add_node and add_link methods in the Network class.
"""

import numpy as np
from complex_network.components.node import Node
from complex_network.components.link import Link
from complex_network.networks.network import Network

def test_add_node_and_link():
    """Test the add_node and add_link methods."""
    
    # Create a simple network with 3 nodes
    nodes = {
        0: Node(0, "external", (0.0, 0.0)),
        1: Node(1, "internal", (1.0, 0.0)),
        2: Node(2, "external", (2.0, 0.0))
    }
    
    links = {
        0: Link(0, "internal", (0, 1)),
        1: Link(1, "internal", (1, 2))
    }
    
    # Create network
    network = Network(nodes, links)
    
    print(f"Initial network:")
    print(f"  Nodes: {len(network.nodes)}")
    print(f"  Links: {len(network.links)}")
    print(f"  Node positions: {[node.position for node in network.nodes]}")
    
    # Test add_node method
    print(f"\nTesting add_node method...")
    try:
        # Add a new node at position (1.5, 1.0) connected to nodes 1 and 2
        network.add_node(
            node_position=(1.5, 1.0),
            node_connections=[1, 2],
            node_type="internal"
        )
        print(f"✓ Successfully added node")
        print(f"  New network has {len(network.nodes)} nodes and {len(network.links)} links")
        
        # Check the new node
        new_node = network.nodes[-1]  # Should be the last node after reindexing
        print(f"  New node position: {new_node.position}")
        print(f"  New node connections: {new_node.sorted_connected_nodes}")
        
    except Exception as e:
        print(f"✗ Error adding node: {e}")
    
    # Test add_link method
    print(f"\nTesting add_link method...")
    try:
        # Get current node indices (after potential reindexing)
        node_indices = list(network.node_dict.keys())
        print(f"  Available node indices: {node_indices}")
        
        # Add a link between first and last nodes (if they're not already connected)
        first_node_idx = min(node_indices)
        last_node_idx = max(node_indices)
        
        # Check if they're already connected
        existing_connections = []
        for link in network.links:
            existing_connections.extend([
                (link.node_indices[0], link.node_indices[1]),
                (link.node_indices[1], link.node_indices[0])
            ])
        
        if (first_node_idx, last_node_idx) not in existing_connections:
            network.add_link([(first_node_idx, last_node_idx)])
            print(f"✓ Successfully added link between nodes {first_node_idx} and {last_node_idx}")
        else:
            print(f"! Nodes {first_node_idx} and {last_node_idx} are already connected")
            
        print(f"  Final network has {len(network.nodes)} nodes and {len(network.links)} links")
        
    except Exception as e:
        print(f"✗ Error adding link: {e}")
    
    print(f"\nFinal network state:")
    for i, node in enumerate(network.nodes):
        print(f"  Node {node.index}: pos={node.position}, connections={node.sorted_connected_nodes}")
    
    for i, link in enumerate(network.links):
        print(f"  Link {link.index}: connects nodes {link.node_indices}, length={link.length:.3f}")

if __name__ == "__main__":
    test_add_node_and_link()
