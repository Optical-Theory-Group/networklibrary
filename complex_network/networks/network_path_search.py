"""Module: network_paths_module

Provides utilities to enumerate optical path lengths and hop counts between nodes in a network and cache
those values to disk (.npy files).

Key features:
- deterministic cache fingerprinting based on NetworkSpec
- atomic .npy saves and loads for both path lengths and hop counts
- BFS path enumeration up to a specified hop limit with proper external node handling
- parallel computation using concurrent.futures (ProcessPoolExecutor)
- synchronized arrays where ith element in path length array corresponds to ith element in hop count array

Notes about design decisions:
- Worker functions are top-level to ensure they are picklable by multiprocessing.
- External nodes are treated as source/sink only - paths don't traverse through them
- Path enumeration allows revisiting internal nodes but handles external nodes specially
"""
from __future__ import annotations

import os
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Set

import numpy as np
import tempfile

from complex_network.networks.network_spec import NetworkSpec


# Data structures for discrete graph search sensing
@dataclass
class GraphData:
    """Lightweight graph representation for multiprocessing."""
    neighbors: Dict[int, List[int]]
    link_lengths: Dict[Tuple[int, int], float]
    internal_links: List[Tuple[int, int]]
    external_nodes: Set[int]
    internal_nodes: Set[int]


def extract_graph_data(network: Any) -> GraphData:
    """Extract lightweight graph data from the heavy network object for multiprocessing.
       The network object is already serializable, but this avoids passing the whole object."""
    neighbors: Dict[int, List[int]] = defaultdict(list)
    link_lengths: Dict[Tuple[int, int], float] = {}
    internal_links: List[Tuple[int, int]] = []
    
    external_nodes = {node.index for node in network.external_nodes}
    internal_nodes = {node.index for node in network.internal_nodes}
    
    # Process all links
    for link in network.links:
        a, b = link.sorted_connected_nodes
        length = link.length
        
        link_lengths[(a,b)] = length
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
        
        if a in internal_nodes and b in internal_nodes:
            internal_links.append((a,b))
    
    # Ensure deterministic ordering
    for n in list(neighbors.keys()):
        neighbors[n] = sorted(set(neighbors[n]))
    
    return GraphData(
        neighbors=dict(neighbors),
        link_lengths=link_lengths,
        internal_links=internal_links,
        external_nodes=external_nodes,
        internal_nodes=internal_nodes
    )

# ________________________Enhanced Detailed Path Enumeration with Coherence Grouping_______________________________

@dataclass
class PathGroup:
    """Group of paths with similar lengths within coherence length tolerance."""
    representative_length: float
    paths: List[Tuple[List[int], float]]  # [(path_nodes, path_length), ...]
    
    def add_path(self, path: List[int], length: float) -> None:
        """Add a path to this group."""
        self.paths.append((path, length))
    
    def get_representative_path(self) -> Tuple[List[int], float]:
        """For each group saved as x = (path, length), we will take the absolute
           value of difference between all the lengths and the representative length
            and then return the path with minimum difference."""
        if not self.paths:
            # No paths, return empty and 0
            return [], 0.0
        return min(self.paths, key=lambda x: abs(x[1] - self.representative_length))


@dataclass 
class CoherentPathData:
    """Path data structure grouping paths that come within the coherence length."""
    source_idx: int
    target_idx: int
    coherence_length: float
    path_groups: List[PathGroup]
    total_paths: int
    
    def get_all_paths(self) -> List[Tuple[List[int], float]]:
        """Get all paths across all groups."""
        all_paths = []
        for group in self.path_groups:
            all_paths.extend(group.paths)
        return all_paths


def find_all_paths_bfs(
    graph: GraphData,
    start: int,
    end: int,
    max_hops: int
) -> List[Tuple[List[int], float]]:
    """Find all paths using BFS with the GraphData structure.
        A cleaner wrapper around the core BFS function."""
    return _find_paths_bfs(
        graph.neighbors,
        graph.link_lengths,
        graph.external_nodes,
        start,
        end,
        max_hops
    )


def prepare_detailed_path_cache(
    graph: GraphData,
    fingerprint: str,
    source_idx: int,
    max_hops: int,
    force_rebuild: bool = False
) -> Dict[Tuple[int, int, int], List[Tuple[List[int], float]]]:
    """Prepare path cache for fault detection.
        Args:
        graph: GraphData object with neighbors, link lengths, and node sets
        fingerprint: Unique fingerprint string for the network spec
        source_idx: Index of the source node
        max_hops: Maximum number of hops to explore
        force_rebuild: If True, ignore existing cache and rebuild"""
    
    
    if not force_rebuild:
        cached = load_detailed_path_cache(fingerprint, source_idx, max_hops)
        if cached is not None:
            return cached

    # Build cache by finding paths to all internal nodes
    path_cache = {}
    
    for target in graph.internal_nodes:
        if target == source_idx:
            continue
                   
        # Paths from source to target
        # (source_idx, target) is the key and that key will map to list of paths from source to target
        paths_to = find_all_paths_bfs(graph, source_idx, target, max_hops // 2)
        if paths_to:
            path_cache[(source_idx, target)] = paths_to
            
            # Since paths are symmetric in optical networks, we can derive the reverse paths
            # by reversing each path and swapping the path length (which remains the same)
            paths_from = [(path[::-1], length) for path, length in paths_to]
            path_cache[(target, source_idx)] = paths_from
    
    # Save to disk cache
    save_detailed_path_cache(fingerprint, source_idx, max_hops, path_cache)
    
    return path_cache
# ____________________________________cache helpers___________________________________________

def _spec_fingerprint(spec: NetworkSpec, salt: Optional[str] = None) -> str:
    """Generate a stable fingerprint string for a NetworkSpec-like object.

    So that networks with identical specs will share the same cache directory.
    """
    ni = getattr(spec, "num_internal_nodes", "NA")
    ne = getattr(spec, "num_external_nodes", "NA")
    net_type = getattr(spec, "network_type", "NA")
    net_shape = getattr(spec, "network_shape", "NA")
    net_size = getattr(spec, "network_size", "NA")
    ext_size = getattr(spec, "external_size", "NA")
    ext_off = getattr(spec, "external_offset", "NA")
    node_s = getattr(spec, "node_S_mat_type", "NA")
    seed = getattr(spec, "random_seed", "NA")

    base = (
        f"ni={ni}-ne={ne}-net_type={net_type}-net_shape={net_shape}"
        f"-net_size={net_size}-ext_size={ext_size}-ext_off={ext_off}-node_s={node_s}-seed={seed}"
    )
    if salt:
        base += f"-salt={salt}"
    return base


# function for making sure cache directory exists
# if not, create it
def _cache_dir() -> str:
    d = os.path.join(os.getcwd(), ".path_cache")
    os.makedirs(d, exist_ok=True)
    return d

# function for making sure fingerprint subdirectory exists
def _ensure_fingerprint_dir(fingerprint: str) -> str:
    base = _cache_dir()
    sub = os.path.join(base, str(fingerprint))
    os.makedirs(sub, exist_ok=True)
    return sub

def load_path_cache(fingerprint: str, source_idx: int, target_idx: int, max_hops: int) -> Optional[np.ndarray]:
    """Load cached path lengths (.npy) for given fingerprint/source/target/hops.

    Returns None if the file isn't present or can't be loaded.
    """
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"pl_s{source_idx}_t{target_idx}_h{max_hops}.npy")
    if os.path.exists(fn):
        try:
            return np.load(fn)
        except Exception:
            return None
    return None


def load_hop_counts_cache(fingerprint: str, source_idx: int, target_idx: int, max_hops: int) -> Optional[np.ndarray]:
    """Load cached hop counts (.npy) for given fingerprint/source/target/hops.

    Returns None if the file isn't present or can't be loaded.
    """
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"hc_s{source_idx}_t{target_idx}_h{max_hops}.npy")
    if os.path.exists(fn):
        try:
            return np.load(fn)
        except Exception:
            return None
    return None


def save_path_lengths_cache(fingerprint: str,
                            source_idx: int,
                            target_idx: int,
                            max_hops: int,
                            lengths: List[float]) -> None:
    """Atomically save path lengths as a .npy file in the fingerprint cache directory.

    Uses a temp file created via tempfile.mkstemp (in same directory), writes via
    an open binary file handle (so np.save doesn't add '.npy'), then os.replace.
    Cleans up on error.
    """
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"pl_s{source_idx}_t{target_idx}_h{max_hops}.npy")
    arr = np.array(lengths, dtype=float)

    # Create a temp file (unique) in same directory
    fd, tmp_path = tempfile.mkstemp(dir=sub)
    # mkstemp returns an open FD; close it and open with python file object instead
    os.close(fd)
    try:
        # Write using open file handle so np.save will not append ".npy"
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        # Atomic replace (rename across same filesystem)
        os.replace(tmp_path, fn)
    except Exception:
        # cleanup any leftover tmp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def save_hop_counts_cache(fingerprint: str,
                         source_idx: int,
                         target_idx: int,
                         max_hops: int,
                         hop_counts: List[int]) -> None:
    """Atomically save hop counts as a .npy file in the fingerprint cache directory.

    Uses a temp file created via tempfile.mkstemp (in same directory), writes via
    an open binary file handle (so np.save doesn't add '.npy'), then os.replace.
    Cleans up on error.
    """
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"hc_s{source_idx}_t{target_idx}_h{max_hops}.npy")
    arr = np.array(hop_counts, dtype=int)

    # Create a temp file (unique) in same directory
    fd, tmp_path = tempfile.mkstemp(dir=sub)
    # mkstemp returns an open FD; close it and open with python file object instead
    os.close(fd)
    try:
        # Write using open file handle so np.save will not append ".npy"
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        # Atomic replace (rename across same filesystem)
        os.replace(tmp_path, fn)
    except Exception:
        # cleanup any leftover tmp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def save_scattering_coeffs_cache(fingerprint: str,
                                 source_idx: int,
                                 target_idx: int,
                                 max_hops: int,
                                 scattering_coeffs: List[float]) -> None:
    """Atomically save scattering coefficients as a .npy file in the fingerprint cache directory.

    Uses a temp file created via tempfile.mkstemp (in same directory), writes via
    an open binary file handle (so np.save doesn't add '.npy'), then os.replace.
    Cleans up on error.
    """
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"ss_s{source_idx}_t{target_idx}_h{max_hops}.npy")
    arr = np.array(scattering_coeffs, dtype=float)

    # Create a temp file (unique) in same directory
    fd, tmp_path = tempfile.mkstemp(dir=sub)
    # mkstemp returns an open FD; close it and open with python file object instead
    os.close(fd)
    try:
        # Write using open file handle so np.save will not append ".npy"
        with open(tmp_path, "wb") as f:
            np.save(f, arr)
        # Atomic replace (rename across same filesystem)
        os.replace(tmp_path, fn)
    except Exception:
        # cleanup any leftover tmp file
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def load_scattering_coeffs_cache(fingerprint: str, source_idx: int, target_idx: int, max_hops: int) -> Optional[np.ndarray]:
    """Load cached scattering coefficients (.npy) for given fingerprint/source/target/hops.

    Returns None if the file isn't present or can't be loaded.
    """
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"ss_s{source_idx}_t{target_idx}_h{max_hops}.npy")
    if os.path.exists(fn):
        try:
            return np.load(fn)
        except Exception:
            return None
    return None

# detailed path cache functions
def load_detailed_path_cache(fingerprint: str, source_idx: int, max_hops: int) -> Optional[Dict]:
    """Load detailed path cache for fault detection."""
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"detailed_paths_s{source_idx}_h{max_hops}.npy")
    if os.path.exists(fn):
        try:
            return np.load(fn, allow_pickle=True).item()
        except Exception:
            return None
    return None


def save_detailed_path_cache(fingerprint: str, source_idx: int, max_hops: int, cache_data: Dict) -> None:
    """Save detailed path cache for fault detection."""
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"detailed_paths_s{source_idx}_h{max_hops}.npy")
    
    # Create a temp file (unique) in same directory
    fd, tmp_path = tempfile.mkstemp(dir=sub)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, cache_data)
        os.replace(tmp_path, fn)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise

def group_paths_by_coherence(paths: List[List[int]], 
                           lengths: List[float], 
                           coherence_length: float = 1e-6) -> List[PathGroup]:
    """Group paths by coherence length tolerance (L_c/2 by default).
    
    Args:
        paths: List of path node sequences [s, n1, n2, ..., target] 
        s: source_node, n_i: internal nodes, target: target_node.
        lengths: Corresponding path lengths
        coherence_length: Coherence length in meters 
        (set to 1um because a bandwidth of 400nm at 1000nm wavelength gives ~1um coherence length)
        
    Returns:
        List of PathGroup objects, each containing paths within coherence tolerance
    """
    if not paths or not lengths:
        return []
        
    tolerance = coherence_length / 2  # L_c/2 = 0.5um by default
    
    # Sort by length for efficient grouping
    sorted_indices = sorted(range(len(paths)), key=lambda i: lengths[i])
    
    groups = []
    current_group = None
    
    for idx in sorted_indices:
        path = paths[idx]
        length = lengths[idx]
        
        if current_group is None or length > current_group.representative_length + tolerance:
            # Start new group when we go outside the tolerance
            current_group = PathGroup(representative_length=length, paths=[])
            groups.append(current_group)
        
        current_group.add_path(path, length)
    
    return groups


def load_grouped_path_cache(fingerprint: str, 
                           source_idx: int, 
                           target_idx: int, 
                           max_hops: int) -> Optional[CoherentPathData]:
    """Load cached grouped path data."""
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"grouped_paths_s{source_idx}_t{target_idx}_h{max_hops}.npy")
    if os.path.exists(fn):
        try:
            data = np.load(fn, allow_pickle=True).item()
            return data
        except Exception:
            return None
    return None


def save_grouped_path_cache(fingerprint: str,
                           source_idx: int,
                           target_idx: int,
                           max_hops: int,
                           coherent_data: CoherentPathData) -> None:
    """Save coherent path data to cache."""
    sub = _ensure_fingerprint_dir(fingerprint)
    fn = os.path.join(sub, f"grouped_paths_s{source_idx}_t{target_idx}_h{max_hops}.npy")
    
    # Create a temp file (unique) in same directory
    fd, tmp_path = tempfile.mkstemp(dir=sub)
    os.close(fd)
    try:
        with open(tmp_path, "wb") as f:
            np.save(f, coherent_data)
        os.replace(tmp_path, fn)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise


def compute_path_scattering_coefficient(path: List[int], 
                                       neighbors: Dict[int, List[int]], 
                                       network_nodes: Dict[int, Any],
                                       external_nodes: Set[int],
                                       k0: complex = 6.28e6) -> float:
    """Compute the scattering coefficient for a given path in the network.
    
    Args:
        path: Sequence of node indices representing the path
        neighbors: Dict mapping node -> list of neighbor nodes (in fixed order)
        network_nodes: Dict mapping node index -> node object with scattering matrix
        external_nodes: Set of external node indices
        k0: Wavenumber for scattering matrix computation in the node method
        
    Returns:
        Absolute value of the total scattering coefficient for the path
    """
    # No hops, no scattering
    if 0 < len(path) < 2:
        return 1.0

    if len(path) == 0:
        raise ValueError("Path must have at least one node")

    # Start with unit amplitude (complex) can accumulate phase/amplitude changes
    total_amplitude = 1.0 + 0j
    
    for i in range(len(path) - 1):
        current_node = path[i]
        next_node = path[i + 1]
        
        # Get the scattering matrix for the current node
        node_obj = network_nodes[current_node]
        
        # For internal nodes, get_S is callable; for external nodes, it's a numpy array
        if node_obj.type == 'internal':
            S_matrix = node_obj.get_S(k0)

        elif node_obj.type == 'external':
            S_matrix = node_obj.get_S
        else:
            raise ValueError(f"Unknown node type {node_obj.type} for node {current_node}")
        
        # Get the neighbor list for port mapping
        node_neighbors = neighbors[current_node]
        
        # Find output port (where we're going to)
        try:
            output_port = node_neighbors.index(next_node)
        except ValueError:
            raise ValueError(f"Node {next_node} not found in neighbors of node {current_node}")
        
        # if the first node, it is entering the network from the external node
        if i == 0:
            """
            For external nodes: signal enters from external side (port 0) and exits to network side (port 1).
            This assumes external nodes have 2 ports: port 0 (external) and port 1 (network).
            """
            if current_node in external_nodes:
                input_port = 0  
                output_port = 1
            else:
                raise ValueError(f"First node {current_node} should be an external node")
        else:
            # Subsequent hops: came from previous node
            prev_node = path[i - 1]
            try:
                input_port = node_neighbors.index(prev_node)
            except ValueError:
                print(f"Error: Previous node {prev_node} not found in neighbors {node_neighbors} of node {current_node}")
                raise ValueError(f"Previous node {prev_node} not found in neighbors of node {current_node}")
        
        # Validate matrix dimensions
        if output_port >= S_matrix.shape[0] or input_port >= S_matrix.shape[1]:
            print(f"Error: Port indices out of bounds: output_port={output_port}, input_port={input_port}, matrix_shape={S_matrix.shape}")
            raise ValueError(f"Port indices out of bounds: output_port={output_port}, input_port={input_port}, matrix_shape={S_matrix.shape}")
        
        # Get scattering coefficient
        scattering_coeff = S_matrix[output_port, input_port]
        
        total_amplitude *= scattering_coeff    
    result = abs(total_amplitude)
    return result

# ________________________Path enumeration_______________________________

def find_all_path_info_bfs(
    neighbors: Dict[int, List[int]],
    link_lengths: Dict[Tuple[int, int], float],
    external_nodes: Set[int],
    internal_nodes: Dict[int, Any],
    start: int,
    end: int,
    max_hops: int,
    k0: complex = 6.28e6,
) -> Tuple[List[float], List[int], List[float], List[List[int]]]:
    """Breadth-first enumeration of path geometric lengths, hop counts, scattering coefficients, and paths.
    
    Key optical network constraints:
    1. External nodes can only be sources or final destinations
    2. Paths cannot traverse "through" external nodes, light enters and exists the system through these nodes
    3. Internal nodes can be revisited multiple times
    4. Source node cannot be revisited in internal portions of paths

    Args:
        neighbors: Dict mapping node -> list of neighbor nodes
        link_lengths: Dict mapping (min_node, max_node) -> length
        external_nodes: Set of external node indices
        internal_nodes: Dict mapping node index -> node object with scattering matrix
        start: Source node index
        end: Target node index
        max_hops: Maximum number of hops to explore
        k0: Wavenumber for scattering matrix computation
        
    Returns:
        Tuple of (geometric path lengths, hop counts, scattering coefficients, paths) 
        where ith element in each list corresponds to the same path
    """
    if end in external_nodes and end != start:
        # Cannot reach other external nodes in optical networks
        return [], [], [], []
        
    lengths: List[float] = []
    hop_counts: List[int] = []
    scattering_coeffs: List[float] = []
    paths: List[List[int]] = []
    
    # Queue entries: (current_node, hops_so_far, geometric_length, visited_path)
    queue = deque([(start, 0, 0.0, [start])])
    
    while queue:
        current, hops, geom_len, path = queue.popleft()
        
        if hops > max_hops:
            continue
            
        # Check if we've reached the target (with at least one hop)
        if current == end and hops > 0:
            lengths.append(geom_len)
            hop_counts.append(hops)
            
            # Compute scattering coefficient for this path
            try:
                scatt_coeff = compute_path_scattering_coefficient(path, neighbors, internal_nodes, external_nodes, k0)
                scattering_coeffs.append(scatt_coeff)
                paths.append(path[:])  # Store a copy of the path
            except Exception as e:
                print(f"Warning: Could not compute scattering coefficient for path {path}: {e}")
                scattering_coeffs.append(0.0)  # Default to 0 if computation fails
                paths.append(path[:])
            
            # Continue exploring from here if target is internal (allows return paths)
            if end not in external_nodes:
                pass  # Continue processing below
            else:
                continue  # External nodes are terminal
                
        # Explore neighbors
        for neighbor in neighbors.get(current, []):
            # Skip if this would be too many hops
            if hops + 1 > max_hops:
                continue
                
            # External node constraints
            if neighbor in external_nodes:
                # Can only go to external nodes if:
                # 1. It's the target we're seeking, OR  
                # 2. It's the source node and we're doing a return path
                if neighbor != end and neighbor != start:
                    continue
                # Don't traverse through external nodes
                if neighbor != end and len(path) > 1:
                    continue
                    
            # Calculate link length
            lk = (min(current, neighbor), max(current, neighbor))
            seg_len = link_lengths.get(lk, 0.0)
            
            new_path = path + [neighbor]
            queue.append((neighbor, hops + 1, geom_len + seg_len, new_path))
    
    return lengths, hop_counts, scattering_coeffs, paths

def find_all_path_lengths_bfs(
    neighbors: Dict[int, List[int]],
    link_lengths: Dict[Tuple[int, int], float],
    external_nodes: Set[int],
    start: int,
    end: int,
    max_hops: int,
) -> Tuple[List[float], List[int]]:
    """Breadth-first enumeration of path geometric lengths and hop counts from start to end.
    
    Key optical network constraints:
    1. External nodes can only be sources or final destinations
    2. Paths cannot traverse "through" external nodes, light enters and exists the system through these nodes
    3. Internal nodes can be revisited multiple times
    4. Source node can be revisited in internal portions of paths
    
    Args:
        neighbors: Dict mapping node -> list of neighbor nodes
        link_lengths: Dict mapping (min_node, max_node) -> length
        external_nodes: Set of external node indices
        start: Source node index
        end: Target node index
        max_hops: Maximum number of hops to explore
        
    Returns:
        Tuple of (geometric path lengths, hop counts) where ith element in each list corresponds to the same path
    """
    if end in external_nodes and end != start:
        # Cannot reach other external nodes in optical networks
        return [], []
        
    lengths: List[float] = []
    hop_counts: List[int] = []
    # Queue entries: (current_node, hops_so_far, geometric_length, visited_path)
    queue = deque([(start, 0, 0.0, [start])])
    
    while queue:
        current, hops, geom_len, path = queue.popleft()
        
        if hops > max_hops:
            continue
            
        # Check if we've reached the target (with at least one hop)
        if current == end and hops > 0:
            lengths.append(geom_len)
            hop_counts.append(hops)
            # Continue exploring from here if target is internal (allows return paths)
            if end not in external_nodes:
                pass  # Continue processing below
            else:
                continue  # External nodes are terminal
                
        # Explore neighbors
        for neighbor in neighbors.get(current, []):
            # Skip if this would be too many hops
            if hops + 1 > max_hops:
                continue
                
            # External node constraints
            if neighbor in external_nodes:
                # Can only go to external nodes if:
                # 1. It's the target we're seeking, OR  
                # 2. It's the source node and we're doing a return path
                if neighbor != end and neighbor != start:
                    continue
                # Don't traverse through external nodes
                if neighbor != end and len(path) > 1:
                    continue
                    
            # Calculate link length
            lk = (min(current, neighbor), max(current, neighbor))
            seg_len = link_lengths.get(lk, 0.0)
            
            new_path = path + [neighbor]
            queue.append((neighbor, hops + 1, geom_len + seg_len, new_path))
    
    return lengths, hop_counts


# Worker helper must be top-level so it can be pickled by multiprocessing
def _compute_target_and_save_star(args: Tuple[int, Dict[int, List[int]],
                                        Dict[Tuple[int, int], float],
                                        Set[int], Dict[int, Any],
                                        int, str, int, complex]):
    """Unpack args, compute all lengths, hop counts, and scattering coeffs from source->target,
       save to cache, return (target, count)."""
    target, neighbors, link_lengths, external_nodes, internal_nodes, source_idx, fingerprint, max_hops, k0 = args

    lengths, hop_counts, scattering_coeffs, paths = find_all_path_info_bfs(
        neighbors, link_lengths, external_nodes, internal_nodes,
        source_idx, target, max_hops, k0
    )
    
    if lengths:  # Only save non-empty results
        save_path_lengths_cache(fingerprint, source_idx, target, max_hops, lengths)
        save_hop_counts_cache(fingerprint, source_idx, target, max_hops, hop_counts)
        save_scattering_coeffs_cache(fingerprint, source_idx, target, max_hops, scattering_coeffs)
    
    return target, len(lengths)


def _compute_coherent_paths_star(args: Tuple[int, Dict[int, List[int]],
                                       Dict[Tuple[int, int], float],
                                       Set[int], Dict[int, Any],
                                       int, str, int, complex, float]):
    """Worker function for computing coherent path data with length grouping."""
    target, neighbors, link_lengths, external_nodes, network_nodes, source_idx, fingerprint, max_hops, k0, coherence_length = args
    
    # Get all path information 
    lengths, hop_counts, scattering_coeffs, paths = find_all_path_info_bfs(
        neighbors, link_lengths, external_nodes, network_nodes,
        source_idx, target, max_hops, k0
    )
    
    # Only process non-empty results
    if lengths:  
        path_groups = group_paths_by_coherence(paths, lengths, coherence_length)
        
        # Create coherent path data structure
        coherent_data = CoherentPathData(
            source_idx=source_idx,
            target_idx=target,
            coherence_length=coherence_length,
            path_groups=path_groups,
            total_paths=len(paths)
        )
        
        # Save to cache
        save_grouped_path_cache(fingerprint, source_idx, target, max_hops, coherent_data)
        
        # Also save legacy format for backward compatibility
        save_path_lengths_cache(fingerprint, source_idx, target, max_hops, lengths)
        save_hop_counts_cache(fingerprint, source_idx, target, max_hops, hop_counts)
        save_scattering_coeffs_cache(fingerprint, source_idx, target, max_hops, scattering_coeffs)
        
        return target, len(paths), len(path_groups)
    
    return target, 0, 0


def compute_grouped_path_cache_all(network: Any, 
                                  source_node_idx: int, 
                                  fingerprint: str, 
                                  max_hops: int, 
                                  coherence_length: float = 1e-6,
                                  k0: complex = 6.28e6, 
                                  n_workers: Optional[int] = None, 
                                  use_processes: bool = True) -> None:
    """Compute coherent path data from source to all internal nodes with coherence length grouping.

    This enhanced version stores paths as node sequences [s, n1, n2, ..., target] and groups them
    by coherence length tolerance (L_c/2 = 0.5um by default). Results are saved as:
    - coherent_paths_s{source}_t{target}_h{max_hops}.npy (coherent path data with grouping)
    - Legacy format files for backward compatibility

    Parameters
    ----------
    network: object
        Network object with nodes and links
    source_node_idx: int
        Source node index (should be external node)
    fingerprint: str
        Cache fingerprint to write results under
    max_hops: int
        Maximum number of hops to explore
    coherence_length: float
        Coherence length in meters (default: 1um = 1e-6m)
    k0: complex
        Wavenumber for scattering matrix computation (default: 6.28e6)
    n_workers: Optional[int]
        Number of parallel workers to use. If None, uses max(1, cpu_count()-2).
    use_processes: bool
        If True, uses ProcessPoolExecutor (default). If False, uses ThreadPoolExecutor.
    """

    # Get node types directly from the network object
    external_nodes = {node.index for node in network.external_nodes}
    internal_nodes = {node.index for node in network.internal_nodes}
    
    print(f"Computing coherent paths with L_c = {coherence_length*1e6:.1f}um (grouping tolerance: {coherence_length*0.5*1e6:.1f}um)")
    print(f"Identified {len(external_nodes)} external nodes: {sorted(external_nodes)}")
    print(f"Identified {len(internal_nodes)} internal nodes: {sorted(internal_nodes)}")
    
    if source_node_idx not in external_nodes:
        print(f"Warning: Source node {source_node_idx} is not identified as external node")
    
    # Build lightweight graph representation
    neighbors: Dict[int, List[int]] = defaultdict(list)
    link_lengths: Dict[Tuple[int, int], float] = {}
    
    # Build node mapping for scattering matrices
    network_nodes: Dict[int, Any] = {}
    for node in network.external_nodes + network.internal_nodes:
        network_nodes[node.index] = node

    # Collect links directly from network properties
    links = network.internal_links + network.external_links

    if not links:
        raise RuntimeError("Network object has no links")

    # Build graph from links
    for link in links:
        a, b = link.sorted_connected_nodes
        L = link.length
        
        k = (min(int(a), int(b)), max(int(a), int(b)))
        link_lengths[k] = L
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))

    # Ensure deterministic ordering of neighbors
    for n in list(neighbors.keys()):
        neighbors[n] = sorted(set(neighbors[n]))

    # Only compute paths to internal nodes
    targets = sorted(internal_nodes)

    # Filter out targets that have already been cached
    missing_targets = []
    for target in targets:
        coherent_cache = load_grouped_path_cache(fingerprint, source_node_idx, target, max_hops)
        if coherent_cache is None:
            missing_targets.append(target)

    if not missing_targets:
        print("All coherent path data is already cached.")
        return
    
    print(f'Found {len(missing_targets)} targets to compute coherent paths from source {source_node_idx} with max_hops {max_hops}')
    print('Generating coherent path data may take a while, Please Wait .... ')

    # Determine worker count
    if n_workers is None:
        try:
            cpu_count = os.cpu_count() or 1
            n_workers = max(1, cpu_count - 2)
        except Exception:
            n_workers = 1

    # Ensure cache dir exists
    _ensure_fingerprint_dir(fingerprint)

    # Prepare arg tuples for worker function
    args_list = [
        (target, dict(neighbors), dict(link_lengths), external_nodes, dict(network_nodes),
         source_node_idx, fingerprint, max_hops, k0, coherence_length)
        for target in missing_targets
    ]

    if use_processes and n_workers > 1:
        Executor = ProcessPoolExecutor
    else:
        # Fallback to threaded execution
        from concurrent.futures import ThreadPoolExecutor
        Executor = ThreadPoolExecutor

    with Executor(max_workers=n_workers) as ex:
        # Process results as they complete
        for target, path_count, group_count in ex.map(_compute_coherent_paths_star, args_list):
            if path_count > 0:
                print(f"Computed coherent paths from source {source_node_idx} -> target {target}: {path_count} paths in {group_count} coherence groups")
            else:
                print(f"No valid paths found from source {source_node_idx} -> target {target}")


def get_coherent_paths_to_target(network: Any,
                                source_node_idx: int, 
                                target_node_idx: int,
                                max_hops: int,
                                coherence_length: float = 1e-6,
                                force_rebuild: bool = False) -> Optional[CoherentPathData]:
    """Get coherent path data from source to target, computing if necessary.
    
    Args:
        network: Network object
        source_node_idx: Source node index
        target_node_idx: Target node index  
        max_hops: Maximum hops to explore
        coherence_length: Coherence length in meters (default: 1um)
        force_rebuild: Force recomputation even if cached data exists
        
    Returns:
        CoherentPathData object or None if no paths found
    """
    fingerprint = _spec_fingerprint(network.spec)
    
    if not force_rebuild:
        cached_data = load_grouped_path_cache(fingerprint, source_node_idx, target_node_idx, max_hops)
        if cached_data is not None:
            return cached_data
    
    # Compute single target 
    print(f"Computing coherent paths from {source_node_idx} to {target_node_idx}...")
    
    # Extract graph data
    graph_data = extract_graph_data(network)
    
    # Build node mapping for scattering matrices
    network_nodes: Dict[int, Any] = {}
    for node in network.external_nodes + network.internal_nodes:
        network_nodes[node.index] = node
    
    # Get path info
    lengths, hop_counts, scattering_coeffs, paths = find_all_path_info_bfs(
        graph_data.neighbors, graph_data.link_lengths, graph_data.external_nodes, 
        network_nodes, source_node_idx, target_node_idx, max_hops
    )
    
    if not lengths:
        return None
        
    # Group by coherence length
    path_groups = group_paths_by_coherence(paths, lengths, coherence_length, tolerance_factor=0.5)
    
    # Create coherent data structure
    coherent_data = CoherentPathData(
        source_idx=source_node_idx,
        target_idx=target_node_idx,
        coherence_length=coherence_length,
        path_groups=path_groups,
        total_paths=len(paths)
    )
    
    # Save to cache
    save_grouped_path_cache(fingerprint, source_node_idx, target_node_idx, max_hops, coherent_data)
    
    return coherent_data


def print_coherent_path_summary(coherent_data: CoherentPathData) -> None:
    """Print a summary of coherent path data for analysis.
    
    Args:
        coherent_data: CoherentPathData object to summarize
    """
    print(f"\nCoherent Path Summary:")
    print(f"Source: {coherent_data.source_idx} -> Target: {coherent_data.target_idx}")
    print(f"Coherence Length: {coherent_data.coherence_length*1e6:.1f}um")
    print(f"Total Paths: {coherent_data.total_paths}")
    print(f"Coherence Groups: {len(coherent_data.path_groups)}")
    
    print(f"\nGroup Details:")
    for i, group in enumerate(coherent_data.path_groups):
        rep_path, rep_length = group.get_representative_path()
        length_range = (
            min(length for _, length in group.paths),
            max(length for _, length in group.paths)
        )
        print(f"  Group {i+1}: {len(group.paths)} paths")
        print(f"    Length range: {length_range[0]*1e6:.3f} - {length_range[1]*1e6:.3f} um")
        print(f"    Representative: {rep_length*1e6:.3f} um")
        print(f"    Example path: {rep_path}")


def get_paths_in_length_range(coherent_data: CoherentPathData, 
                            min_length: float, 
                            max_length: float) -> List[Tuple[List[int], float]]:
    """Get all paths within a specific length range.
    
    Args:
        coherent_data: CoherentPathData object
        min_length: Minimum path length (meters)
        max_length: Maximum path length (meters)
        
    Returns:
        List of (path, length) tuples within the specified range
    """
    filtered_paths = []
    for group in coherent_data.path_groups:
        for path, length in group.paths:
            if min_length <= length <= max_length:
                filtered_paths.append((path, length))
    
    return filtered_paths


def get_coherent_groups_for_analysis(coherent_data: CoherentPathData) -> List[Dict]:
    """Extract coherent group information for further analysis.
    
    Args:
        coherent_data: CoherentPathData object
        
    Returns:
        List of dictionaries with group statistics
    """
    group_info = []
    
    for i, group in enumerate(coherent_data.path_groups):
        lengths = [length for _, length in group.paths]
        paths = [path for path, _ in group.paths]
        hop_counts = [len(path) - 1 for path in paths]
        
        info = {
            'group_id': i,
            'num_paths': len(group.paths),
            'representative_length': group.representative_length,
            'min_length': min(lengths),
            'max_length': max(lengths),
            'mean_length': sum(lengths) / len(lengths),
            'length_std': np.std(lengths) if len(lengths) > 1 else 0.0,
            'min_hops': min(hop_counts),
            'max_hops': max(hop_counts),
            'mean_hops': sum(hop_counts) / len(hop_counts),
            'example_path': group.get_representative_path()[0]
        }
        group_info.append(info)
    
    return group_info


def compute_cache_all_path_info(network: Any, source_node_idx: int, fingerprint: str, max_hops: int, k0: complex = 6.28e6, n_workers: Optional[int] = None, use_processes: bool = True) -> None:
    """Compute path-length distributions, hop counts, and scattering coefficients from source_node_idx to every reachable node.

    Results are saved per-target as .npy files inside the cache fingerprint directory.
    Path lengths, hop counts, and scattering coefficients are saved as:
    - pl_s{source}_t{target}_h{max_hops}.npy (path lengths)
    - hc_s{source}_t{target}_h{max_hops}.npy (hop counts) 
    - ss_s{source}_t{target}_h{max_hops}.npy (scattering coefficients)
    Only computes paths to internal nodes (external nodes other than source are unreachable).

    Parameters
    ----------
    network: object
        Network object with nodes and links
    source_node_idx: int
        Source node index (should be external node)
    fingerprint: str
        Cache fingerprint to write results under
    max_hops: int
        Maximum number of hops to explore
    k0: complex
        Wavenumber for scattering matrix computation (default: 6.28e6)
    n_workers: Optional[int]
        Number of parallel workers to use. If None, uses max(1, cpu_count()-2).
    use_processes: bool
        If True, uses ProcessPoolExecutor (default). If False, uses ThreadPoolExecutor.
    """

    # Get node types directly from the network object
    external_nodes = {node.index for node in network.external_nodes}
    internal_nodes = {node.index for node in network.internal_nodes}
    
    print(f"Identified {len(external_nodes)} external nodes: {sorted(external_nodes)}")
    print(f"Identified {len(internal_nodes)} internal nodes: {sorted(internal_nodes)}")
    
    if source_node_idx not in external_nodes:
        print(f"Warning: Source node {source_node_idx} is not identified as external node")
    
    # Build lightweight graph representation
    neighbors: Dict[int, List[int]] = defaultdict(list)
    link_lengths: Dict[Tuple[int, int], float] = {}
    
    # Build node mapping for scattering matrices
    network_nodes: Dict[int, Any] = {}
    for node in network.external_nodes + network.internal_nodes:
        network_nodes[node.index] = node

    # Collect links directly from network properties
    links = network.internal_links + network.external_links

    if not links:
        raise RuntimeError("Network object has no links")

    # Build graph from links
    for link in links:
        a, b = link.sorted_connected_nodes
        L = link.length
        
        k = (min(int(a), int(b)), max(int(a), int(b)))
        link_lengths[k] = L
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))

    # Ensure deterministic ordering of neighbors
    for n in list(neighbors.keys()):
        neighbors[n] = sorted(set(neighbors[n]))

    # Only compute paths to internal nodes (external nodes other than source are unreachable)
    targets = sorted(internal_nodes)

    # Filter out targets that have already been cached
    missing_targets = []
    for target in targets:
        path_cache = load_path_cache(fingerprint, source_node_idx, target, max_hops)
        hop_cache = load_hop_counts_cache(fingerprint, source_node_idx, target, max_hops)
        scatt_cache = load_scattering_coeffs_cache(fingerprint, source_node_idx, target, max_hops)
        if path_cache is None or hop_cache is None or scatt_cache is None:
            missing_targets.append(target)

    if not missing_targets:
        print("All target paths are already cached.")
        return
    
    print(f'Found {len(missing_targets)} targets to compute from source {source_node_idx} with max_hops {max_hops}')

    print(f"Computing paths from source {source_node_idx} to {len(missing_targets)} internal nodes")
    print('Generating paths may take a while, Please Wait .... ')

    # Determine worker count
    if n_workers is None:
        try:
            cpu_count = os.cpu_count() or 1
            n_workers = max(1, cpu_count - 2)
        except Exception:
            n_workers = 1

    # Ensure cache dir exists
    _ensure_fingerprint_dir(fingerprint)

    # Prepare arg tuples for worker function
    args_list = [
        (target, dict(neighbors), dict(link_lengths), external_nodes, dict(network_nodes),
         source_node_idx, fingerprint, max_hops, k0)
        for target in targets
    ]

    if use_processes and n_workers > 1:
        Executor = ProcessPoolExecutor
    else:
        # Fallback to threaded execution
        from concurrent.futures import ThreadPoolExecutor
        Executor = ThreadPoolExecutor

    with Executor(max_workers=n_workers) as ex:
        # Process results as they complete
        for target, count in ex.map(_compute_target_and_save_star, args_list):
            if count > 0:
                print(f"Computed and cached path lengths, hop counts, and scattering coeffs from source {source_node_idx} -> target {target}: {count} paths")
            else:
                print(f"No valid paths found from source {source_node_idx} -> target {target}")


# Backward compatibility alias
compute_and_cache_all_path_lengths = compute_cache_all_path_info


#________________________Detailed Path Enumerations_______________________________________________________
def find_paths_to_target_distance(
    network: Any,
    source_node_idx: int,
    target_link: Tuple[int, int],
    target_distance: float,
    max_hops: int,
    tolerance: float = 1e-6,
    max_bounces: int = 3
) -> List[Tuple[List, float]]:
    """Find paths to a specific total geometric distance from source that involve reflection on a target link.
    
    This function finds all valid paths from a source node back to itself that cross the target link
    at least once, such that the total path length equals the target distance. Reflections can occur
    at any point where the path crosses the target link.
    
    Args:
        network: Network object containing nodes and links
        source_node_idx: Index of the source node
        target_link: Tuple of (node_a, node_b) representing the target link
        target_distance: Target total geometric distance from source node
        max_hops: Maximum number of hops to explore for each path
        tolerance: Tolerance for distance matching (default: 1e-6)
        max_bounces: Maximum number of bounces/reflections with the fault allowed (default: 3)
        
    Returns:
        List of tuples (detailed_path, position_from_min_node), where:
        - detailed_path: list containing node indices and markers ('R', 'T')
        - position_from_min_node: distance along the link from the smaller node index
    """
    # Get link information and ensure canonical ordering
    node_a, node_b = min(target_link), max(target_link)
    target_link_length = network.get_link_by_node_indices((node_a,node_b)).length
    
    # Build lightweight graph representation
    neighbors: Dict[int, List[int]] = defaultdict(list)
    link_lengths: Dict[Tuple[int, int], float] = {}
    external_nodes = {node.index for node in network.external_nodes}
    
    for link in network.links:
        (a, b) = link.sorted_connected_nodes
        L = link.length
        
        k = (a,b)
        link_lengths[k] = L
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    
    # Ensure deterministic ordering
    for n in list(neighbors.keys()):
        neighbors[n] = sorted(set(neighbors[n]))
    
    results = []
    target_link_key = (min(target_link), max(target_link))
    
    # Find all paths from source back to source that cross the target link
    all_paths = _find_paths_with_target_link_crossings(
        neighbors, link_lengths, external_nodes, source_node_idx, 
        source_node_idx, max_hops, target_link_key
    )
    
    # Process each path that crosses the target link
    for path, _ , crossings in all_paths:
        if crossings == 0:  # Skip paths that don't cross the target link
            continue
            
        # For each possible reflection configuration
        for reflection_node in [node_a, node_b]:
            other_node = node_b if reflection_node == node_a else node_a

            # We consider each occurrence index of reflection_node in the base path as a potential anchor
            refl_indices: List[int] = [i for i, n in enumerate(path) if n == reflection_node]
            if not refl_indices:
                continue

            for refl_idx in refl_indices:
                # For each occurrence (refl_idx) we split the path at that index; the builder uses the split to insert 'R'/'T' markers.
                anchored_path = path

                # Try both first-action modes: reflection-first and transmit-first
                for first_action in ('R', 'T'):
                    # Try different numbers of bounces (>=1). Do not tie to base hops; rely on solver to reject.
                    for n_bounces in range(1, max_bounces + 1):
                        detailed_path = _build_detailed_path_with_reflections(
                            anchored_path,
                            reflection_node,
                            other_node,
                            n_bounces,
                            target_link_key,
                            first_action=first_action,
                            reflection_anchor_index=refl_idx,
                        )
                        if detailed_path is None:
                            continue

                        # Solve for segment length 'd' so that total path length equals target_distance
                        solved = _solve_segment_length_for_detailed_path(
                            detailed_path,
                            reflection_node,
                            other_node,
                            link_lengths,
                            target_link_key,
                            target_link_length,
                            target_distance,
                            tolerance,
                        )
                        if solved is None:
                            continue
                        d = solved

                        # Map to position from smaller node index
                        if reflection_node == node_a:
                            position_from_min_node = d
                        else:
                            position_from_min_node = target_link_length - d

                        results.append((detailed_path, position_from_min_node))
    
    return results


def _find_paths_with_target_link_crossings(
    neighbors: Dict[int, List[int]],
    link_lengths: Dict[Tuple[int, int], float],
    external_nodes: Set[int],
    start: int,
    end: int,
    max_hops: int,
    target_link_key: Tuple[int, int]
) -> List[Tuple[List[int], float, int]]:
    """Find all paths from start to end, tracking target link crossings."""
    paths = []
    queue = deque([(start, [start], 0.0, 0)])  # (current, path, length, crossings)
    
    while queue:
        current, path, geom_len, crossings = queue.popleft()
        
        if len(path) - 1 > max_hops:
            continue            
        if current == end and len(path) > 1:
            paths.append((path[:], geom_len, crossings))
            continue
            
        for neighbor in neighbors.get(current, []):
            if len(path) - 1 + 1 > max_hops:
                continue
                
            # External node constraints
            if neighbor in external_nodes:
                if neighbor != end and neighbor != start:
                    continue
                # Do not traverse through other external nodes; start/end are allowed entry points
                if neighbor != end and len(path) > 1:
                    continue
                
            # Calculate link key and length
            link_key = (min(current, neighbor), max(current, neighbor))
            link_len = link_lengths.get(link_key, 0.0)
            
            # Check if this is a target link crossing
            new_crossings = crossings
            if link_key == target_link_key:
                new_crossings += 1
                
            new_path = path + [neighbor]
            new_len = geom_len + link_len
            
            queue.append((neighbor, new_path, new_len, new_crossings))
    
    return paths



def _build_detailed_path_with_reflections(
    path: List[int], 
    reflection_node: int, 
    other_node: int, 
    n_bounces: int, 
    target_link_key: Tuple[int, int],
    first_action: str = 'R',
    reflection_anchor_index: Optional[int] = None,
) -> Optional[List]:
    """Build detailed path with R and T markers for target-link interactions.

    Aligns with the reference logic used in the provided notebook:
    - Anchor at the first valid occurrence of the reflection_node in the path such that
      the step into it is NOT coming from other_node (i.e., we didn't just cross the link).
    - Wrap the first reflection as [reflection_node, 'R', reflection_node] so there's
      always a node between markers, avoiding consecutive markers.
    - Support transmit-first by inserting a transmit sequence before the first reflection
      without creating adjacent markers.
    """
    target_link_set = set(target_link_key)

    # Find the first valid reflection occurrence (not immediately coming from other_node)
    refl_idx = None
    if reflection_anchor_index is not None:
        # Honor the provided occurrence if valid
        i = reflection_anchor_index
        if 0 <= i < len(path) and path[i] == reflection_node:
            refl_idx = i
    if refl_idx is None:
        for i, node in enumerate(path):
            if node == reflection_node:
                refl_idx = i
                break
    if refl_idx is None:
        return None

    # Split the path at the reflection point
    path_to_reflection = path[:refl_idx + 1]    # includes reflection_node
    path_from_reflection = path[refl_idx + 1:]  # excludes reflection_node

    detailed: List[Any] = []

    # Add path up to the reflection node, inserting 'T' before any target-link crossing
    if path_to_reflection:
        detailed.append(path_to_reflection[0])
        for i in range(len(path_to_reflection) - 1):
            curr = path_to_reflection[i]
            nxt = path_to_reflection[i + 1]
            if {curr, nxt} == target_link_set:
                detailed.append('T')
            detailed.append(nxt)

    # Insert first action
    if first_action == 'R':
        # reflection-first: ... rn, 'R', rn
        detailed.append(reflection_node)
        detailed.append('R')
        detailed.append(reflection_node)
        # subsequent extra bounces (beyond the first reflection)
        for _ in range(1, n_bounces):
            detailed.append('T')
            detailed.append(other_node)
            detailed.append('T')
            detailed.append(reflection_node)
            detailed.append('R')
            detailed.append(reflection_node)
    else:
        # transmit-first: ensure we are at the reflection side before crossing
        detailed.append(reflection_node)
        # cross to other side, come back, then reflect once
        # Sequence ensures no adjacent markers and valid link steps
        detailed.append('T')
        detailed.append(other_node)
        detailed.append('T')
        detailed.append(reflection_node)
        detailed.append('R')
        detailed.append(reflection_node)
        # additional bounces after the first reflection
        for _ in range(1, n_bounces):
            detailed.append('T')
            detailed.append(other_node)
            detailed.append('T')
            detailed.append(reflection_node)
            detailed.append('R')
            detailed.append(reflection_node)

    # Append the remainder of the path, adding 'T' marker before any subsequent target-link crossings
    for i, node in enumerate(path_from_reflection):
        if i == 0:
            # Edge from reflection_node -> node
            if {reflection_node, node} == target_link_set:
                detailed.append('T')
        else:
            prev = path_from_reflection[i - 1]
            if {prev, node} == target_link_set:
                detailed.append('T')
        detailed.append(node)

    # Sanitize to avoid adjacent duplicate markers or duplicate consecutive nodes
    cleaned: List[Any] = []
    for item in detailed:
        if cleaned:
            last = cleaned[-1]
            # Remove consecutive 'T' markers
            if isinstance(item, str) and item == 'T' and isinstance(last, str) and last == 'T':
                continue
            # Remove accidental duplicate consecutive nodes
            if isinstance(item, int) and isinstance(last, int) and item == last:
                continue
        cleaned.append(item)

    return cleaned


def _solve_segment_length_for_detailed_path(
    detailed_path: List[Any],
    reflection_node: int,
    other_node: int,
    link_lengths: Dict[Tuple[int, int], float],
    target_link_key: Tuple[int, int],
    target_link_length: float,
    target_distance: float,
    tolerance: float,
) -> Optional[float]:
    """Solve for segment length d such that total length along detailed_path equals target_distance.

    Treat 'R' and 'T' markers as a single perturbation node inserted on the target link between
    reflection_node and other_node at distance d from reflection_node (and L-d from other_node).
    """
    target_link_set = set(target_link_key)
    base_non_target = 0.0
    c_ref = 0  # count edges between reflection_node and perturbation
    c_other = 0  # count edges between other_node and perturbation

    # Iterate over consecutive elements, counting marker edges and summing non-target edges
    def is_marker(x: Any) -> bool:
        return isinstance(x, str) and x in ('R', 'T')

    for i in range(1, len(detailed_path)):
        a = detailed_path[i - 1]
        b = detailed_path[i]

        if not is_marker(a) and not is_marker(b):
            # Numeric-numeric edge
            if {a, b} == target_link_set:
                # Should be rare (we try to insert 'T'), but handle gracefully
                base_non_target += target_link_length
            else:
                lk = (min(int(a), int(b)), max(int(a), int(b)))
                base_non_target += link_lengths.get(lk, 0.0)
        elif is_marker(a) and not is_marker(b):
            # marker -> node
            if b == reflection_node:
                c_ref += 1
            elif b == other_node:
                c_other += 1
            else:
                # Should not happen; ignore
                pass
        elif not is_marker(a) and is_marker(b):
            # node -> marker
            if a == reflection_node:
                c_ref += 1
            elif a == other_node:
                c_other += 1
            else:
                # Should not happen; ignore
                pass
        else:
            # marker -> marker (should not occur after sanitization)
            pass

    # Total length function: L(d) = base_non_target + c_other * L + (c_ref - c_other) * d
    denom = (c_ref - c_other)
    if denom == 0:
        total_len = base_non_target + c_other * target_link_length
        if abs(total_len - target_distance) <= tolerance:
            # Any d in [0,L] works; choose midpoint for stability
            return max(0.0, min(target_link_length, target_link_length / 2.0))
        return None

    d = (target_distance - base_non_target - c_other * target_link_length) / float(denom)
    if d < -tolerance or d > target_link_length + tolerance:
        return None
    # Clamp small numeric drifts
    d = max(0.0, min(target_link_length, d))

    # Require strictly interior location to avoid degenerate endpoints
    eps = max(1e-12, tolerance)
    if d <= eps or d >= target_link_length - eps:
        return None

    # Optional: verify back-substitution within tolerance
    total_len = base_non_target + c_other * target_link_length + (c_ref - c_other) * d
    if abs(total_len - target_distance) > tolerance:
        return None
    return float(d)


def _find_paths_bfs(
    neighbors: Dict[int, List[int]],
    link_lengths: Dict[Tuple[int, int], float],
    external_nodes: Set[int],
    start: int,
    end: int,
    max_hops: int
) -> List[Tuple[List[int], float]]:
    """Find ALL paths from start to end using exhaustive search with geometric lengths.
    
    This version allows ALL possible paths within the hop limit, including:
    - Loops and cycles
    - Immediate node revisiting (e.g., A → B → A)
    - Complex path patterns
    
    The only constraints are:
    1. Maximum hop limit
    2. External node handling (external nodes can only be start/end points)
    """
    paths = []
    queue = deque([(start, [start], 0.0)])
    
    while queue:
        current, path, geom_len = queue.popleft()
        
        # Check if we've reached the maximum hop limit
        current_hops = len(path) - 1
        if current_hops > max_hops:
            continue
            
        # If we've reached the target and it's not a trivial path, record it
        if current == end and len(path) > 1:
            paths.append((path[:], geom_len))
            # Continue exploring from this point if we haven't reached max hops
            # This allows finding longer paths that also end at the target
            if current_hops >= max_hops:
                continue
            
        # Explore all neighbors if we haven't exceeded the hop limit
        if current_hops < max_hops:
            for neighbor in neighbors.get(current, []):
                # STRICT external node constraints
                if neighbor in external_nodes:
                    # Case 1: Neighbor is the target (end) - always allowed
                    if neighbor == end:
                        pass  # Allow this
                    # Case 2: Neighbor is not the target but is external
                    else:
                        # External nodes can only be start or end, never intermediate
                        # So we can never visit a non-target external node
                        continue
                
                # CRITICAL: If current node is external and we're continuing from it,
                # this is only valid if current is the start node
                if current in external_nodes and current != start:
                    # We should never be continuing from an external node that isn't the start
                    continue
                
                # Calculate link length
                lk = (min(current, neighbor), max(current, neighbor))
                seg_len = link_lengths.get(lk, 0.0)
                
                # Create new path - allow revisiting internal nodes only
                new_path = path + [neighbor]
                queue.append((neighbor, new_path, geom_len + seg_len))
    
    return paths


