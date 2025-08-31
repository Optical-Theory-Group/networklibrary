"""
discrete_graph_search_sensing.py

Drop-in replacement module that finds perturbation locations on links that explain measured OPL peaks.

Features:
- Fixed logic bugs from original (node-visit checking, validation).
- Disk cache of enumerated detailed paths per (network spec fingerprint, source_idx, max_hops).
- Extracts a picklable lightweight graph for multiprocessing.
- Parallel processing across links using ProcessPoolExecutor.
- Object-oriented API with methods to get best candidate, candidates list and scores for all links.
- Separate caching system for detailed paths (different from path lengths/hops cache)
"""

import os
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import deque, defaultdict
import math

# Import path generation functions from network_path_search
from complex_network.networks.network_path_search import (
    GraphData, 
    extract_graph_data, 
    prepare_discrete_path_cache,
    find_all_paths_bfs,
    load_detailed_path_cache,
    save_detailed_path_cache,
    _spec_fingerprint,
    _link_key
)

# Type aliases
NodeIdx = int
Link = Tuple[int, int]
Path = List[int]


# ---------------------
# Core validation & path building
# ---------------------

def validate_path_for_reflection(path: Path, reflection_node: int, other_node: int, source_idx: int) -> bool:
    """
    Validate that:
    1. The path doesn't cross the perturbation link before reaching the reflection point
    2. The path doesn't revisit the source node in the middle (only allowed at start and end)
    """
    # Check for source revisits (except at start and end)
    for i in range(1, len(path) - 1):
        if path[i] == source_idx:
            return False

    # Find first occurrence of reflection node
    refl_idx = None
    for i, node in enumerate(path):
        if node == reflection_node:
            refl_idx = i
            break

    if refl_idx is None:
        return False

    # Check if we cross the perturbation link before the reflection
    for i in range(refl_idx):
        if i < len(path) - 1:
            curr, next_node = path[i], path[i+1]
            # If we traverse the perturbation link before reflection, invalid
            if _link_key(curr, next_node) == _link_key(reflection_node, other_node):
                return False

    return True


def build_detailed_path(
    path: List[int], reflection_node: int, other_node: int, n_bounces: int
) -> List:
    """
    Construct a detailed path with R markers for reflections and T markers for transmissions.
    """
    # Find the reflection point
    reflection_idx = None
    for i, node in enumerate(path):
        if node == reflection_node:
            # Make sure we didn't come from the other node (crossed the link)
            if i == 0 or path[i-1] != other_node:
                reflection_idx = i
                break

    if reflection_idx is None:
        return None

    # Split the path
    path_to_reflection = path[:reflection_idx + 1]  # Include reflection node
    path_from_reflection = path[reflection_idx + 1:]  # Exclude reflection node

    # Build detailed path
    detailed = []

    # Add path to reflection (no T markers yet)
    detailed.extend(path_to_reflection[:-1])  # Don't duplicate reflection node

    # Add reflection sequence
    detailed.append(reflection_node)
    detailed.append('R')
    detailed.append(reflection_node)

    # Add additional bounces if needed
    for bounce in range(1, n_bounces):
        detailed.append('T')
        detailed.append(other_node)
        detailed.append('T')
        detailed.append(reflection_node)
        detailed.append('R')
        detailed.append(reflection_node)

    # Add return path with T markers where appropriate
    for i, node in enumerate(path_from_reflection):
        if i == 0:
            if {reflection_node, node} == {reflection_node, other_node}:
                detailed.append('T')
        else:
            prev_node = path_from_reflection[i-1]
            if {prev_node, node} == {reflection_node, other_node}:
                detailed.append('T')
        detailed.append(node)

    return detailed


# ---------------------
# Single-link solver (pure function for parallelization)
# ---------------------

def _solve_for_single_link(
    graph: GraphData,
    source_idx: int,
    link_ab: Tuple[int,int],
    path_cache: Dict[Tuple[int,int,int], List[Tuple[Path,float]]],
    first_peak: float,
    validation_peaks: List[float],
    max_hops: int,
    n_index: float,
    tolerance: float,
    max_bounces: int
) -> Dict:
    a,b = link_ab
    L = graph.link_lengths.get(_link_key(a,b))
    if L is None:
        return {'best_location': None, 'explained_count': 0, 'explained_details': {}, 'unexplained_opls': validation_peaks}

    if source_idx in (a, b):
        return {'best_location': None, 'explained_count': 0, 'explained_details': {}, 'unexplained_opls': validation_peaks}

    candidates = []
    seen_positions = set()

    # For both reflection directions
    for refl, other in [(a, b), (b, a)]:
        # Get paths from cache (use half hops for to/from)
        to_paths = path_cache.get((source_idx, refl, max_hops//2), [])
        from_paths = path_cache.get((refl, source_idx, max_hops//2), [])

        for pin, pin_len in to_paths:
            if len(pin) == 0 or pin[-1] != refl:
                continue
            for pout, pout_len in from_paths:
                if len(pout) == 0 or pout[0] != refl:
                    continue

                # Combine paths
                path = pin[:-1] + pout

                # Validate the complete path
                if not validate_path_for_reflection(path, refl, other, source_idx):
                    continue

                # Check that path doesn't have unnecessary loops
                node_visits = {}
                for node in path:
                    if node != refl:  # Reflection node can be visited twice
                        node_visits[node] = node_visits.get(node, 0) + 1
                        if node_visits[node] > 2:
                            break
                else:
                    # All visits are valid
                    base_opl = (pin_len + pout_len) * n_index
                    if base_opl > first_peak + tolerance:
                        continue

                    rem = first_peak - base_opl

                    for nb in range(1, min(max_bounces+1, max_hops-(len(path)-1)+1)):
                        if len(path)-1+nb > max_hops:
                            break
                        seg = rem / (nb * 2 * n_index)
                        if 0 < seg < L - 1e-12:
                            pos = seg if refl < other else L - seg
                            key = (tuple(path), refl, nb, round(pos,12))
                            if key in seen_positions:
                                continue
                            seen_positions.add(key)
                            detailed = build_detailed_path(path, refl, other, nb)
                            if detailed is not None:
                                candidates.append({
                                    'detailed_path': detailed,
                                    'location': pos,
                                    'location_from_node': (refl, seg),
                                    'reflection_node': refl,
                                    'n_bounces': nb,
                                    'total_hops': len(path)-1+nb,
                                    'base_path': path,
                                    'base_opl': base_opl
                                })

    if not candidates:
        return {'best_location': None, 'explained_count': 0, 'explained_details': {}, 'unexplained_opls': validation_peaks}

    # Minimal OPL validation
    all_measured_opls = [first_peak] + validation_peaks
    valid_candidates = []

    for cand in candidates:
        loc = cand['location']
        refl = cand['reflection_node']
        seg = cand['location_from_node'][1]

        # Find the minimal OPL from this location (search small hop)
        min_opl = float('inf')
        min_path = None
        to_paths_min = path_cache.get((source_idx, refl, 6), [])
        from_paths_min = path_cache.get((refl, source_idx, 6), [])

        for pin, pin_len in to_paths_min:
            if len(pin) == 0 or pin[-1] != refl:
                continue
            for pout, pout_len in from_paths_min:
                if len(pout) == 0 or pout[0] != refl:
                    continue
                test_path = pin[:-1] + pout
                if not validate_path_for_reflection(test_path, refl, (a if refl == b else b), source_idx):
                    continue
                total_opl = (pin_len + pout_len) * n_index + 2 * seg * n_index
                if total_opl < min_opl:
                    min_opl = total_opl
                    min_path = test_path

        if min_opl < first_peak - tolerance:
            peak_found = any(abs(measured - min_opl) < tolerance for measured in all_measured_opls)
            if not peak_found:
                # interference check: need >=2 paths at same min_opl
                interference_count = 0
                for pin, pin_len in to_paths_min:
                    if len(pin) == 0 or pin[-1] != refl:
                        continue
                    for pout, pout_len in from_paths_min:
                        if len(pout) == 0 or pout[0] != refl:
                            continue
                        test_path = pin[:-1] + pout
                        if not validate_path_for_reflection(test_path, refl, (a if refl == b else b), source_idx):
                            continue
                        test_opl = (pin_len + pout_len) * n_index + 2 * seg * n_index
                        if abs(test_opl - min_opl) < tolerance:
                            interference_count += 1
                            if interference_count >= 2:
                                break
                    if interference_count >= 2:
                        break
                if interference_count < 2:
                    continue  # reject candidate
        valid_candidates.append(cand)

    if not valid_candidates:
        return {
            'best_location': None,
            'explained_count': 0,
            'explained_details': {},
            'unexplained_opls': validation_peaks,
            'rejection_reason': 'All candidates rejected due to missing minimal OPL peaks'
        }

    # Score candidates (group by location)
    best = {
        'best_location': None,
        'explained_count': 0,
        'explained_details': {},
        'unexplained_opls': []
    }

    unique = {}
    for c in valid_candidates:
        loc_key = f"{c['location']:.12f}"
        if loc_key not in unique or c['base_opl'] < unique[loc_key]['base_opl']:
            unique[loc_key] = c

    for loc_key, cand in unique.items():
        loc = cand['location']
        details = {first_peak: [cand]}
        count = 1
        unexplained = []
        for opl in validation_peaks:
            matches_for_opl = []
            refl = cand['reflection_node']
            seg = cand['location_from_node'][1]
            to_paths = path_cache.get((source_idx, refl, max_hops//2), [])
            from_paths = path_cache.get((refl, source_idx, max_hops//2), [])

            for pin, pin_len in to_paths:
                if len(pin) == 0 or pin[-1] != refl:
                    continue
                for pout, pout_len in from_paths:
                    if len(pout) == 0 or pout[0] != refl:
                        continue
                    path = pin[:-1] + pout
                    if not validate_path_for_reflection(path, refl, (a if refl == b else b), source_idx):
                        continue
                    base_opl = (pin_len + pout_len) * n_index
                    for nb in range(1, min(max_bounces+1, max_hops-(len(path)-1)+1)):
                        hops = len(path)-1+nb
                        if hops > max_hops:
                            break
                        total_opl = base_opl + nb*2*seg*n_index
                        if abs(total_opl - opl) < tolerance:
                            detailed = build_detailed_path(path, refl, (a if refl == b else b), nb)
                            if detailed is not None:
                                matches_for_opl.append({
                                    'detailed_path': detailed,
                                    'n_bounces': nb,
                                    'calculated_opl': total_opl,
                                    'reflection_node': refl,
                                    'location_from_node': (refl, seg),
                                    'total_hops': hops
                                })
                                break
            if matches_for_opl:
                count += 1
                details[opl] = matches_for_opl
            else:
                unexplained.append(opl)
        if count > best['explained_count']:
            best = {
                'best_location': loc,
                'explained_count': count,
                'explained_details': details,
                'unexplained_opls': unexplained
            }

    return best


class DiscreteGraphSearchSensing:
    """
    Object oriented entry point for discrete graph search sensing.
    Detects perturbation locations on network links based on measured OPL peaks.
    """

    def __init__(self, net: Any, source_idx: int, spec: Optional[Any] = None, cache_salt: Optional[str] = None):
        self.net = net
        self.source_idx = int(source_idx)
        self.spec = spec
        self.cache_salt = cache_salt
        # Build graph data
        self.graph = extract_graph_data(net)
        # fingerprint for caching
        if spec is not None:
            self.fingerprint = _spec_fingerprint(spec, salt=cache_salt)
        else:
            summary = {
                'n_nodes': len(self.graph.neighbors),
                'n_links': len(self.graph.internal_links),
            }
            self.fingerprint = f"n_nodes={summary['n_nodes']}-n_links={summary['n_links']}"

        # path cache stored in memory: keys are (start,end,hop_limit)
        self._path_cache: Dict[Tuple[int,int,int], List[Tuple[Path,float]]] = {}

    def prepare_detailed_path_cache(self, max_hops: int, force_rebuild: bool=False) -> Dict[Tuple[int,int,int], List[Tuple[Path,float]]]:
        """
        Build or load detailed path cache for given max_hops.
        This uses the separate discrete path cache system in network_path_search.py.
        """
        self._path_cache = prepare_discrete_path_cache(
            self.graph, 
            self.fingerprint, 
            self.source_idx, 
            max_hops, 
            force_rebuild
        )
        return self._path_cache

    def find_best_for_link(
        self,
        link: Tuple[int,int],
        first_peak: float,
        validation_peaks: List[float],
        max_hops: int = 15,
        n_index: float = 1.5,
        tolerance: float = 1e-6,
        max_bounces: int = 2
    ) -> Dict:
        """
        Find best perturbation location for a single link.
        """
        self.prepare_detailed_path_cache(max_hops)
        return _solve_for_single_link(
            self.graph, self.source_idx, _link_key(*link), self._path_cache,
            float(first_peak), list(validation_peaks),
            max_hops, float(n_index), float(tolerance), int(max_bounces)
        )

    def find_best_for_all_links(
        self,
        links: List[Tuple[int,int]],
        measured_opl_provider,
        max_hops: int = 15,
        n_index: float = 1.5,
        tolerance: float = 1e-6,
        max_bounces: int = 2,
        parallel: bool = True,
        workers: Optional[int] = None,
        debug: bool = False
    ) -> Dict[Tuple[int,int], Dict]:
        """
        Compute best perturbation location for each link.
        """
        # Prepare path cache once
        self.prepare_detailed_path_cache(max_hops)

        # Canonicalize links (min,max)
        canonical_links = [_link_key(*l) for l in links]

        # Normalize provider
        if callable(measured_opl_provider):
            def provider(link): return np.array(measured_opl_provider(link), dtype=float)
        else:
            fixed_arr = np.array(measured_opl_provider, dtype=float)
            def provider(_link): return fixed_arr

        # Build work items
        work_items = []
        for link in canonical_links:
            arr = provider(link)
            if arr.size == 0:
                work_items.append((link, None))
                continue
            first = float(arr[0])
            validation = list(arr[1:])
            args = (self.graph, self.source_idx, link, self._path_cache, first, validation, max_hops, n_index, tolerance, max_bounces)
            work_items.append((link, args))

        results_map: Dict[Tuple[int,int], Dict] = {}

        if parallel:
            if workers is None:
                workers = max(1, (os.cpu_count() or 1) - 1)
            with ProcessPoolExecutor(max_workers=workers) as exe:
                future_to_link = {}
                for link, args in work_items:
                    if args is None:
                        results_map[link] = {'best_location': None, 'explained_count': 0, 'explained_details': {}, 'unexplained_opls': []}
                        continue
                    future = exe.submit(_solve_for_single_link, *args)
                    future_to_link[future] = link
                for fut in as_completed(future_to_link):
                    link = future_to_link[fut]
                    try:
                        res = fut.result()
                    except Exception as e:
                        res = {'best_location': None, 'explained_count': 0, 'explained_details': {}, 'unexplained_opls': [], 'error': str(e)}
                    results_map[link] = res
        else:
            for link, args in work_items:
                if args is None:
                    results_map[link] = {'best_location': None, 'explained_count': 0, 'explained_details': {}, 'unexplained_opls': []}
                    continue
                try:
                    res = _solve_for_single_link(*args)
                except Exception as e:
                    res = {'best_location': None, 'explained_count': 0, 'explained_details': {}, 'unexplained_opls': [], 'error': str(e)}
                results_map[link] = res

        if debug:
            for link in canonical_links:
                res = results_map.get(link, {})
                loc = res.get('best_location')
                score = res.get('explained_count', 0)
                reason = res.get('rejection_reason') or res.get('error')
                print(f"Link {link}: best_location={loc}, score={score}, note={reason}")

        return results_map

    def get_candidates_for_all_links(
        self,
        links: List[Tuple[int,int]],
        measured_opl_provider,
        max_hops: int = 15,
        n_index: float = 1.5,
        tolerance: float = 1e-6,
        max_bounces: int = 2,
        parallel: bool = True,
        workers: Optional[int] = None
    ) -> Dict[Tuple[int,int], List[Dict]]:
        """
        Return the raw candidate lists (validated candidates before grouping) for each link.
        """
        results = self.find_best_for_all_links(
            links,
            measured_opl_provider,
            max_hops=max_hops,
            n_index=n_index,
            tolerance=tolerance,
            max_bounces=max_bounces,
            parallel=parallel,
            workers=workers
        )
        out = {}
        for link, res in results.items():
            if not res or 'explained_details' not in res or not res['explained_details']:
                out[link] = []
                continue
            first_peak = next(iter(res['explained_details'].keys()))
            out[link] = res['explained_details'].get(first_peak, [])
        return out

    def get_scores_for_all_links(
        self,
        links: List[Tuple[int,int]],
        measured_opl_provider,
        max_hops: int = 15,
        n_index: float = 1.5,
        tolerance: float = 1e-6,
        max_bounces: int = 2,
        parallel: bool = True,
        workers: Optional[int] = None
    ) -> Dict[Tuple[int,int], int]:
        """
        Return the explained_count score for each link.
        """
        results = self.find_best_for_all_links(
            links,
            measured_opl_provider,
            max_hops=max_hops,
            n_index=n_index,
            tolerance=tolerance,
            max_bounces=max_bounces,
            parallel=parallel,
            workers=workers
        )
        return {link: (res.get('explained_count', 0) if res else 0) for link, res in results.items()}


# ---------------------
# Convenience function for single OPL target analysis
# ---------------------

def get_perturbation_candidates_for_opl(
    net: Any,
    source_idx: int,
    target_link_nodes: Tuple[int, int],
    target_opl: float,
    max_hops: int = 15,
    n_index: float = 1.5,
    tolerance: float = 1e-6,
    max_bounces: int = 2,
    spec: Optional[Any] = None
) -> List[Dict]:
    """
    Find perturbation candidates for a specific OPL on a target link.
    
    Args:
        net: Network object
        source_idx: Source node index  
        target_link_nodes: Tuple of (node_a, node_b) for the target link
        target_opl: Target optical path length to match
        max_hops: Maximum hops to search
        n_index: Refractive index
        tolerance: Tolerance for OPL matching
        max_bounces: Maximum number of bounces
        spec: Optional network spec for caching
        
    Returns:
        List of candidate dictionaries with detailed path information
    """
    sensor = DiscreteGraphSearchSensing(net, source_idx, spec=spec)
    
    # Use the target OPL as both first peak and validation
    result = sensor.find_best_for_link(
        target_link_nodes,
        first_peak=target_opl,
        validation_peaks=[],  # Only looking for the single OPL
        max_hops=max_hops,
        n_index=n_index,
        tolerance=tolerance,
        max_bounces=max_bounces
    )
    
    # Extract candidates from the result
    if result.get('explained_details'):
        first_peak_candidates = next(iter(result['explained_details'].values()))
        return first_peak_candidates
    else:
        return []


#_______________Example Testing ____________________________________

if __name__ == "__main__":
    import time
    from complex_network.networks.network_factory import generate_network
    from complex_network.networks.network_spec import NetworkSpec

    start = time.time()
    
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

    source_node = 10
    # ensure canonical tuple form for links
    links = [tuple(link.sorted_connected_nodes) for link in net.internal_links]

    measured_opls = np.array([275.80551611,  311.90623812,  332.00664013,  348.70697414,  381.30762615,
        389.30778616,  399.30798616,  406.50813016,  433.20866417,  442.20884418,
        453.60907218,  463.00926019,  478.10956219,  486.90973819,  511.6102322]) * 1e-6

    sensor = DiscreteGraphSearchSensing(net, source_node, spec=spec)

    def provider(_link):
        return measured_opls

    # Start with parallel processing
    results_all = sensor.find_best_for_all_links(
        links, provider,
        max_hops=12, n_index=1.5, tolerance=1e-6, max_bounces=2,
        parallel=True, debug=True
    )

    best_location_array = [results_all.get(tuple(l), {}).get('best_location') for l in links]
    print("Best locations per link:", best_location_array)
