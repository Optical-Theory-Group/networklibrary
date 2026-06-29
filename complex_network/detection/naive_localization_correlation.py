"""
Naive fault localization algorithm for complex networks.
The algorithm generates fault position on each link based on 
a set of peaks that fall above a certain threshold from all the sources
if no peaks are found, the threshold is lowered until at least one peak is found.
The peaks are generated from the OLCR measurement and then uses
the other peaks to score the candidates that were generated.

The peaks are scored based on how close the predicted position
is to the candidate position using an exponential distance-based scoring.

We find the candidates
from multiples sources and see which candidates are common or
within one coherence length and then score those candidates
based on all the peaks from all sources.

"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import matplotlib.pyplot as plt

# Import path generation functions from network_path_search
from complex_network.networks.network_path_search import (
    extract_graph_data,
    prepare_detailed_path_cache,
    _spec_fingerprint)
from complex_network.networks.network import Network

# Type aliases
NodeIdx = int
Link = Tuple[int, int]
Path = List[int]


@dataclass
class Candidate:
    """Represents a fault position candidate on a link."""
    link: Tuple[int, int]      # (node_A, node_B)
    position: float            # fractional distance from node A (0 < x < 1)
    type: Tuple[int,int]       # "AA", "BB", "AB", or "BA"
    generating_peak: float     # peak used to generate this candidate
    path_in: Path              # inbound path to link
    path_out: Path             # outbound path from link
    score: float = 0.0         # Score of the candidate or how well it explains peaks
    num_explained: int = 0     # number of peaks this candidate explains


@dataclass
class PathBin:
    """Paths within half a coherence length that were binned together."""
    source: int                     # index of source node
    target: int                     # index of target node
    representative_length: float    # representative optical path length of the bin
    paths: List[Path]               # list of paths in this bin

@dataclass
class AllPathBins:
    """All the path bins that go to any particular node.
        collection of PathBin objects."""
    source: int                     # index of source node
    target: int                     # index of target nodes
    path_bins: List[PathBin]        # list of PathBin objects


class FaultLocalizer:
    """
    Fault localizer using naive path matching, candidate generation and scoring approach.
    """
    def __init__(self, 
                 network: Network,
                 source_indices: List[int],
                 max_hops: int,
                 n_index: float = 1.5,
                 coherence_length: float = 3e-9,
                 interference_threshold: int = 2,
                 use_multiprocessing: bool = False,
                 opls: Optional[np.ndarray] = None,
                 aggregation_method: str = 'sum'
                ):
        """
        Initialize the fault localizer.
        
        Args:
            network: Network object
            source_indices: List of source node indices we are using
            n_index: Refractive index of medium
            coherence_length: Coherence length for peak matching tolerance
            interference_threshold: Minimum number of paths to consider interference effects
            use_multiprocessing: Whether to use multiprocessing for internal operations
            opls: Optical path length grid for OLCR measurements (required for correlation scoring)
            aggregation_method: How to aggregate scores across sources ('sum' or 'product')
        """
        self.network = network
        self.source_indices = [int(idx) for idx in source_indices]
        self.n_index = float(n_index)
        self.coherence_length = float(coherence_length)
        self.interference_threshold = int(interference_threshold)
        self.max_hops = int(max_hops)
        self.use_multiprocessing = bool(use_multiprocessing)
        self.opls = opls
        self.aggregation_method = aggregation_method.lower()
        
        if self.aggregation_method not in ['sum', 'product']:
            raise ValueError(f"aggregation_method must be 'sum' or 'product', got {aggregation_method}")

        # Caches for optimization
        self._path_validation_cache = {}
        self._reduced_variants_cache = {}
        self._simple_path_cache = {}

        # Extract lightweight graph data
        self.graph = extract_graph_data(network)
        
        # Generate fingerprint for caching
        if getattr(network, 'spec', None) is not None:
            self.fingerprint = _spec_fingerprint(network.spec)
        else:
            raise ValueError("Network was not created from a valid spec | cannot generate cache fingerprint")
        
        # Precompute paths for all source nodes
        self.binned_paths_per_source = self._precompute_paths_all_sources()
        
        
    def _precompute_paths_all_sources(self) -> Dict[int, Dict[int, AllPathBins]]:
        """Pre-compute all paths from all source nodes to each node using caching infrastructure.
           The Data structure is:
           { source_idx : { target_idx : AllPathBins } }

           Returns:
            Dictionary mapping source indices to target indices to AllPathBins."""
        binned_paths_per_source = {}

        for source_idx in self.source_indices:
            detailed_cache = prepare_detailed_path_cache(
                self.graph, self.fingerprint, source_idx, self.max_hops)

            target_to_bins = {}
            for key, paths in detailed_cache.items():
                source, target = key[0], key[1]
                if source == source_idx:
                    target_to_bins[target] = AllPathBins(
                        source=source,
                        target=target,
                        path_bins=self._bin_paths(source, target, paths, self.coherence_length)
                    )
            binned_paths_per_source[source_idx] = target_to_bins

        return binned_paths_per_source

    
    def _link_key(self, a: int, b: int) -> Tuple[int, int]:
        """Return sorted link key as (min(a, b), max(a, b))."""
        return (min(a, b), max(a, b))
    
    def _get_link_length(self, link: Tuple[int, int]) -> float:
        """Get physical length of a link."""
        return self.graph.link_lengths.get(self._link_key(*link), 0.0)
    
    def _binary_search_start(self, sorted_bins: List[PathBin], min_length: float) -> int:
        """Find the starting index in sorted bins where representative_length >= min_length.
        
        Args:
            sorted_bins: List of PathBin objects sorted by representative_length
            min_length: Minimum required length
            
        Returns:
            Starting index for iteration (0 if min_length is less than all bins)
        """
        if not sorted_bins or min_length <= sorted_bins[0].representative_length:
            return 0
        if min_length > sorted_bins[-1].representative_length:
            return len(sorted_bins)
        
        left, right = 0, len(sorted_bins) - 1
        result = 0
        
        while left <= right:
            mid = (left + right) // 2
            if sorted_bins[mid].representative_length >= min_length:
                result = mid
                right = mid - 1
            else:
                left = mid + 1
        
        return result

    def _bin_paths(self,
                   source: int,
                   target: int,
                   paths: List[Tuple[Path, float]],
                   coherence_length: float) -> List[PathBin]:
        """
        Bin paths to a target node based on their lengths.
        """
        paths = sorted(paths, key=lambda x: x[1])
        spatial_resolution = coherence_length / (2 * self.n_index)

        path_bins: List[PathBin] = []
        num_paths = len(paths)

        binned_paths: List[Path] = []
        lengths: List[float] = []
        bin_start: int = 0
        
        first_path, first_length = paths[0]
        binned_paths.append(first_path)
        lengths.append(first_length)

        for i in range(1, num_paths):
            path, length = paths[i]
            _, length_start = paths[bin_start]

            if abs(length - length_start) < spatial_resolution/2:
                binned_paths.append(path)
                lengths.append(length)
            else:
                if binned_paths:
                    path_bins.append(PathBin(
                        source=source,
                        target=target,
                        representative_length=(lengths[0]+lengths[-1])/2,
                        paths=binned_paths.copy()
                    ))
                    bin_start = i
                    binned_paths = [path]
                    lengths = [length]

        if binned_paths:
            path_bins.append(PathBin(
                source=source,
                target=target,
                representative_length=(lengths[0]+lengths[-1])/2,
                paths=binned_paths.copy()
            ))
        return path_bins


    def generate_candidates_for_link(self,
                                    source_idx: int,
                                    link: Tuple[int, int], 
                                    peak: float,
                                    measured_peaks: List[float]) -> List[Candidate]:
        """Generate fault position candidates for a given link and peak from a specific source."""
        node_a, node_b = link
        L_link = self._get_link_length(link)
        
        path_bins_to_a = self.binned_paths_per_source[source_idx][node_a]
        path_bins_to_b = self.binned_paths_per_source[source_idx][node_b]

        candidates = self._generate_class_candidates(link, peak, path_bins_to_a, path_bins_to_b, L_link, measured_peaks)

        return candidates

    def _generate_class_candidates(self,
                                link: Tuple[int, int],
                                peak: float,
                                path_bins_to_a: AllPathBins,
                                path_bins_to_b: AllPathBins,
                                L_link: float,
                                measured_peaks: List[float]) -> List[Candidate]:
        """Generate candidates for a specific path class."""
        candidates = []
        
        # Sort path bins by representative length for efficient searching
        sorted_bins_a = sorted(path_bins_to_a.path_bins, key=lambda pb: pb.representative_length)
        sorted_bins_b = sorted(path_bins_to_b.path_bins, key=lambda pb: pb.representative_length)

        # Case AA
        for i, path_bin in enumerate(sorted_bins_a):
            # Calculate the required range for path_bin_prime
            L_min_required = (peak - 2*self.n_index*L_link) / self.n_index - path_bin.representative_length
            L_max_required = peak / self.n_index - path_bin.representative_length
            
            # Find starting index using binary search
            start_idx = self._binary_search_start(sorted_bins_a, L_min_required)
            
            # Only iterate through relevant bins
            for path_bin_prime in sorted_bins_a[start_idx:]:
                if path_bin_prime.representative_length > L_max_required:
                    break
                    
                LAA = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                if peak - 2*self.n_index*L_link < LAA < peak:
                    x = (peak - LAA) / (2 * self.n_index * L_link)
                    candidate = Candidate(
                        link=link,
                        position=x,
                        type=(path_bin.target, path_bin_prime.target),
                        generating_peak=peak,
                        path_in=path_bin.paths,
                        path_out=path_bin_prime.paths[::-1]
                    )
                    if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                        if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                            candidates.append(candidate)
                    else:
                        candidates.append(candidate)

        # Case BB
        for i, path_bin in enumerate(sorted_bins_b):
            # Calculate the required range for path_bin_prime
            L_min_required = (peak - 2*self.n_index*L_link) / self.n_index - path_bin.representative_length
            L_max_required = peak / self.n_index - path_bin.representative_length
            
            # Find starting index using binary search
            start_idx = self._binary_search_start(sorted_bins_b, L_min_required)
            
            # Only iterate through relevant bins
            for path_bin_prime in sorted_bins_b[start_idx:]:
                if path_bin_prime.representative_length > L_max_required:
                    break
                    
                LBB = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                if peak - 2*self.n_index*L_link < LBB < peak:
                    x = 1 - (peak - LBB) / (2 * self.n_index * L_link)
                    candidate = Candidate(
                        link=link,
                        position=x,
                        type=(path_bin.target, path_bin_prime.target),
                        generating_peak=peak,
                        path_in=path_bin.paths,
                        path_out=path_bin_prime.paths[::-1]
                    )
                    if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                        if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                            candidates.append(candidate)
                    else:
                        candidates.append(candidate)

        # Case AB
        for i, path_bin in enumerate(sorted_bins_a):
            # Calculate the required range for path_bin_prime
            L_min_required = (peak - 3*self.n_index*L_link) / self.n_index - path_bin.representative_length
            L_max_required = (peak - self.n_index*L_link) / self.n_index - path_bin.representative_length
            
            # Find starting index using binary search
            start_idx = self._binary_search_start(sorted_bins_b, L_min_required)
            
            # Only iterate through relevant bins
            for path_bin_prime in sorted_bins_b[start_idx:]:
                if path_bin_prime.representative_length > L_max_required:
                    break
                    
                LAB = path_bin.representative_length * self.n_index + path_bin_prime.representative_length * self.n_index
                if peak - 3*self.n_index*L_link < LAB < peak - self.n_index*L_link:
                    x = (peak - LAB - self.n_index*L_link) / (2 * self.n_index * L_link)
                    candidate = Candidate(
                        link=link,
                        position=x,
                        type=(path_bin.target, path_bin_prime.target),
                        generating_peak=peak,
                        path_in=path_bin.paths,
                        path_out=path_bin_prime.paths[::-1]
                    )
                    if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                        if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                            candidates.append(candidate)
                    else:
                        candidates.append(candidate)

        # Case BA
        for i, path_bin in enumerate(sorted_bins_b):
            # Calculate the required range for path_bin_prime
            L_min_required = (peak - 3*self.n_index*L_link) / self.n_index - path_bin.representative_length
            L_max_required = (peak - self.n_index*L_link) / self.n_index - path_bin.representative_length
            
            # Find starting index using binary search
            start_idx = self._binary_search_start(sorted_bins_a, L_min_required)
            
            # Only iterate through relevant bins
            for path_bin_prime in sorted_bins_a[start_idx:]:
                if path_bin_prime.representative_length > L_max_required:
                    break
                    
                LBA = path_bin.representative_length * self.n_index + path_bin_prime.representative_length * self.n_index
                if peak - 3*self.n_index*L_link < LBA < peak - self.n_index*L_link:
                    x = (3*self.n_index*L_link + LBA - peak) / (2 * self.n_index * L_link)
                    candidate = Candidate(
                        link=link,
                        position=x,
                        type=(path_bin.target, path_bin_prime.target),
                        generating_peak=peak,
                        path_in=path_bin.paths,
                        path_out=path_bin_prime.paths[::-1]
                    )
                    if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                        if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                            candidates.append(candidate)
                    else:
                        candidates.append(candidate)

        return candidates
    
    def find_common_candidates(self,
                            source_peak_dict: Dict[int, Dict[str, List[float]]],
                            qualifying_peak_dict: Dict[int, Dict[str, List[float]]], 
                            links: Optional[List[Tuple[int, int]]] = None
                            ) -> Dict[Tuple[int, int], List[Candidate]]:
        """Find candidates that matched the most number of times across all links."""
        if links is None:
            links = [tuple(link.sorted_connected_nodes) for link in self.network.internal_links]

        sources = list(qualifying_peak_dict.keys())
        source_location_link_pairs = [(source, loc, link, measured_peaks) 
                                     for source in sources 
                                     for loc in qualifying_peak_dict[source][0] 
                                     for link in links 
                                     for measured_peaks in [source_peak_dict[source][0]]]

        # Only use multiprocessing if workload is large enough to reduce overhead
        # Otherwise serialization overhead dominates
        num_tasks = len(source_location_link_pairs)
        
        if num_tasks < 50 or not self.use_multiprocessing:  # Threshold for when multiprocessing is worth it
            # Direct execution for small workloads or when multiprocessing is disabled
            all_candidates_with_loc = {}
            for (source, loc, link, measured_peaks) in source_location_link_pairs:
                key = (source, link, loc)
                all_candidates_with_loc[key] = self.generate_candidates_for_link(source, link, loc, measured_peaks)
        else:
            # Multiprocessed approach for large workloads
            with ProcessPoolExecutor() as executor:
                futures = {(source, link, loc): executor.submit(self.generate_candidates_for_link, source, link, loc, measured_peaks)
                            for (source, loc, link, measured_peaks) in source_location_link_pairs}
                all_candidates_with_loc = {key: future.result() for key, future in futures.items()}

        all_candidates = defaultdict(list)
        for (source, link, _), candidates in all_candidates_with_loc.items():
            all_candidates[(source, link)].extend(candidates)

        # Check for common candidates across all sources for each link
        max_common_link_candidates = {}
        global_max_frequency = 0

        for link in links:
            all_candidates_for_link = {source: all_candidates[(source, link)] for source in sources}
            candidates_with_freq = self._get_candidates_with_frequency(all_candidates_for_link, sources)
            
            link_frequencies = list(candidates_with_freq.keys())
            link_max_freq = max(link_frequencies) if link_frequencies else 0

            if link_max_freq > global_max_frequency:
                global_max_frequency = link_max_freq

        # Filter to only keep candidates with max_frequency
        for link in links:
            all_candidates_for_link = {source: all_candidates[(source, link)] for source in sources}
            candidates_with_freq = self._get_candidates_with_frequency(all_candidates_for_link, sources)
            
            if global_max_frequency in candidates_with_freq.keys():
                max_common_link_candidates[link] = candidates_with_freq[global_max_frequency]
        
        return max_common_link_candidates

    def _get_candidates_with_frequency(self,
                                    all_candidates_for_link: Dict[int, List[Candidate]],
                                    sources: List[int]) -> Dict[int, List[Candidate]]:
        """Group candidates by frequency of appearance across sources."""
        if len(sources) < 1:
            raise ValueError("No sources provided")
        
        non_empty_sources = {source: candidates for source, candidates in 
                            all_candidates_for_link.items() if candidates}
        
        if not non_empty_sources:
            return {}
        
        candidate_source_list = [(candidate, source) for source, candidates in
                                non_empty_sources.items() for candidate in candidates]
        candidates = np.array([candidate for candidate, _ in candidate_source_list])
        candidate_locations = np.array([candidate.position for candidate, _ in candidate_source_list])
        candidate_sources = np.array([source for _, source in candidate_source_list])

        sorted_indices = np.argsort(candidate_locations)
        candidates = candidates[sorted_indices]
        candidate_locations = candidate_locations[sorted_indices]
        candidate_sources = candidate_sources[sorted_indices]

        first_non_empty_source = next(iter(non_empty_sources.keys()))
        candidate_link = non_empty_sources[first_non_empty_source][0].link
        candidate_link_length = self._get_link_length(candidate_link)
        threshold = self.coherence_length / (2 * self.n_index * candidate_link_length)

        # Two-pass clustering
        groups = []
        current_group = [0]
        cluster_start_pos = candidate_locations[0]
        
        for i in range(1, len(candidate_locations)):
            consecutive_diff = candidate_locations[i] - candidate_locations[i-1]
            distance_from_start = candidate_locations[i] - cluster_start_pos
            
            if consecutive_diff > threshold or distance_from_start > threshold:
                groups.append(np.array(current_group))
                current_group = [i]
                cluster_start_pos = candidate_locations[i]
            else:
                current_group.append(i)
        
        if current_group:
            groups.append(np.array(current_group))

        grouped_candidates = defaultdict(list)

        for group in groups:
            if len(group) == 0:
                continue
                
            cluster_position = candidate_locations[group].mean()
            base_candidate = candidates[group[0]]
            
            grouped_candidate = Candidate(
                link=base_candidate.link,
                position=cluster_position,
                type='grouped',
                generating_peak=None,
                path_in=None,
                path_out=None,
                score=0
            )

            contributing_sources = len(np.unique(candidate_sources[group]))
            grouped_candidates[contributing_sources].append(grouped_candidate)
            
        return dict(grouped_candidates)

    
    def _generate_rect_signal_for_candidate(self,
                                            candidate: Candidate,
                                            source_idx: int,
                                            olcr_ref_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                            olcr_perturb_dict: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        """
        Generate a rect signal for a candidate location using path-based prediction.

        Args:
            candidate: Candidate with link and position information
            source_idx: Source node index

        Returns:
            rect_signal: Array of same shape as self.opls with +1/-1 spikes at predicted locations
        """
        if self.opls is None:
            raise ValueError("opls grid must be provided to generate rect signals")

        _, I_ref_envelope = olcr_ref_dict[source_idx]
        _, I_pert_envelope = olcr_perturb_dict[source_idx]

        # Compute difference of envelopes
        diff_signal = I_pert_envelope - I_ref_envelope

        # Normalize difference signal to match rect signal scale
        if np.max(np.abs(diff_signal)) > 0:
            diff_signal_normalized = diff_signal / np.max(np.abs(diff_signal))
        else:
            diff_signal_normalized = diff_signal

        window = int(np.ceil(self.coherence_length / (self.opls[1] - self.opls[0])))
        # find the peaks in the difference signal
        peaks, _ = find_peaks(np.abs(diff_signal_normalized), height=1e-4, distance=window)
        peak_locations = self.opls[peaks]
        peak_heights = diff_signal_normalized[peaks]
        
        rect_signal = np.zeros_like(self.opls)
        
        node_a, node_b = candidate.link
        x = candidate.position
        L_link = self._get_link_length(candidate.link)
        
        # Get path bins to both nodes
        path_bins_to_a = self.binned_paths_per_source[source_idx][node_a]
        path_bins_to_b = self.binned_paths_per_source[source_idx][node_b]
        
        # Collect all valid round-trip combinations
        round_trip_data = []  # (opl, is_reflection)
        
        # Build a single list of "bins with side labels" so we only need one 2-for loop
        # over all possible (in_bin, out_bin) combinations.
        labeled_bins = ( [("A", pb) for pb in path_bins_to_a.path_bins] +
                         [("B", pb) for pb in path_bins_to_b.path_bins] )

        for in_side, pb_in in labeled_bins:
            for out_side, pb_out in labeled_bins:
                L_in = pb_in.representative_length
                L_out = pb_out.representative_length

                # Same-side round trips: AA / BB (reflection)
                if in_side == "A" and out_side == "A":
                    # source -> A -> fault(x) -> A -> source
                    opl = self.n_index * (L_in + 2 * (x * L_link) + L_out)
                    round_trip_data.append((opl, True))

                    # double reflection (<= 2 interactions with the fault)
                    # (source -> A -> fault(x) -> A -> fault(x) -> A -> source)
                    opl_dr = self.n_index * (L_in + 4 * (x * L_link) + L_out)
                    round_trip_data.append((opl_dr, True))

                elif in_side == "B" and out_side == "B":
                    # source -> B -> fault(1-x) -> B -> source
                    opl = self.n_index * (L_in + 2 * ((1 - x) * L_link) + L_out)
                    round_trip_data.append((opl, True))

                    # double reflection (<= 2 interactions with the fault)
                    # (source -> B -> fault(1-x) -> B -> fault(1-x) -> B -> source)
                    opl_dr = self.n_index * (L_in + 4 * ((1 - x) * L_link) + L_out)
                    round_trip_data.append((opl_dr, True))

                # Cross-side combinations: AB / BA
                elif in_side == "A" and out_side == "B":
                    # Reflection-transmission (<= 2 interactions with the fault)
                    # (source -> A -> fault -> B -> A -> source)
                    opl_rt = self.n_index * (L_in + L_link * (1 + 2 * x) + L_out)
                    round_trip_data.append((opl_rt, True))

                    # Transmission
                    # (source -> A -> fault -> B -> source)
                    opl_tx = self.n_index * (L_in + L_link + L_out)
                    round_trip_data.append((opl_tx, False))

                elif in_side == "B" and out_side == "A":
                    # Reflection-transmission (<= 2 interactions with the fault)
                    # (source -> B -> fault -> A -> B -> source)
                    opl_rt = self.n_index * (L_in + L_link * (3 - 2 * x) + L_out)
                    round_trip_data.append((opl_rt, True))

                    # Transmission
                    # (source -> B -> fault -> A -> source)
                    opl_tx = self.n_index * (L_in + L_link + L_out)
                    round_trip_data.append((opl_tx, False))
        
        # Group round trips by coherence length and create rect signal
        if round_trip_data:
            # Sort by OPL
            round_trip_data.sort(key=lambda x: x[0])
            
            # Bin by coherence length
            groups = []
            current_group_opls = [round_trip_data[0][0]]
            current_group_reflections = [round_trip_data[0][1]]
            
            for opl, is_refl in round_trip_data[1:]:
                if opl - current_group_opls[0] <= self.coherence_length:
                    current_group_opls.append(opl)
                    current_group_reflections.append(is_refl)
                else:
                    groups.append((current_group_opls, current_group_reflections))
                    current_group_opls = [opl]
                    current_group_reflections = [is_refl]
            
            groups.append((current_group_opls, current_group_reflections))
            
            # Create rect signal for each group
            for group_opls, group_reflections in groups:
                # Representative OPL (mean of group)
                center_opl = np.mean(group_opls)
                
                # Determine polarity by majority vote
                num_reflections = sum(group_reflections)
                is_reflection_bin = num_reflections >= len(group_reflections) / 2
                
                # Create rect window
                left = center_opl - self.coherence_length / 2
                right = center_opl + self.coherence_length / 2
                
                # Set rect values
                mask = (self.opls >= left) & (self.opls <= right)
                if is_reflection_bin:
                    rect_signal[mask] = 1
                else:
                    rect_signal[mask] = +1
        return rect_signal
    
    def score_candidate_multi_source(self,
                                    candidate: Candidate,
                                    source_peak_dict: Dict[int, Dict[np.ndarray, np.ndarray]],
                                    olcr_ref_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                    olcr_perturb_dict: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> float:
        """Score a candidate based on correlation between predicted and measured signals across all sources.

        Args:
            candidate: Candidate to score
            source_peak_dict: Dictionary mapping source index to (peak_locations, peak_heights)
            olcr_ref_dict: Dictionary mapping source index to (opls, reference_interferogram)
            olcr_perturb_dict: Dictionary mapping source index to (opls, perturbed_interferogram)

        Returns:
            Aggregated correlation score across all sources
        """
        if self.aggregation_method == 'sum':
            total_score = 0.0
            for source_idx in source_peak_dict.keys():
                source_score = self._score_candidate_single_source(
                    candidate, source_idx, olcr_ref_dict, olcr_perturb_dict
                )
                total_score += source_score
        else:  # product
            total_score = 1.0
            for source_idx in source_peak_dict.keys():
                source_score = self._score_candidate_single_source(
                    candidate, source_idx, olcr_ref_dict, olcr_perturb_dict
                )
                # Add small epsilon to avoid multiplication by zero
                total_score *= (source_score + 1e-10)

        candidate.score = total_score
        return total_score
    
    def _score_candidate_single_source(self,
                                      candidate: Candidate,
                                      source_idx: int,
                                      olcr_ref_dict: Dict[int, Tuple[np.ndarray, np.ndarray]],
                                      olcr_perturb_dict: Dict[int, Tuple[np.ndarray, np.ndarray]]) -> float:
        """
        Score a candidate based on correlation between predicted rect signal and actual difference signal.

        Args:
            candidate: Candidate to score
            source_idx: Source node index
            olcr_ref_dict: Reference OLCR measurements (opls, envelope)
            olcr_perturb_dict: Perturbed OLCR measurements (opls, envelope)

        Returns:
            Correlation coefficient (Pearson correlation)
        """
        # Generate predicted rect signal for this candidate
        rect_signal = self._generate_rect_signal_for_candidate(candidate, source_idx, olcr_ref_dict, olcr_perturb_dict)

        # Get actual difference signal from OLCR envelope measurements
        # NOTE: The interferogram data should already be envelopes
        _, I_ref_envelope = olcr_ref_dict[source_idx]
        _, I_pert_envelope = olcr_perturb_dict[source_idx]

        # Compute difference of envelopes
        diff_signal = I_pert_envelope - I_ref_envelope

        # Normalize difference signal to match rect signal scale
        if np.max(np.abs(diff_signal)) > 0:
            diff_signal_normalized = diff_signal / np.max(np.abs(diff_signal))
        else:
            diff_signal_normalized = diff_signal

        diff_signal_normalized = np.abs(diff_signal_normalized)
        
        # Calculate Pearson correlation coefficient
        if np.std(rect_signal) > 0 and np.std(diff_signal_normalized) > 0:
            correlation = np.corrcoef(rect_signal, diff_signal_normalized)[0, 1]
            # Handle NaN values (can occur with very small signals)
            if np.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # plt.figure(figsize=(10, 4),dpi=1000)
        # # plot the difference signal
        # plt.plot(self.opls*1e6, diff_signal_normalized,lw=0.1)
        # plt.plot(self.opls*1e6, rect_signal,lw=0.1)
        # plt.title(f"Link_{candidate.link}_position_{candidate.position:.3f}_source_{source_idx}_correlation_{correlation:.3f}")
        # plt.savefig(f"rect_signal_link_{candidate.link[0]}_{candidate.link[1]}_pos_{candidate.position:.3f}_source_{source_idx}.png")
        # plt.close()
        
        return correlation

    def _is_simple_path(self, path_in: Path, path_out: Path) -> bool:
        """Check if a path is simple (no loops or repeated directed link traversals) - CACHED."""
        cache_key = (tuple(path_in), tuple(path_out))
        if cache_key not in self._simple_path_cache:
            result = (len(set(path_in)) == len(path_in) and len(set(path_out)) == len(path_out))
            self._simple_path_cache[cache_key] = result
        return self._simple_path_cache[cache_key]

    def _is_valid(self,
                path: Path,
                path_prime: Path,
                candidate: Candidate,
                measured_peaks: List[float]) -> bool:
        """Check if a specific path pair is valid - CACHED."""
        cache_key = (tuple(path), tuple(path_prime), candidate.link, candidate.position)
        
        if cache_key not in self._path_validation_cache:
            if self._is_simple_path(path, path_prime):
                result = True
            else:
                result = self._has_all_valid_reduced_variants(path, path_prime, candidate, measured_peaks)
            self._path_validation_cache[cache_key] = result
        
        return self._path_validation_cache[cache_key]

    def _has_all_valid_reduced_variants(self,
                                path_in: Path,
                                path_out: Path, 
                                candidate: Candidate,
                                measured_peaks: List[float]) -> bool:
        """Check if all reduced variants of a complex path have corresponding peaks."""
        for simple_path_in in self._get_reduced_variants_of_path(path_in):
            if len(simple_path_in) < len(path_in):
                if not self._test_reduced_variant(simple_path_in, path_out, candidate, measured_peaks):
                    return False
        
        for simple_path_out in self._get_reduced_variants_of_path(path_out):
            if len(simple_path_out) < len(path_out):
                if not self._test_reduced_variant(path_in, simple_path_out, candidate, measured_peaks):
                    return False
        
        return True

    def _get_reduced_variants_of_path(self, path: Path) -> List[Path]:
        """Generate all simple variants of a path - CACHED."""
        path_tuple = tuple(path)
        if path_tuple not in self._reduced_variants_cache:
            self._reduced_variants_cache[path_tuple] = self._compute_reduced_variants(path)
        return self._reduced_variants_cache[path_tuple]

    def _compute_reduced_variants(self, path: Path) -> List[Path]:
        """Generate all simple variants of a path by recursively removing closed subpaths."""
        if len(path) <= 3:
            return [path]
        
        all_variants = set()
        paths_to_process = [path]
        
        while paths_to_process:
            current_path = paths_to_process.pop(0)
            current_tuple = tuple(current_path)
            all_variants.add(current_tuple)
            
            closed_subpaths = []
            for i in range(len(current_path) - 2):
                current_node = current_path[i]
                for j in range(i + 2, len(current_path)):
                    if current_path[j] == current_node:
                        closed_subpaths.append((i, j))
            
            closed_subpaths.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            
            if not closed_subpaths:
                continue
            
            for start_idx, end_idx in closed_subpaths:
                variant = current_path[:start_idx+1] + current_path[end_idx+1:]
                
                if len(variant) >= 2:
                    variant_tuple = tuple(variant)
                    if variant_tuple not in all_variants:
                        paths_to_process.append(variant)
        
        return [list(path_tuple) for path_tuple in all_variants]
    
    def _test_reduced_variant(self,
                            path_in: Path,
                            path_out: Path,
                            candidate: Candidate,
                            measured_peaks: List[float]) -> bool:
        """Test if a specific simple variant explains any measured peak."""
        node_a, node_b = candidate.link
        candidate_type = candidate.type
        if candidate_type == 'grouped':
            return True
        
        L_link = self._get_link_length(candidate.link)
        x = candidate.position
        
        simple_in_length = sum(self.graph.link_lengths.get(self._link_key(path_in[i], path_in[i+1]), 0)
                              for i in range(len(path_in) - 1))
        simple_out_length = sum(self.graph.link_lengths.get(self._link_key(path_out[i], path_out[i+1]), 0)
                               for i in range(len(path_out) - 1))
        
        if candidate_type == (node_a, node_a):
            L_simple = (simple_in_length + simple_out_length) * self.n_index
            predicted_peak = L_simple + 2 * self.n_index * L_link * x
        elif candidate_type == (node_b, node_b):
            L_simple = (simple_in_length + simple_out_length) * self.n_index
            predicted_peak = L_simple + 2 * self.n_index * L_link * (1 - x)
        elif candidate_type == (node_a, node_b):
            L_simple = simple_in_length * self.n_index + simple_out_length * self.n_index
            predicted_peak = L_simple + self.n_index * L_link * (2*x + 1)
        elif candidate_type == (node_b, node_a):
            L_simple = simple_in_length * self.n_index + simple_out_length * self.n_index
            predicted_peak = L_simple + self.n_index * L_link * (3 - 2*x)
        else:
            raise ValueError(f"Unknown candidate type: {candidate_type}")
        
        for measured_peak in measured_peaks:
            if abs(predicted_peak - measured_peak) < self.coherence_length:
                return True
        
        return False

    def localize_fault(self,
                       olcr_ref_dict: Dict[int,Tuple[np.ndarray, np.ndarray]],
                       olcr_perturb_dict: Dict[int,Tuple[np.ndarray, np.ndarray]],
                       links: Optional[List[Tuple[int, int]]] = None) -> Tuple[Tuple[int, int], float, float]:
        """
        Main method to localize fault in the network using multiple sources.
        """
        # Set opls from the first source if not already set
        if self.opls is None:
            first_source = list(olcr_ref_dict.keys())[0]
            self.opls = olcr_ref_dict[first_source][0]

        source_peak_dict = self._find_peaks_in_olcr_scan(olcr_ref_dict, olcr_perturb_dict)
        qualifying_peak_dict = self._find_peaks_with_adaptive_threshold()

        if links is None:
            links = [tuple(link.sorted_connected_nodes) for link in self.network.internal_links]

        common_candidates_per_link = self.find_common_candidates(
            source_peak_dict, qualifying_peak_dict, links
        )

        best_score = 0.0
        best_candidates = []
        valid_links = list(common_candidates_per_link.keys())

        # OPTIMIZED: Direct iteration instead of multiprocessing
        # After vectorization, this is faster than serialization overhead
        all_scores = {}
        for link in valid_links:
            for candidate in common_candidates_per_link[link]:
                score = self.score_candidate_multi_source(
                    candidate, source_peak_dict, olcr_ref_dict, olcr_perturb_dict
                )
                all_scores[(candidate.link, candidate.position)] = score

        for (link, position), score in all_scores.items():
            if score > best_score:
                best_score = score
                best_candidates = [(link, position, score)]
            elif score == best_score and score > 0:
                is_duplicate = False
                for existing_link, existing_pos, existing_score in best_candidates:
                    if (existing_link == link and
                        abs(existing_pos - position) < 1e-10):
                        is_duplicate = True
                        break

                if not is_duplicate:
                    best_candidates.append((link, position, score))

        return best_candidates[0] if best_candidates else (None, None, 0.0)

    def _find_peaks_in_olcr_scan(self,
                                 olcr_ref_dict: Dict[int,Tuple[np.ndarray, np.ndarray]],
                                 olcr_perturb_dict: Dict[int,Tuple[np.ndarray, np.ndarray]],
                                 noise_threshold: float = 0.05) -> Dict[int, np.ndarray]:
        """Identify peaks in the olcr scan above a given threshold."""
        sources = list(olcr_ref_dict.keys())
        if len(sources) < 1:
            raise ValueError("No sources found in olcr scan")
        if sorted(list(olcr_ref_dict.keys())) != sorted(list(olcr_perturb_dict.keys())):
            raise ValueError("Source keys in reference and perturbed OLCR scans do not match")

        self.source_peaks_dict = {}

        max_peaks = [max(olcr_perturb_dict[source][1] - olcr_ref_dict[source][1]) for source in sources]
        global_max = max(max_peaks) if max_peaks else 0

        for source in sources:
            ref_scan_x, ref_scan_data = olcr_ref_dict[source]
            perturbed_scan_x, perturbed_scan_data = olcr_perturb_dict[source]

            if not np.array_equal(ref_scan_x, perturbed_scan_x):
                raise ValueError(f"Scan x arrays do not match for source {source}")

            difference_signal = perturbed_scan_data - ref_scan_data

            dx = (ref_scan_x[-1] - ref_scan_x[0])/(len(ref_scan_x)-1)
            smooth_window = int(np.ceil(self.coherence_length / dx))
            if smooth_window % 2 == 0:
                smooth_window += 1
            if global_max > 0:
                difference_signal /= global_max

            peak_indices, _ = find_peaks(difference_signal,
                                         height=noise_threshold,
                                         distance=smooth_window)

            peak_locations = ref_scan_x[peak_indices]
            peak_heights = difference_signal[peak_indices]
            self.source_peaks_dict[source] = (peak_locations, peak_heights)

        return self.source_peaks_dict

    def _find_peaks_with_adaptive_threshold(self,
                                            initial_threshold: float = 0.5,
                                            reduction_factor: float = 0.25,
                                            max_reductions: int = 4) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """Apply adaptive thresholding to select qualifying peaks for candidate generation."""
        self.qualifying_peaks_dict = {}
        for source, (peak_locations, peak_heights) in self.source_peaks_dict.items():
            threshold = initial_threshold
            qualifying_peaks = []
            for _ in range(max_reductions):
                qualifying_peaks = [(loc, height) for loc, height in zip(peak_locations, peak_heights) if height >= threshold]
                if qualifying_peaks:
                    break
                threshold *= reduction_factor

                if not qualifying_peaks:
                    qualifying_peaks = []

            loc_array = np.array([loc for loc, _ in qualifying_peaks])
            height_array = np.array([height for _, height in qualifying_peaks])
            qualifying_peaks = (loc_array[0:1], height_array[0:1])  # TODO: Do something about this man...either its too small or too big
            # qualifying_peaks = (loc_array, height_array)
            # print(f"Source {source}: Selected {len(qualifying_peaks[0])} qualifying peaks with threshold {threshold:.4f}")

            self.qualifying_peaks_dict[source] = qualifying_peaks

        return self.qualifying_peaks_dict


# Helper function for batch processing with multiprocessing at the job level
def process_single_fault_localization(args):
    """
    Helper function for multiprocessing batch runs.
    Each worker processes an entire fault localization.
    """
    network, source_indices, max_hops, olcr_ref_dict, olcr_perturb_dict, n_index, coherence_length, opls, aggregation_method = args
    
    # Disable internal multiprocessing when running in batch mode
    localizer = FaultLocalizer(
        network=network,
        source_indices=source_indices,
        max_hops=max_hops,
        n_index=n_index,
        coherence_length=coherence_length,
        use_multiprocessing=False,  # Important: disable nested parallelism
        opls=opls,
        aggregation_method=aggregation_method
    )
    
    result = localizer.localize_fault(olcr_ref_dict, olcr_perturb_dict)
    return result


def run_batch_localizations(network_configs, max_workers=None):
    """
    Run multiple fault localizations in parallel.
    Proper multiprocessing at the job level, not candidate level.
    
    Args:
        network_configs: List of tuples (network, source_indices, max_hops, olcr_ref, olcr_pert, n_index, coh_length, opls, agg_method)
        max_workers: Number of parallel workers (defaults to CPU count)
    
    Returns:
        List of results from each fault localization
    """
    import os
    if max_workers is None:
        max_workers = os.cpu_count()
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(process_single_fault_localization, network_configs))
    
    return results


if __name__ == "__main__":
    import numpy as np
    import os
    import time
    from complex_network.networks.network_factory import generate_network
    from complex_network.networks.network_perturbator import NetworkPerturbator
    from complex_network.interferometry.olcr import OLCR
    from complex_network.networks.network_spec import NetworkSpec

    spec = NetworkSpec(
                    num_external_nodes=1,
                    num_internal_nodes=6,
                    network_type="delaunay",
                    network_shape="slab",
                    network_size=(200e-6, 200e-6),
                    external_offset=10e-6,
                    random_seed=84,
                    fully_connected=True,
                    node_S_mat_type='neumann'
                )

    network = generate_network(spec)

    wavelength_sampling = int(1e7)
    start_opl = 0
    max_opl = 1000e-6
    max_length_sampling = int(1e6)
    use_mp = True

    olcr_ref_s1 = OLCR(
        network=network,
        input_node=0,
        measurement_node=0,
        central_wavelength=1000e-9,
        bandwidth=400e-9,
        integeration_method='czt',
        num_wavelength_sample=wavelength_sampling,
        optical_path_length=[start_opl, max_opl],
        num_optical_path_length_sample=max_length_sampling,
        use_multi_proc=use_mp
    )

    I_ref_s1 = olcr_ref_s1.get_interferogram()
    opls = olcr_ref_s1.opls

    olcr_ref_s2 = OLCR(
        network=network,
        input_node=1,
        measurement_node=1,
        central_wavelength=1000e-9,
        bandwidth=400e-9,
        integeration_method='czt',
        num_wavelength_sample=wavelength_sampling,
        optical_path_length=[start_opl, max_opl],
        num_optical_path_length_sample=max_length_sampling,
        use_multi_proc=use_mp
    )

    I_ref_s2 = olcr_ref_s2.get_interferogram()

    link_index = 7
    ratio = 0.3
    perturbator = NetworkPerturbator(network)
    perturbator.add_perturbation_node(link_index=link_index, fractional_position=ratio)
    perturbed_network = perturbator.perturbed_network

    # Generate perturbed interferogram for source 1
    olcr_pert_s1 = OLCR(
        network=perturbed_network,
        input_node=0,
        measurement_node=0,
        central_wavelength=1000e-9,
        bandwidth=400e-9,
        integeration_method='czt',
        num_wavelength_sample=wavelength_sampling,
        optical_path_length=[start_opl, max_opl],
        num_optical_path_length_sample=max_length_sampling,
        use_multi_proc=use_mp
    )

    I_pert_s1 = olcr_pert_s1.get_interferogram()

    olcr_pert_s2 = OLCR(
        network=perturbed_network,
        input_node=1,
        measurement_node=1,
        central_wavelength=1000e-9,
        bandwidth=400e-9,
        integeration_method='czt',
        num_wavelength_sample=wavelength_sampling,
        optical_path_length=[start_opl, max_opl],
        num_optical_path_length_sample=max_length_sampling,
        use_multi_proc=use_mp
    )

    I_pert_s2 = olcr_pert_s2.get_interferogram()

    # Compute envelopes for correlation scoring
    env_ref_s1 = olcr_ref_s1._compute_envelope()
    env_ref_s2 = olcr_ref_s2._compute_envelope()
    env_pert_s1 = olcr_pert_s1._compute_envelope()
    env_pert_s2 = olcr_pert_s2._compute_envelope()

    # Create OLCR data dictionaries with envelopes
    # External node index is 6 (first external node in the network)
    external_node_idx1 = 6
    external_node_idx2 = 7
    olcr_ref_dict = {external_node_idx1: (opls, env_ref_s1), external_node_idx2: (opls, env_ref_s2)}
    olcr_perturb_dict = {external_node_idx1: (opls, env_pert_s1), external_node_idx2: (opls, env_pert_s2)}

    # Calculate coherence length
    coherence_length = 0.4412712003053032 * (1000e-9)**2 / 400e-9/2
    
    print("="*80)
    print("FAULT LOCALIZATION TEST - Correlation-based Scoring")
    print("="*80)
    print(f"True fault location: Link index={link_index}, Position={ratio:.4f}")
    print(f"Coherence length: {coherence_length*1e6:.4f} μm")
    print()

    # Test with 'sum' aggregation method
    print("Testing with 'sum' aggregation method:")
    print("-" * 40)
    localizer_sum = FaultLocalizer(
        network=network,
        source_indices=[6,7],
        max_hops=12,
        n_index=1.5,
        coherence_length=coherence_length,
        opls=opls,
        aggregation_method='sum'
    )

    start_time = time.time()
    result_sum = localizer_sum.localize_fault(olcr_ref_dict, olcr_perturb_dict)
    elapsed_sum = time.time() - start_time
    
    predicted_link_sum, predicted_pos_sum, score_sum = result_sum
    print(f"Predicted: Link={predicted_link_sum}, Position={predicted_pos_sum:.4f}, Score={score_sum:.6f}")
    print(f"Time elapsed: {elapsed_sum:.2f} seconds")
    
    # Check if prediction is correct
    true_link = tuple(sorted(network.internal_links[link_index].sorted_connected_nodes))
    if predicted_link_sum == true_link:
        pos_error = abs(predicted_pos_sum - ratio)
        print(f"✓ Link correctly identified!")
        print(f"  Position error: {pos_error:.4f} ({pos_error*100:.2f}%)")
    else:
        print(f"✗ Wrong link predicted. True link: {true_link}")
    print()

    # Test with 'product' aggregation method
    print("Testing with 'product' aggregation method:")
    print("-" * 40)
    localizer_product = FaultLocalizer(
        network=network,
        source_indices=[6,7],
        max_hops=12,
        n_index=1.5,
        coherence_length=coherence_length,
        opls=opls,
        aggregation_method='product'
    )

    start_time = time.time()
    result_product = localizer_product.localize_fault(olcr_ref_dict, olcr_perturb_dict)
    elapsed_product = time.time() - start_time
    
    predicted_link_product, predicted_pos_product, score_product = result_product
    print(f"Predicted: Link={predicted_link_product}, Position={predicted_pos_product:.4f}, Score={score_product:.6f}")
    print(f"Time elapsed: {elapsed_product:.2f} seconds")
    
    # Check if prediction is correct
    if predicted_link_product == true_link:
        pos_error = abs(predicted_pos_product - ratio)
        print(f"✓ Link correctly identified!")
        print(f"  Position error: {pos_error:.4f} ({pos_error*100:.2f}%)")
    else:
        print(f"✗ Wrong link predicted. True link: {true_link}")
    
    print()
    print("="*80)
    
