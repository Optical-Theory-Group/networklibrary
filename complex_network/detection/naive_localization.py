"""
naive_localization.py

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
import time
import numpy as np
from scipy.signal import find_peaks
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict

# Import path generation functions from network_path_search
from complex_network.networks.network_path_search import (
    extract_graph_data,
    prepare_detailed_path_cache,
    _spec_fingerprint
)
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
    source: int # index of source node
    target: int # index of target node
    representative_length: float # representative optical path length of the bin
    paths: List[Path] # list of paths in this bin

@dataclass
class AllPathBins:
    """All the path bins that go to any particular node.
        collection of PathBin objects."""
    source: int # index of source node
    target: int # index of target nodes
    path_bins: List[PathBin] # list of PathBin objects


class FaultLocalizer:
    """
    Fault localizer
    """
    def __init__(self, 
                 network: Network,
                 source_indices: List[int],
                 max_hops: int,
                 n_index: float = 1.5, # TODO: account for the dispersion effects
                 coherence_length: float = 1.1e-6, #TODO:set it according to the coherence length of the source
                 interference_threshold: int = 2
                ):
        """
        Initialize the fault localizer.
        
        Args:
            network: Network object
            source_indices: List of source node indices we are using
            n_index: Refractive index of medium
            coherence_length: Coherence length for peak matching tolerance
            interference_threshold: Minimum number of paths to consider interference effects
        """
        self.network = network
        self.source_indices = [int(idx) for idx in source_indices]  # Support multiple sources
        self.n_index = float(n_index)
        self.coherence_length = float(coherence_length)
        self.interference_threshold = int(interference_threshold)
        self.max_hops = int(max_hops)


        # Extract lightweight graph data
        self.graph = extract_graph_data(network)
        
        # Generate fingerprint for caching
        if getattr(network, 'spec', None) is not None:
            self.fingerprint = _spec_fingerprint(network.spec)
        else:
            raise ValueError("Network was not created from a valid spec | cannot generate cache fingerprint")
        
        # Precompute paths for all source nodes
        self.binned_paths_per_source = self._precompute_paths_all_sources()
        
        
    def _precompute_paths_all_sources(self) -> Dict[int, List[AllPathBins]]:
        """Pre-compute all paths from all source nodes to each node using caching infrastructure."""
        binned_paths_per_source = {}
        
        for source_idx in self.source_indices:
            detailed_cache = prepare_detailed_path_cache(
                self.graph, self.fingerprint, source_idx, self.max_hops
            )
            
            all_binned_paths = []
            for key, paths in detailed_cache.items():
                source, target = key[0], key[1]

                # Check if the source is the current source node (should always be true)
                if source == source_idx:
                    # We will bin paths and make them into path classes
                    all_binned_paths.append(AllPathBins(source=source,
                                                       target=target,
                                                       path_bins=self._bin_paths(source, target, paths, self.coherence_length)))

            binned_paths_per_source[source_idx] = all_binned_paths
            
        return binned_paths_per_source

    
    def _link_key(self, a: int, b: int) -> Tuple[int, int]:
        """Return canonical link key as (min(a,b), max(a,b))."""
        return (min(a, b), max(a, b))
    
    def _get_link_length(self, link: Tuple[int, int]) -> float:
        """Get physical length of a link."""
        return self.graph.link_lengths.get(self._link_key(*link), 0.0)

    def _bin_paths(self,
                   source: int,
                   target: int,
                   paths: List[Tuple[Path, float]],
                   coherence_length: float) -> List[PathBin]:
        """
        Bin paths to a target node based on their optical path lengths.
        Paths whose lengths differ by less than half the coherence length are grouped into the same bin (PathClass).

        Args:
            target: The target node index for which paths are being binned.
            paths: List of (path, length) tuples to the target node.
            coherence_length: The coherence length used as the binning threshold.

        Returns:
            List of PathClass objects, each representing a bin of similar-length paths.
        """

        # Ensure paths are sorted by length for binning
        paths = sorted(paths, key=lambda x: x[1])
        spatial_resolution = coherence_length / (2 * self.n_index)

        path_bins: List[PathBin] = []
        num_paths = len(paths)

        binned_paths: List[Path] = []
        lengths: List[float] = []
        bin_start: int = 0
        
        # add the first path and length
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
                # Create a new path bin for the previous bin
                if binned_paths:
                    path_bins.append(PathBin(source=source,
                                                target=target,
                                                representative_length=(lengths[0]+lengths[-1])/2,
                                                paths=binned_paths.copy()
                                                ))
                    # Now start a new bin
                    bin_start = i
                    binned_paths = [path]
                    lengths = [length]


        # Bin any remaining paths
        if binned_paths:
            path_bins.append(PathBin(source=source,
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
        """
        Generate fault position candidates for a given link and peak from a specific source.
        
        Args:
            source_idx: Source node index
            link: Link tuple (node_A, node_B)
            peak: Peak optical path length for candidate generation
            measured_peaks: List of measured peaks for validation
            
        Returns:
            List of candidates
        """
        node_a, node_b = link
        L_link = self._get_link_length(link)
        
        # Get paths to both nodes from the specified source
        binned_paths = self.binned_paths_per_source[source_idx]
        path_bins_to_a = [all_path_bin for all_path_bin in binned_paths if all_path_bin.target == node_a][0]
        path_bins_to_b = [all_path_bin for all_path_bin in binned_paths if all_path_bin.target == node_b][0]

        candidates = self._generate_class_candidates(link, peak, path_bins_to_a, path_bins_to_b, L_link, measured_peaks)


        # print(source_idx,candidates)
        return candidates

    def _generate_class_candidates(self,
                                link: Tuple[int, int],
                                peak: float,
                                path_bins_to_a: AllPathBins,
                                path_bins_to_b: AllPathBins,
                                L_link: float,
                                measured_peaks: List[float]) -> List[Candidate]:
        """Generate candidates for a specific path class.
        Args:
            link: Link tuple (node_A, node_B)
            peak: Peak optical path length for candidate generation
            path_bins_to_a: AllPathBins object for paths to node A
            path_bins_to_b: AllPathBins object for paths to node B
            L_link: Physical length of the link
            measured_peaks: List of measured peaks for validation
        Returns:
            List of generated candidates
        """
        candidates = []

        # case AA
        for path_bin in path_bins_to_a.path_bins:
            for path_bin_prime in path_bins_to_a.path_bins:
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
                    # I need to test if the candidate is explained by a single complex path
                    if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                        if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                            candidates.append(candidate)
                        else:
                            pass
                    else:
                        candidates.append(candidate)

        # case BB
        for path_bin in path_bins_to_b.path_bins:
            for path_bin_prime in path_bins_to_b.path_bins:
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
                    # I need to test if the candidate is explained by a single complex path
                    if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                        if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                            candidates.append(candidate)
                        else:
                            pass
                    else:
                        candidates.append(candidate)
        # case AB
        for path_bin in path_bins_to_a.path_bins:
            for path_bin_prime in path_bins_to_b.path_bins:
            
                LAB = path_bin.representative_length * self.n_index + path_bin_prime.representative_length * self.n_index
                if peak - 3*self.n_index*L_link < LAB < peak - self.n_index*L_link:
                    x = (peak - LAB - self.n_index*L_link) / (2 * self.n_index * L_link)

                    candidate = Candidate(
                        link=link,
                        position=x,
                        type=(path_bin.target, path_bin_prime.target),
                        generating_peak=peak,
                        path_in=path_bin.paths,
                        path_out=path_bin_prime.paths[::-1])

                    # I need to test if the candidate is explained by a single complex path
                    if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                        if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                            candidates.append(candidate)
                        else:
                            pass
                            
                    else:
                        candidates.append(candidate)
        # case BA
        for path_bin in path_bins_to_b.path_bins:
            for path_bin_prime in path_bins_to_a.path_bins:
            
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
                    # I need to test if the candidate is explained by a single complex path
                    if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                        if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                            candidates.append(candidate)
                        else:
                            pass
                    else:
                        candidates.append(candidate)


        return candidates
    
    def find_common_candidates(self,
                            source_peak_dict: Dict[int, Dict[str, List[float]]],
                            qualifying_peak_dict: Dict[int, Dict[str, List[float]]], 
                            links: Optional[List[Tuple[int, int]]] = None
                            ) -> Dict[Tuple[int, int], List[Candidate]]:
        """
        Find candidates that matched the most number of times across all links.
            Args:
                source_peak_dict: Dictionary mapping source indices to dict peak locations
                qualifying_peak_dict: Dictionary mapping source indices to qualifying peak locations.
                links: Optional list of links to consider (defaults to all internal links)
            Returns:
                Dictionary mapping links to lists of common candidates
        """

        # If no specific links provided, consider all internal links
        if links is None:
            links = [tuple(link.sorted_connected_nodes) for link in self.network.internal_links]

        sources = list(qualifying_peak_dict.keys())
        source_location_link_pairs = [(source, loc, link, measured_peaks) for source in sources for loc in qualifying_peak_dict[source][0] for link in links for measured_peaks in [source_peak_dict[source][0]]]

        # A multiprocessed approach for finding all candidates for each link and source and peak
        with ProcessPoolExecutor() as executor:
            futures = {(source, link): executor.submit(self.generate_candidates_for_link, source, link, loc, measured_peaks)
                        for (source, loc, link, measured_peaks) in source_location_link_pairs}
            all_candidates = {key: future.result() for key, future in futures.items()}

        # Now, we will check for common candidates across all sources for each link
        max_common_link_candidates = {}
        global_max_frequency = 0

        for link in links:
            all_candidates_for_link = {source: all_candidates[(source, link)] for source in sources}

            candidates_with_freq = self._get_candidates_with_frequency(
                all_candidates_for_link, sources
            )

            # max frequency is the highest key in candidates_with_freq
            link_frequencies = list(candidates_with_freq.keys())
            link_max_freq = max(link_frequencies) if link_frequencies else 0

            if link_max_freq > global_max_frequency:
                global_max_frequency = link_max_freq
                # Add candidates with this frequency

        # filter to only keep candidates with max_frequency
        for link in links:
            all_candidates_for_link = {source: all_candidates[(source, link)] for source in sources}
            # Get candidates with frequencies for this link
            candidates_with_freq = self._get_candidates_with_frequency(
                all_candidates_for_link, sources
            )
            # See if this link has candidates with the global max frequency
            if global_max_frequency in candidates_with_freq.keys():
                max_common_link_candidates[link] = candidates_with_freq[global_max_frequency]

        return max_common_link_candidates

    def _get_candidates_with_frequency(self,
                                    all_candidates_for_link: Dict[int, List[Candidate]],
                                    sources: List[int]) -> Dict[int, List[Candidate]]:
        """
        We will look at candidates from all sources for a particular link and see how many 
        sources they appeared in (within one coherence length). And return one candidate
        representing those candidates along with the frequency count."""
        if len(sources) < 1:
            raise ValueError("No sources provided")
        
        # Filter out sources with empty candidate lists
        non_empty_sources = {source: candidates for source, candidates in 
                            all_candidates_for_link.items() if candidates}
        
        # If all sources have empty candidate lists, return empty dict
        if not non_empty_sources:
            return {}
        
        candidate_source_list = [(candidate, source) for source, candidates in
                                non_empty_sources.items() for candidate in candidates]
        candidates = np.array([candidate for candidate, _ in candidate_source_list])
        candidate_locations = np.array([candidate.position for candidate, _ in candidate_source_list])
        candidate_sources = np.array([source for _, source in candidate_source_list])

        # Sort candidates by location
        sorted_indices = np.argsort(candidate_locations)
        candidates = candidates[sorted_indices]
        candidate_locations = candidate_locations[sorted_indices]
        candidate_sources = candidate_sources[sorted_indices]

        # Calculate threshold
        first_non_empty_source = next(iter(non_empty_sources.keys()))
        candidate_link = non_empty_sources[first_non_empty_source][0].link
        candidate_link_length = self._get_link_length(candidate_link)
        threshold = self.coherence_length / (2 * self.n_index * candidate_link_length)

        # TWO-PASS CLUSTERING: Break chains by checking distance from cluster start
        groups = []
        current_group = [0]
        cluster_start_pos = candidate_locations[0]
        
        for i in range(1, len(candidate_locations)):
            # Check both: consecutive diff AND distance from cluster start
            consecutive_diff = candidate_locations[i] - candidate_locations[i-1]
            distance_from_start = candidate_locations[i] - cluster_start_pos
            
            # Break if EITHER condition is violated
            if consecutive_diff > threshold or distance_from_start > threshold:
                groups.append(np.array(current_group))
                current_group = [i]
                cluster_start_pos = candidate_locations[i]
            else:
                current_group.append(i)
        
        # Add last group
        if current_group:
            groups.append(np.array(current_group))

        # Create grouped candidates
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



    
    def score_candidate_multi_source(self, 
                                    candidate: Candidate, 
                                    source_peak_dict: Dict[int, Dict[np.ndarray, np.ndarray]]) -> float:
        """
        Score a candidate based on exponential distance scoring across all sources.
        Uses ALL peaks for scoring, not just qualifying peaks.
        
        Args:
            candidate: Candidate to score
            source_peak_dict: Dictionary mapping source indices to dict with 'locations' and other keys
            
        Returns:
            Total exponential score based on position prediction accuracy across all sources
        """

        total_score = 0.0
        for source_idx, source_data in source_peak_dict.items():
            all_peaks_loc, _ = source_data
            
            # Score using peaks from this source and paths for this source
            source_score = self._score_candidate_single_source(candidate, source_idx, all_peaks_loc)
            total_score += source_score    
        candidate.score = total_score

        return total_score
    
    def _score_candidate_single_source(self, 
                                      candidate: Candidate, 
                                      source_idx: int,
                                      measured_peaks: np.ndarray) -> float:
        """
        Score a candidate using peaks from a single source with exponential distance-based scoring.
        
        Args:
            candidate: Candidate to score
            source_idx: Source node index
            measured_peaks: List of measured peak optical path lengths from this source
            
        Returns:
            Sum of exponential scores based on position prediction accuracy
        """
        
        L_link = self._get_link_length(candidate.link)
        node_a, node_b = candidate.link
        x = candidate.position
        
        # Get paths to both nodes from the specified source
        binned_paths = self.binned_paths_per_source[source_idx]
        path_bins_to_a = [all_path_bin for all_path_bin in binned_paths if all_path_bin.target == node_a][0]
        path_bins_to_b = [all_path_bin for all_path_bin in binned_paths if all_path_bin.target == node_b][0]
        
        # For each measured peak, find the best predicted position and calculate exponential score
        total_score = 0.0
        
        for measured_peak in measured_peaks:
            best_score_for_peak = 0.0
            
            # best_score_aa = 0.0
            # best_score_ab = 0.0
            # best_score_ba = 0.0
            # best_score_bb = 0.0
            # Type AA: Source → A → fault → A → source
            for path_bin in path_bins_to_a.path_bins:
                for path_bin_prime in path_bins_to_a.path_bins:
                    L_AA = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                    
                    # Calculate predicted position from the measured peak
                    # measured_peak = L_AA + 2 * n_index * L_link * x_pred
                    # Solve for x_pred: x_pred = (measured_peak - L_AA) / (2 * n_index * L_link)
                    if L_link > 0:
                        x_pred = (measured_peak - L_AA) / (2 * self.n_index * L_link)
                        
                        # Check if predicted position is within valid range and coherence length
                        if 0 <= x_pred <= 1:
                            predicted_length = L_AA + 2 * self.n_index * L_link * x_pred
                            
                            if abs(measured_peak - predicted_length) < self.coherence_length:
                                # Validate complex paths if needed
                                is_valid = True
                                if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                                    is_valid = self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks)
                                
                                if is_valid:
                                    # Calculate distance between candidate position and predicted position
                                    position_distance = abs(x - x_pred)
                                    # Convert to physical distance for consistency
                                    physical_distance = position_distance * L_link
                                    # Calculate exponential score (higher when positions are closer)
                                    score = np.exp(-physical_distance / (self.coherence_length / (2*self.n_index)))
                                    best_score_for_peak = max(best_score_for_peak, score)
                                    # best_score_for_peak += score
                                    # best_score_aa = max(best_score_aa, score)
                            #     else:
                            #         score = 0.0
                            # else:
                            #     score = 0.0

            # Type BB: Source → B → fault → B → source
            for path_bin in path_bins_to_b.path_bins:
                for path_bin_prime in path_bins_to_b.path_bins:
                    L_BB = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                    
                    # measured_peak = L_BB + 2 * n_index * L_link * (1-x_pred)
                    # Solve for x_pred: x_pred = 1 - (measured_peak - L_BB) / (2 * n_index * L_link)
                    if L_link > 0:
                        x_pred = 1 - (measured_peak - L_BB) / (2 * self.n_index * L_link)
                        
                        if 0 <= x_pred <= 1:
                            predicted_length = L_BB + 2 * self.n_index * L_link * (1 - x_pred)
                            
                            if abs(measured_peak - predicted_length) < self.coherence_length:
                                is_valid = True
                                if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                                    is_valid = self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks)
                                
                                if is_valid:
                                    position_distance = abs(x - x_pred)
                                    physical_distance = position_distance * L_link
                                    score = np.exp(-physical_distance / (self.coherence_length / (2*self.n_index)))
                                    best_score_for_peak = max(best_score_for_peak, score)
                                    # best_score_for_peak += score
                                    # best_score_bb = max(best_score_bb, score)
                            #     else:
                            #         score = 0.0
                            # else:
                            #     score = 0.0

            # Type AB: Source → A → fault → B → source
            for path_bin in path_bins_to_a.path_bins:
                for path_bin_prime in path_bins_to_b.path_bins:
                    L_AB = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                    
                    # measured_peak = L_AB + n_index * L_link * (2*x_pred + 1)
                    # Solve for x_pred: x_pred = (measured_peak - L_AB - n_index * L_link) / (2 * n_index * L_link)
                    if L_link > 0:
                        x_pred = (measured_peak - L_AB - self.n_index * L_link) / (2 * self.n_index * L_link)
                        
                        if 0 <= x_pred <= 1:
                            predicted_length = L_AB + self.n_index * L_link * (2 * x_pred + 1)
                            
                            if abs(measured_peak - predicted_length) < self.coherence_length:
                                is_valid = True
                                if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                                    is_valid = self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks)
                                
                                if is_valid:
                                    position_distance = abs(x - x_pred)
                                    physical_distance = position_distance * L_link
                                    score = np.exp(-physical_distance / (self.coherence_length / (2*self.n_index)))
                                    best_score_for_peak = max(best_score_for_peak, score)
                                    best_score_for_peak += score
                                    # best_score_ab = max(best_score_ab, score)
                            #     else:
                            #         score = 0.0
                            # else:
                            #     score = 0.0

            # Type BA: Source → B → fault → A → source
            for path_bin in path_bins_to_b.path_bins:
                for path_bin_prime in path_bins_to_a.path_bins:
                    L_BA = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                    
                    # measured_peak = L_BA + n_index * L_link * (3 - 2*x_pred)
                    # Solve for x_pred: x_pred = (3*n_index*L_link + L_BA - measured_peak) / (2 * n_index * L_link)
                    if L_link > 0:
                        x_pred = (3 * self.n_index * L_link + L_BA - measured_peak) / (2 * self.n_index * L_link)
                        
                        if 0 <= x_pred <= 1:
                            predicted_length = L_BA + self.n_index * L_link * (3 - 2 * x_pred)
                            
                            if abs(measured_peak - predicted_length) < self.coherence_length:
                                is_valid = True
                                if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                                    is_valid = self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks)
                                
                                if is_valid:
                                    position_distance = abs(x - x_pred)
                                    physical_distance = position_distance * L_link
                                    score = np.exp(-physical_distance / (self.coherence_length / (2*self.n_index)))
                                    best_score_for_peak = max(best_score_for_peak, score)
                                    # best_score_for_peak += score
                                    # best_score_ba = max(best_score_ba, score)
                            #     else:
                            #         score = 0.0
                            # else:
                            #     score = 0.0
            
            # Add the best score found for this peak (0 if no valid predictions within coherence length)
            total_score += best_score_for_peak
            # total_score += (best_score_aa + best_score_ab + best_score_ba + best_score_bb)

        return total_score
    
    def num_peaks_explained(self, candidate: Candidate, measured_peaks: List[float]) -> int:
        """
        This gives the number of peaks a candidate can explain.
        
        Args:
            candidate: Candidate under consideration
            measured_peaks: List of measured peak optical path lengths
            
        Returns:
            Number of peaks the candidate can explain
        """
        L_link = self._get_link_length(candidate.link)
        node_a, node_b = candidate.link
        x = candidate.position
        
        # Get paths to both nodes
        path_bins_to_a = [all_path_bin for all_path_bin in self.binned_paths if all_path_bin.target == node_a][0]
        path_bins_to_b = [all_path_bin for all_path_bin in self.binned_paths if all_path_bin.target == node_b][0]
        
        
        # For each measured peak, check if this candidate can explain it
        num_explained = 0
        for measured_peak in measured_peaks:
            
            aa_matched = False
            # Type AA: Source → A → fault → A → source
            for path_bin in path_bins_to_a.path_bins:
                for path_bin_prime in path_bins_to_a.path_bins:

                    L_AA = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                    predicted_length = L_AA + 2 * self.n_index * L_link * x
                    if abs(measured_peak - predicted_length) < self.coherence_length:
                        # If multiple paths in the bin, we will add the score without complexity check

                        # I need to test if the candidate is explained by a single complex path
                        if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                            if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                                aa_matched = True
                            else:
                                aa_matched = False
                        else:
                            aa_matched = True


            # Type BB: Source → B → fault → B → source
            bb_matched = False
            for path_bin in path_bins_to_b.path_bins:
                for path_bin_prime in path_bins_to_b.path_bins:
                    L_BB = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                    predicted_length = L_BB + 2 * self.n_index * L_link * (1-x)
                    if abs(measured_peak - predicted_length) < self.coherence_length:

                        # I need to test if the candidate is explained by a single complex path
                        if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                            if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                                bb_matched = True
                            else:
                                bb_matched = False
                        else:
                            bb_matched = True

            # Type AB: Source → A → fault → B → source
            ab_matched = False
            for path_bin in path_bins_to_a.path_bins:
                for path_bin_prime in path_bins_to_b.path_bins:
                    L_AB = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                    predicted_length = L_AB + self.n_index * L_link * (2*x + 1)
                    if abs(measured_peak - predicted_length) < self.coherence_length:

                        if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                            if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                                ab_matched = True
                            else:
                                ab_matched = False
                        else:
                            ab_matched = True

            # Type BA: Source → B → fault → A → source
            ba_matched = False
            for path_bin in path_bins_to_b.path_bins:
                for path_bin_prime in path_bins_to_a.path_bins:
                    L_BA = (path_bin.representative_length + path_bin_prime.representative_length) * self.n_index
                    predicted_length = L_BA + self.n_index * L_link * (3 - 2*x)
                    if abs(measured_peak - predicted_length) < self.coherence_length:

                        if len(path_bin.paths) < self.interference_threshold and len(path_bin_prime.paths) < self.interference_threshold:
                            if self._is_valid(path_bin.paths[0], path_bin_prime.paths[0], candidate, measured_peaks):
                                ba_matched = True
                            else:
                                ba_matched = False
                        else:
                            ba_matched = True

            explained = aa_matched or bb_matched or ab_matched or ba_matched
            
            # if explained by any path class, increment score
            if explained:
                num_explained += 1

        candidate.num_explained = num_explained
        return num_explained

    def _is_simple_path(self, path_in: Path, path_out: Path) -> bool:
        """
        Check if a path is simple (no loops or repeated directed link traversals).    
        Args:
            path_in: Inbound path to the fault
            path_out: Outbound path from the fault
            
        Returns:
            True if the path is simple, False if complex
        """
        # Check for repeated nodes in the half-paths
        if len(set(path_in)) < len(path_in) or len(set(path_out)) < len(path_out):
            return False
        
        # No repeated nodes, so no closed subpaths
        return True

    # defining a valid function to see if a candidate is valid
    # I need to test if the candidate is explained by a single complex path
    def _is_valid(self,
                path: Path,
                path_prime: Path,
                candidate: Candidate,
                measured_peaks: List[float]) -> bool:
        """Check if a specific path pair is valid."""
        
        if self._is_simple_path(path, path_prime):    
            return True
        else:
            # Complex path validation
            return self._has_all_valid_reduced_variants(path, path_prime, candidate, measured_peaks)

    
    def _has_all_valid_reduced_variants(self,
                                path_in: Path,
                                path_out: Path, 
                                candidate: Candidate,
                                measured_peaks: List[float]) -> bool:
        """
        Check if all reduced variants of a complex path has a corresponding peak in measured_peaks.
        Tests one variant at a time - stops as soon as one fails.
        
        Args:
            path_in: Complex inbound path
            path_out: Complex outbound path
            candidate: The fault candidate
            measured_peaks: All measured peaks
            
        Returns:
            True if at least one simpler variant has a corresponding measured peak
        """
        # Test reduced variants of the complex path

        # Test paths one by one from the inbound path
        for simple_path_in in self._get_reduced_variants_of_path(path_in):
            if len(simple_path_in) < len(path_in):  # Only if actually simpler
                if not self._test_reduced_variant(simple_path_in, path_out, candidate, measured_peaks):
                    return False
        
        # Test paths one by one from the outbound path  
        for simple_path_out in self._get_reduced_variants_of_path(path_out):
            if len(simple_path_out) < len(path_out):  # Only if actually simpler
                if not self._test_reduced_variant(path_in, simple_path_out, candidate, measured_peaks):
                    return False
        
        return True

    def _get_reduced_variants_of_path(self, path: Path) -> List[Path]:
        """
        Generate all simple variants of a path by recursively removing closed subpaths.
        
        A k-closed subpath means nodes at positions i and i+k are the same.
        This implementation recursively finds and removes all closed subpaths,
        returning all generated path variants with duplicates removed.
        
        Args:
            path: Input path that may contain closed subpaths    
        Returns:
            List of all path variants with closed subpaths removed (including original)
        """
        if len(path) <= 3:  # Cannot have closed subpaths for the half-path
            # should be less than 3, but since we assume that faults cannot be
            # located in a external link, we should make it less than 4
            return [path]
        
        # Use a set to track all unique paths we've generated
        all_variants = set()
        
        # Queue for paths to process
        paths_to_process = [path]
        
        while paths_to_process:
            current_path = paths_to_process.pop(0)
            current_tuple = tuple(current_path)
            
            # Add current path to our collection
            all_variants.add(current_tuple)
            
            # Find all k-closed subpaths in current path
            closed_subpaths = []
            for i in range(len(current_path) - 2):
                current_node = current_path[i]
                for j in range(i + 2, len(current_path)):
                    if current_path[j] == current_node:
                        closed_subpaths.append((i, j))
            
            # Sort by start position, then by length (prefer longer subpaths first)
            closed_subpaths.sort(key=lambda x: (x[0], -(x[1] - x[0])))
            
            # If no closed subpaths, this is a simple path - no further processing needed
            if not closed_subpaths:
                continue
            
            # Create variants by removing each closed subpath
            for start_idx, end_idx in closed_subpaths:
                # Remove the closed subpath: keep node at start_idx, skip to end_idx+1
                variant = current_path[:start_idx+1] + current_path[end_idx+1:]
                
                # Only add valid paths (at least 2 nodes)
                if len(variant) >= 2:
                    variant_tuple = tuple(variant)
                    
                    # Only process if we haven't seen this variant before
                    if variant_tuple not in all_variants:
                        paths_to_process.append(variant)
        
        # Convert back to list of lists and return
        return [list(path_tuple) for path_tuple in all_variants]
    
    def _test_reduced_variant(self,
                            path_in: Path,
                            path_out: Path,
                            candidate: Candidate,
                            measured_peaks: List[float]) -> bool:
        """
        Test if a specific simple variant explains any measured peak.
        
        Args:
            path_in: Simplified inbound path  
            path_out: Simplified outbound path
            candidate: Fault candidate
            measured_peaks: All measured peaks           
        Returns:
            True if this variant explains any measured peak
        """
        node_a, node_b = candidate.link
        candidate_type = candidate.type
        if candidate_type == 'grouped':
            # Cannot test grouped candidates
            return True
        L_link = self._get_link_length(candidate.link)
        x = candidate.position
        # Calculate physical lengths
        simple_in_length = sum(self.graph.link_lengths.get(self._link_key(path_in[i], path_in[i+1]), 0)
                              for i in range(len(path_in) - 1))
        simple_out_length = sum(self.graph.link_lengths.get(self._link_key(path_out[i], path_out[i+1]), 0)
                               for i in range(len(path_out) - 1))
        
        # Calculate prediction based on candidate type
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
        
        # Check if this prediction matches any measured peak
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
        
        Args:
            links: List of links to test (if None, tests all internal links)
        Returns:
            List of tuples (link, position, score) for all candidates with the highest score
        """
        source_peak_dict = self._find_peaks_in_olcr_scan(olcr_ref_dict, olcr_perturb_dict)
        qualifying_peak_dict = self._find_peaks_with_adaptive_threshold()
        
        if links is None:
            links = [tuple(link.sorted_connected_nodes) for link in self.network.internal_links]

        # Find common candidates across all sources
        common_candidates_per_link = self.find_common_candidates(
            source_peak_dict, qualifying_peak_dict, links
        )
        
        best_score = 0.0
        best_candidates = []

        # Links with max common candidates
        valid_links = list(common_candidates_per_link.keys())

        # Lets score candidates using multiprocessing
        # create a link candidates tuple for each link
        candidate_source_tuple = [(candidate, source_peak_dict) for link in valid_links for candidate in common_candidates_per_link[link]]

        with ProcessPoolExecutor() as executor:
            futures = {(candidate.link, candidate.position): executor.submit(self.score_candidate_multi_source, candidate, source_peak_dict)
                        for (candidate, source_peak_dict) in candidate_source_tuple}
            all_scores = {key: future.result() for key, future in futures.items()}

        # Now, we will find the best candidates across all links
        for (link, position), score in all_scores.items():
            if score > best_score:
                best_score = score
                best_candidates = [(link, position, score)]
            elif score == best_score and score > 0:
                # Check if this candidate is a duplicate (same link and very similar position)
                is_duplicate = False
                for existing_link, existing_pos, existing_score in best_candidates:
                    if (existing_link == link and 
                        abs(existing_pos - position) < 1e-10):  # Very small tolerance for floating point
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    best_candidates.append((link, position, score))

        # Return best candidates with highest score if multiple have the same highest score
        # return the first one

        return best_candidates[0] # Return only the best candidate for now

    def _find_peaks_in_olcr_scan(self,
                                 olcr_ref_dict: Dict[int,Tuple[np.ndarray, np.ndarray]],
                                 olcr_perturb_dict: Dict[int,Tuple[np.ndarray, np.ndarray]],
                                 noise_threshold: float = 0.05) -> Dict[int, np.ndarray]:
        """
        Identify peaks in the olcr scan above a given threshold.
        
        Args:
            olcr_ref_dict: dictionary of reference scan data (Envelope signal of interferrogram)
            olcr_perturb_dict: dictionary of perturbed scan data (Envelope signal of interferrogram)
            threshold: Minimum height for a peak to be considered
        Returns:
            Dictionary mapping source indices to lists of peak locations
        """
        # check if there are any sources
        sources = list(olcr_ref_dict.keys())
        if len(sources) < 1:
            raise ValueError("No sources found in olcr scan")
        if sorted(list(olcr_ref_dict.keys())) != sorted(list(olcr_perturb_dict.keys())):
            raise ValueError("Source keys in reference and perturbed OLCR scans do not match maybe check the ordering or the inputs")   
        sources = list(olcr_ref_dict.keys())

        self.source_peaks_dict = {}
        
        max_peaks = [max(olcr_perturb_dict[source][1] - olcr_ref_dict[source][1]) for source in sources]
        # find global max across all sources
        global_max = max(max_peaks) if max_peaks else 0

        for source in sources:
            ref_scan_x, ref_scan_data = olcr_ref_dict[source]
            perturbed_scan_x, perturbed_scan_data = olcr_perturb_dict[source]

            # Check that x arrays are the same
            if not np.array_equal(ref_scan_x, perturbed_scan_x):
                raise ValueError(f"Scan x arrays do not match for source {source}")

            difference_signal = perturbed_scan_data - ref_scan_data

            dx = (ref_scan_x[-1] - ref_scan_x[0])/(len(ref_scan_x)-1)
            smooth_window = int(np.ceil(self.coherence_length / dx))
            if smooth_window % 2 == 0:
                smooth_window += 1
            if global_max > 0:
                difference_signal /= global_max

            # find peaks above threshold
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
        """ From the detected peaks, we will apply an adaptive thresholding to select qualifying peaks for candidate generation."""

        self.qualifying_peaks_dict = {}
        for source, (peak_locations, peak_heights) in self.source_peaks_dict.items():
            threshold = initial_threshold
            qualifying_peaks = []
            for _ in range(max_reductions):
                qualifying_peaks = [(loc, height) for loc, height in zip(peak_locations, peak_heights) if height >= threshold]
                if qualifying_peaks:
                    break
                threshold *= reduction_factor  # Reduce threshold if no peaks found

                # If after max reductions no peaks found, we keep an empty list
                if not qualifying_peaks:
                    qualifying_peaks = []

            loc_array = np.array([loc for loc, _ in qualifying_peaks])
            height_array = np.array([height for _, height in qualifying_peaks])
            qualifying_peaks = (loc_array[0:1], height_array[0:1]) # remove this later

            # max value of qualifying peaks
            # max_qualifying_peak = height_array.max()
            # # get corresponding locations
            # max_qualifying_locations = loc_array[height_array == max_qualifying_peak]
            # qualifying_peaks = (max_qualifying_locations, max_qualifying_peak)
            self.qualifying_peaks_dict[source] = qualifying_peaks

        return self.qualifying_peaks_dict