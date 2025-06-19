import numpy as np
import matplotlib.pyplot as plt
from complex_network.networks.network import Network
from scipy.signal import hilbert, savgol_filter
from typing import List, Union, Iterable
from collections import deque
from complex_network.components.link import Link

class OLCR:
    def __init__(self,
                network: Network,
                input_node: int,
                measurement_node: int,
                central_wavelength: float,
                bandwidth: float,
                num_wavelength_sample: int,
                max_optical_path_length: float,
                num_optical_path_length_sample: int):
        
        self.network = network
        self.input_node = input_node
        self.measurement_node = measurement_node
        self.central_lamda = central_wavelength
        self.bandwidth = bandwidth
        self.num_lamda = num_wavelength_sample
        self.max_opl = max_optical_path_length
        self.num_opl = num_optical_path_length_sample
        self.width_factor = 3.0

    def generate_broadband_source(self):
        # Generate an low-coherence broadband source
        # Gaussian standard deviation in wavelength
        self.sigma_lamda = self.bandwidth / (2 * np.sqrt(2 * np.log(2)))

        self.k_min = 2 * np.pi / (self.central_lamda + self.width_factor*self.sigma_lamda)
        self.k_max = 2 * np.pi / (self.central_lamda - self.width_factor*self.sigma_lamda)

        self.k = np.linspace(self.k_min, self.k_max, self.num_lamda)
        self.lamda = 2 * np.pi / self.k

        # Gaussian distribution for wavelength
        self.gaussian_wavelength = np.exp(-0.5 * ((self.lamda - self.central_lamda)
                                                   / (self.width_factor*self.sigma_lamda))**2)

        # Convert the gaussian to frequency space using Jacobian |d lambda/ dk| = 2.pi / k^2
        self.gaussian_frequency = self.gaussian_wavelength * (2 * np.pi / self.k ** 2)

        # Normalize the gaussian
        self.gaussian_frequency /= np.sum(self.gaussian_frequency)

        return self.k, self.gaussian_frequency

    def generate_interferogram(self):
        # Generate the interferogram by propagating the broadband source through the network
        self.opls = np.linspace(0, self.max_opl, self.num_opl)

        # Find the number of external nodes in the network
        self.num_external_nodes = len(self.network.external_nodes)

        self.interferogram = np.zeros_like(self.opls, dtype=np.float64)
        self.input_signal = np.zeros(self.num_external_nodes, dtype=np.complex128)

        # We will launch a beam-splitted light through the input node
        self.input_signal[self.input_node] = 1.0/np.sqrt(2)

        k, gaussian_frequency = self.generate_broadband_source()
        for k_val, gaussian_freq in zip(k, gaussian_frequency):
            # Propagate the reference beam to the detector (air propagation)
            E_reference = np.exp(1j* k_val * self.opls)/np.sqrt(2)
            # Propogate the sample signal through the network and get the output field
            E_sample = self.network.get_S_ee(k_val)@self.input_signal

            # Signal at the measurement node
            E_measurement = E_sample[self.measurement_node]

            # Add the incoherent intensities (low-coherence assumption )
            self.interferogram += gaussian_freq * np.abs(E_measurement + E_reference) ** 2

        return self.interferogram
    
    def _calculate_envelope(self,
                            smooth_envelope:bool=True,
                            window_size:int=11,
                            poly_order:int=3):
        # We need to the remove the background
        signal_actual = self.interferogram - np.mean(self.interferogram)

        # Calculate the envelope using Hilbert transform
        envelope = np.abs(hilbert(signal_actual))

        # Smooth the envelope if required (True by default)
        if smooth_envelope:
            envelope = savgol_filter(envelope, window_size, poly_order)      
        # restore the vertical offset so that the envelope sits on the signal
        envelope += np.mean(self.interferogram)
        return envelope
    
    def get_theoretical_opls(
        self,
        measurement_node: int | None = None,
        max_path_length: float | None = None,
        traversal_param: int | None = None,
    ) -> list[dict]:
        """Return every optical path from *input* → *measurement* that satisfies
        **both** of the following (optional) limits:

        1. *Geometric* + *scattering* optical length  ≤  ``max_path_length``  (if given).
        2. Maximum internal‑link traversal count      <  ``traversal_param``   (if given).

        ``traversal_param`` offers a *topological* cutoff that is often far
        faster than the traditional *metric* cutoff when loops abound.  If
        **neither** parameter is supplied, the method falls back to the class‑
        level default ``self.max_opl``.

        Returned structure for each path::

            {
                "path"         : [global_node_indices…],
                "opl"          : float,  # metres
                "max_traversal": int
            }
        """

        # ------------------------------------------------------------------
        # 1) Resolve defaults & constants
        # ------------------------------------------------------------------
        if measurement_node is None:
            measurement_node = self.measurement_node
        if max_path_length is None and traversal_param is None:
            max_path_length = self.max_opl  # fallback if nothing given

        n_const = 1.5
        k0 = 2 * np.pi / self.central_lamda

        ni = self.network.num_internal_nodes
        src_global = self.input_node + ni
        dst_global = measurement_node + ni

        banned_globals = set(self.network.external_node_indices)
        banned_globals.difference_update({src_global, dst_global})

        # ------------------------------------------------------------------
        # 2) Adjacency + link lookup (with n·L cached)
        # ------------------------------------------------------------------
        adjacency: dict[int, list[tuple[int, float]]] = {}
        link_lookup: dict[tuple[int, int], "Link"] = {}
        for link in self.network.links:
            a, b = link.node_indices
            opl_geom = n_const * link.length
            adjacency.setdefault(a, []).append((b, opl_geom))
            adjacency.setdefault(b, []).append((a, opl_geom))
            link_lookup[(a, b)] = link_lookup[(b, a)] = link

        # ------------------------------------------------------------------
        # 3) Helper – full OPL
        # ------------------------------------------------------------------
        def _opl_from_path(path: list[int]) -> float:
            total = 0.0
            for u, v in zip(path[:-1], path[1:]):
                total += n_const * link_lookup[(u, v)].length
            for idx, nidx in enumerate(path):
                if idx == 0 or idx == len(path) - 1:
                    continue  # cannot determine ports at ends reliably
                prev_node, next_node = path[idx - 1], path[idx + 1]
                node = self.network.get_node(nidx)
                try:
                    in_port = node.sorted_connected_nodes.index(prev_node)
                    out_port = node.sorted_connected_nodes.index(next_node)
                except ValueError:
                    continue
                S = node.get_S(k0) if node.node_type == "internal" else node.get_S
                amp = S[out_port, in_port]
                if amp != 0.0:
                    total += abs(np.angle(amp)) / k0
            return total

        # ------------------------------------------------------------------
        # 4) Helper – max internal traversal
        # ------------------------------------------------------------------
        def _max_internal_link_traversal(path: list[int]) -> int:
            counts: dict[tuple[int, int], int] = {}
            for u, v in zip(path[:-1], path[1:]):
                if u < ni and v < ni:
                    key = (u, v) if u < v else (v, u)
                    counts[key] = counts.get(key, 0) + 1
            return max(counts.values(), default=0)

        # ------------------------------------------------------------------
        # 5) BFS enumeration with dual pruning (OPL + traversal)
        # ------------------------------------------------------------------
        results: list[dict] = []
        queue: deque[tuple[int, list[int], float]] = deque()
        queue.append((src_global, [src_global], 0.0))

        while queue:
            current, path, opl_partial = queue.popleft()

            # destination reached ------------------------------------------------
            if current == dst_global and len(path) > 1:
                max_trav = _max_internal_link_traversal(path)
                if traversal_param is not None and max_trav >= traversal_param:
                    continue
                full_opl = _opl_from_path(path)
                if max_path_length is not None and full_opl > max_path_length:
                    continue
                results.append({"path": path, "opl": full_opl, "max_traversal": max_trav})
                continue

            # explore neighbours -------------------------------------------------
            for neigh, edge_opl in adjacency.get(current, []):
                if neigh in banned_globals:
                    continue
                new_opl = opl_partial + edge_opl
                if max_path_length is not None and new_opl > max_path_length:
                    continue
                # quick local traversal check (cheap upper bound)
                if traversal_param is not None:
                    # If we revisit an internal link, count might increase by 1
                    if current < ni and neigh < ni:
                        key = (current, neigh) if current < neigh else (neigh, current)
                        # estimate current count for this key in existing path
                        occurrences = sum(
                            1 for u, v in zip(path[:-1], path[1:])
                            if ((u, v) == key or (v, u) == key)
                        )
                        if occurrences + 1 >= traversal_param:
                            continue  # would exceed limit
                queue.append((neigh, path + [neigh], new_opl))

        return results
                         
    
    def plot_interferogram(self, plot_signal:bool=True,
                           plot_envelope:bool=True,
                           smooth_envelope:bool=True,
                           show_opls:bool=False,
                           signal_window:None|tuple[float,float]=None,
                           saveto:str=None):
        # Figure-size
        plt.figure(figsize=(10,6))

        interferogram = self.generate_interferogram()

        # Set the figure size and plot the interferogram
        plt.xlabel('Optical Path Difference (mu m)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Interferogram')

        if plot_signal:
            plt.plot(self.opls*1e6, interferogram, label='Interferogram', color='blue')

        if plot_envelope:
            plt.plot(self.opls*1e6, self._calculate_envelope(smooth_envelope), label='Envelope', color='red')

        if signal_window is not None:
            plt.xlim(signal_window)
        else:
            plt.xlim(0, self.max_opl*1e6)

        if show_opls:
            # Show the theoretical OPLs    
            all_paths = self.get_theoretical_opls(measurement_node=self.measurement_node, max_path_length=1.5e-3,traversal_param=5)
            labelled = set()                         # remember which classes we showed

            for path in all_paths:
                m = path["max_traversal"]
                path_length = path["opl"] * 1e6

                # pick colour + legend text for this class
                if m == 0:
                    colour, legend, key = "green",  "0", 0
                elif m == 1:
                    colour, legend, key = "orange", "1", 1
                elif m == 2:
                    colour, legend, key = "purple", "2", 2
                else:                   # m >= 3   → collapse into a single “>2” class
                    colour, legend, key = "black",  ">2", "gt2"

                label = legend if key not in labelled else None   # only first time
                labelled.add(key)

                plt.axvline(x=path_length,
                            color=colour, linestyle=".", lw=0.5,
                            label=label,alpha=0.75)


        plt.legend(loc=(1.01, 0.7), title="Max traversal",)
        if not saveto:
            plt.tight_layout()
            plt.show()
        else:
            plt.savefig(saveto, dpi=300)
            plt.close()
        





