""" DORT or decomposition of the time reversal operator is a method to detect targets in complex media.
    It is based on the eigenvalue decomposition of the time reversal operator, 
    which allows to identify the dominant scattering paths and thus locate the targets. 
    This module implements the DORT algorithm for target detection in complex networks.
    The DORT algorithm consists of the following steps:
    1. Construct the time reversal operator from the measured signals of different frequencies 
        T(k) = Delta S Delta S^dagger.
    2. Perform eigenvalue decomposition of the time reversal operator.
    3. Identify the dominant eigenvalues and corresponding eigenvectors - lambda_1 and v_1.
    4. Set up green's function for each link - g(x) = a(k)e^(ikx) + b(k)e^(-ikx), where a(k) and b(k)
         are the coefficients of the forward and backward propagating waves, respectively.
    5. Set up the projection operator that looks at the projection value of the eigenvectors on the green's function
        P = <v_1, g(x)>.
    6. Points are tested by calculating the coherent or incoherent sum of projection across all frequencies.
    7. Test each link using a coarse grid search to see which link has the highest projection value, which indicates the presence of a target.
    8. Do a fine grid search around the link with the highest projection value to accurately locate the target."""


from typing import Dict, List, Tuple, Optional
from complex_network.networks.network import Network
import numpy as np


class DORT:

    def __init__(self,healthy_S_matrix_array: np.ndarray,
                    spectrum: Tuple[List[float], List[complex]],
                    healthy_network: Network,
                    faulty_S_matrix_array: Optional[np.ndarray] = None,
                    coarse_grid_sampling: int = 50,
                    fine_grid_sampling: int = 2000,
                    edge_margin: float = 0.05):

        """The constructor performs the per-network precompute only. It depends solely on the
        healthy network and the spectrum, NOT on the faulty measurement, so a single DORT
        object can be built once for a network and reused to localise many different faults by
        passing each faulty S-matrix to localise_fault. This is what makes a large statistical
        sweep efficient: the expensive field solve runs once per network, not once per fault.

        Args:
            healthy_S_matrix_array: A 3D numpy array of shape (n_frequencies, n_ext, n_ext) containing the scattering matrices
              of the healthy network for each frequency. Order has to match the order of frequencies in the spectrum tuple.

            spectrum: A tuple of two lists, where the first list contains the frequencies and the second list contains the corresponding spectrum values.

            healthy_network: The healthy network object, which contains the information about
              the network topology and link lengths.

            faulty_S_matrix_array: Optional 3D numpy array of shape (n_frequencies, n_ext, n_ext) with the scattering
              matrices of the faulty network for each frequency (same frequency order as the spectrum). May be omitted
              here and supplied per call to localise_fault instead, which is the recommended pattern for sweeps that
              reuse one healthy precompute across many faults.

            coarse_grid_sampling: The number of points to sample in the coarse grid search
            for link identification.

            fine_grid_sampling: The number of points to sample in the fine grid search for position identification.

            edge_margin: The margin to avoid the edges of the link when searching for targets due to edge effects.
        """
        
        self.healthy_S_matrix_array = healthy_S_matrix_array
        self.Faulty_S_matrix_array = faulty_S_matrix_array
        self.healthy_network = healthy_network
        self.coarse_grid_sampling = coarse_grid_sampling
        self.fine_grid_sampling = fine_grid_sampling
        self.edge_margin = edge_margin

        # We will precompute fields along all the links for all frequencies to speed up the search process.
        kvalues,spectrum_values = spectrum

        # sort k values
        k_value_args = np.argsort(kvalues)
        kvalues = [kvalues[i] for i in k_value_args]
        spectrum_values = [spectrum_values[i] for i in k_value_args]

        self.kvalues = kvalues
        self.spectrum_values = spectrum_values

        # Real per-frequency weight. A complex source *field* carries power |S|^2;
        # a real spectrum is already a weight and is used unchanged.
        spec = np.asarray(spectrum_values)
        if np.iscomplexobj(spec) and np.any(spec.imag != 0):
            self.spectrum_weights = np.abs(spec) ** 2
        else:
            self.spectrum_weights = np.real(spec)

        # call the precompute function to calculate the fields along all the links for all frequencies
        self._precompute_values((kvalues, spectrum_values), self.healthy_network)


    def _precompute_values(self, 
                           spectrum : Tuple[List[float], List[complex]],
                           network: Network) -> None:
        """Precompute the values along all the links for all frequencies to speed up the search process.
            values are forward propogating amplitude A, backward propogating amplitude B for each 
            link at each frequency and the beta = n(k)*k values for each link at each frequency. 

        Args:
            spectrum: A tuple containing the list of frequencies and the corresponding spectrum for each frequency.
            network: The healthy network object, which contains the information about
              the network topology,link lengths, and fields

        Returns:
            np.ndarray: The precomputed values for each link at each frequency.
        """
        GNORM_EPSILON = 1e-30  # A small value to avoid division by zero in normalization
        kvalues, _ = spectrum
        num_frequencies = len(kvalues)
        num_links = len(network.internal_links)
        num_external_nodes = len(network.external_nodes)

        # Initialize arrays to store the precomputed values
        self.A_amplitudes = np.zeros((num_frequencies, num_links, num_external_nodes), dtype=complex)
        self.B_amplitudes = np.zeros((num_frequencies, num_links, num_external_nodes), dtype=complex)
        self.beta_values = np.zeros((num_frequencies, num_links), dtype=complex)
        # Physical link lengths [m]; needed to turn the fractional grid into physical
        # position x = frac*L, and to reference the backward wave to node A (e^{jβL}).
        self.link_lengths = np.array([network.internal_links[li].length for li in range(num_links)])

        for ik, k in enumerate(kvalues):
            for port in range(num_external_nodes):
                # generate excitation filed
                excitation_field = np.zeros(num_external_nodes, dtype=complex)
                excitation_field[port] = 1.0  # Excite the port with a unit amplitude

                field_along_links = network.get_internal_link_fields(k, excitation_field)
                self.A_amplitudes[ik, :, port] = field_along_links[:, 0]  # Forward propagating amplitude
                self.B_amplitudes[ik, :, port] = field_along_links[:, 1]  # Backward propagating amplitude

        for l_index in range(num_links):
            link = network.internal_links[l_index]
            self.beta_values[:, l_index] = np.array([self._link_n(link, k)*k for k in kvalues])

        # Reference the backward wave to node A by absorbing e^{jβL} into B:
        #   g(x) = A·e^{jβx} + (B·e^{jβL})·e^{-jβx} = A·e^{jβx} + B·e^{jβ(L-x)}.
        self.B_amplitudes *= np.exp(1j * self.beta_values[:, :, np.newaxis]
                                    * self.link_lengths[np.newaxis, :, np.newaxis])

        # measure the fields along the coarse grid
        self.coarse_grid = np.linspace(self.edge_margin, 1 - self.edge_margin, self.coarse_grid_sampling)

        #G0 has one complex field vector per freqquency, link, coarse grid point, and port.
        # The shape of G0 is (num_frequencies, num_links, coarse_grid_sampling, num_external_nodes)
        self.G0 = self._calculate_g(self.coarse_grid)
        # Finding the norm of G0 for normalization.
        self.G0_norm = np.maximum(np.sum(np.abs(self.G0)**2, axis=-1), GNORM_EPSILON)

        return None  # The precomputed values are stored in the instance variables, so we don't need to return anything.
    

    def localise_fault(self, method: str = 'eigen',
                       sum:str='incoherent',
                       phase_anchoring: bool = False,
                       faulty_S_matrix_array: Optional[np.ndarray] = None) -> Tuple[int, float]:
        """Localise the fault in the network using the DORT algorithm.

        Args:
            method: eigen or best_column
                    eigen: uses the eigenvector corresponding to the largest eigenvalue to find the link and position of the fault.
                    best_column: uses the column of Delta S that has the largest norm to find the link and position of the fault.
            sum: 'coherent', 'incoherent', or 'both' — which position estimate(s) to return.
            
            phase_anchoring: if True, de-rotate the per-frequency signature phase in the
                    COHERENT sum against the incoherent position estimate x_hat (a blind
                    bootstrap: x_hat comes from the data). This lets the
                    raw-eigh phase cancel so the coherent sum builds. Default False.

            faulty_S_matrix_array: Optional faulty S-matrix array (n_freq, n_ext, n_ext) for THIS
                    measurement. If given it overrides the one passed to the constructor, letting a
                    single precomputed DORT object localise many different faults. Frequency order
                    must match the spectrum.

        Returns:
            Tuple[int, float]: The index of the link with the highest projection value and the position along that link.
            For sum='both' returns (predicted_link_index, x_frac_coherent, x_frac_incoherent). Regardless of `sum`,
            both estimates and the coarse-ID diagnostics are also stored as attributes: self.predicted_link_index,
            self.link_rank_order, self.link_scores, self.purity, self.predicted_x_frac_inc, self.predicted_x_frac_coh.
        """
        EPSILON = 1e-300  # A small value to avoid division by zero in normalization
        faulty_S_matrix_array = (faulty_S_matrix_array if faulty_S_matrix_array is not None
                                 else self.Faulty_S_matrix_array)
        if faulty_S_matrix_array is None:
            raise ValueError("No faulty S-matrix provided. Pass faulty_S_matrix_array to "
                             "localise_fault(...) or to the DORT constructor.")
        TRO = self._time_reversal_operator(self.healthy_S_matrix_array, faulty_S_matrix_array)
        eigenvalues, eigenvectors = np.linalg.eigh(TRO)   # returns eigenvalues in ascending order
        self.v_1 = eigenvectors[:, :, -1]  # The eigenvector corresponding to the largest eigenvalue
        self.lambda_1 = eigenvalues[:, -1]  # The largest eigenvalue

        # The second largest eigenvalue, clipped to be non-negative (numerical stability)
        self.lambda_2 = np.clip(eigenvalues[:, -2], a_min=0.0, a_max=None)

        # Purity measure to assess the quality of the eigenvector
        self.purity = np.where(self.lambda_1>EPSILON, 1.0 - np.sqrt(self.lambda_2/np.maximum(self.lambda_1, EPSILON)), 0.0)
        self.weight = self.spectrum_weights*self.purity  # w(k) = S(k)·purity(k)

        self.total_weight = float(np.sum(self.weight) + EPSILON)  # Total weight for normalization

        #______________________Coarse Link Identification____________________________________
        if method == "eigen":
            vector_to_use = self.v_1
        elif method == "best_column":
            column_norms = np.sum(np.abs(self.Delta_S)**2, axis=1)  # Norm of each column of Delta S
            best_column_index = np.argmax(column_norms, axis=1)  # Index of the column with the largest norm for each frequency
            vector_to_use = np.take_along_axis(self.Delta_S, best_column_index[:, np.newaxis, np.newaxis], axis=2)[:, :, 0]  # Extract the best column for each frequency

        norm_vector_to_use = np.linalg.norm(vector_to_use, axis=1)  # (n_freq,)

        projection = np.einsum('kp, klxp -> klx', vector_to_use.conj(), self.G0)  # Projection of the best column onto the Green's function
        projection_intensity = (np.abs(projection)**2) / self.G0_norm

        link_scores = np.max(np.einsum('k,klx -> lx', self.weight*norm_vector_to_use, projection_intensity), axis=-1)
        # Max projection intensity across coarse grid points for each link
        link_rank_order = np.argsort(link_scores)[::-1]  # Indices of links sorted by their scores in descending order
        predicted_link_index = int(link_rank_order[0])  # Index of the link with the highest score

        # Store coarse-ID diagnostics so callers (e.g. statistical sweeps) can read the full
        # ranking without recomputing anything.
        self.link_scores = link_scores
        self.link_rank_order = link_rank_order
        self.predicted_link_index = predicted_link_index


        #______________________Fine Position Identification____________________________________
        # Fine grid search around the predicted link to find the position of the fault
        fine_grid = np.linspace(self.edge_margin, 1 - self.edge_margin, self.fine_grid_sampling)  # fractional grid
        G_test = self._calculate_g(fine_grid, index=predicted_link_index)[:, 0]  # (n_freq, grid, ports)

        G_test_norm = np.sum(np.abs(G_test)**2, axis=-1)      # (n_freq, grid): squared norm over ports
        eps = 0.01*np.median(G_test_norm, axis=1)[:, None]    # floor standing-wave nulls, (n_freq, 1)
        G_norm = np.maximum(G_test_norm, eps)                 # (n_freq, grid)

        projection_fine = np.einsum('kp, kxp -> kx', vector_to_use.conj(), G_test)  # <sig, g(x,k)>, (n_freq, grid)

        # Incoherent summation - Sum_k w(k) |<sig, ĝ(x,k)>|^2  removes the phase.
        # first because its peak x_hat is the anchor for the optional coherent phase fix.
        projection_intensity_fine = np.sum(self.weight[:, None] * np.abs(projection_fine)**2 / G_norm, axis=0) / self.total_weight
        idx_inc = int(np.argmax(projection_intensity_fine))
        predicted_x_frac_inc = fine_grid[idx_inc]  # Position with the highest incoherent intensity

        # Coherent summation - | Sum_k w(k) <sig, ĝ(x,k)> |,  ĝ = g/||g||
        coh_proj = projection_fine
        if phase_anchoring:
            # Blind phase anchor: reference each frequency's phase to ĝ(x_hat,k) at the
            # incoherent estimate x_hat (NOT the true x_f). Rotating sig by α/|α| with
            # α = <ĝ(x_hat,k), sig(k)> makes <sig, ĝ(x_hat)> real-positive for every k,
            # so the raw-eigh phase cancels and the coherent terms add at x_hat.

            g_hat_xhat = G_test[:, idx_inc, :] / np.sqrt(G_norm[:, idx_inc])[:, None]  # (n_freq, ports)
            alpha = np.einsum('kp,kp->k', g_hat_xhat.conj(), vector_to_use)            # <ĝ(x_hat), sig>
            phase = np.where(np.abs(alpha) > 1e-30, alpha / np.abs(alpha), 1.0)
            coh_proj = phase[:, None] * projection_fine

        projection_amplitude = np.sum(self.weight[:, None] * coh_proj / np.sqrt(G_norm), axis=0) / self.total_weight
        projection_amplitude_fine = np.abs(projection_amplitude)
        predicted_x_frac_coh = fine_grid[int(np.argmax(projection_amplitude_fine))]  # Position with the highest coherent amplitude

        # Store both fine-grid estimates and their images so callers can read whichever they need
        # (and the raw images for diagnostics) regardless of which `sum` was requested.
        self.predicted_x_frac_inc = float(predicted_x_frac_inc)
        self.predicted_x_frac_coh = float(predicted_x_frac_coh)
        self.projection_intensity_fine = projection_intensity_fine
        self.projection_amplitude_fine = projection_amplitude_fine

        if sum == 'coherent':
            return predicted_link_index, predicted_x_frac_coh
        elif sum == 'incoherent':
            return predicted_link_index, predicted_x_frac_inc
        elif sum == 'both':
            return predicted_link_index, predicted_x_frac_coh, predicted_x_frac_inc
        else:
            raise ValueError("Invalid value for 'sum'. Choose from 'coherent', 'incoherent', or 'both'.")



    def _link_n(self, link, k0):
        # n could be a complex number too indicating the presence of loss in the link.
        # Only the real part sets the propagation phase beta = Re{n}*k used for the Green's basis.
        n = link.n(k0) if callable(getattr(link, 'n', None)) else getattr(link, 'n', None)
        if n is None:
            raise AttributeError('link has no attribute n')
        return float(np.real(n))
    
    def _time_reversal_operator(self, 
                                S_healthy_array: np.ndarray, 
                                S_after_array: np.ndarray) -> np.ndarray:
        """Construct the time reversal operator from the measured signals of different frequencies.
        
        Args:
            S_healthy_array: A 3D numpy array of shape (n_frequencies, n_ext, n_ext) containing the scattering matrices of the healthy network.
            S_after_array: A 3D numpy array of shape (n_frequencies, n_ext, n_ext) containing the scattering matrices of the faulty network.

        Returns:
            np.ndarray: The time reversal operator T(k) = Delta S Delta S^dagger is a Hermitian matrix
        """
        self.Delta_S = S_after_array - S_healthy_array

        # Time reversal operator T(k) = Delta S Delta S^dagger for each frequency
        # use conjugate transpose on the last two axes for each frequency
        T = np.matmul(self.Delta_S, self.Delta_S.conj().transpose(0, 2, 1))  # (n_freq, n_ports, n_ports)
        return T
    
    def _calculate_g(self,
                     grid_points: np.ndarray,
                     index: Optional[int] = None) -> np.ndarray:
        """Calculate the Green's function for each link at the given grid points.

        Args:
            grid_points: A 1D array of points along the link where the Green's function is evaluated.
            index: The index of the link for which to calculate the Green's function.
                    If None, the Green's function is calculated for all links.
        Returns:
            np.ndarray: The Green's function evaluated at the given grid points for each link and frequency
        """
        # Ensure grid_points is a 1D array of fractional positions in [0, 1]
        gp = np.asarray(grid_points)
        if gp.ndim != 1:
            gp = gp.ravel()

        # Select links if index is provided
        if index is None:
            beta = self.beta_values[:, :, np.newaxis]           # (k, links, 1)
            L = self.link_lengths[np.newaxis, :, np.newaxis]     # (1, links, 1)
            A = self.A_amplitudes                                # (k, links, ports)
            B = self.B_amplitudes                                # (k, links, ports), node-A referenced
        else:
            beta = self.beta_values[:, index:index+1, np.newaxis]  # (k, 1, 1)
            L = self.link_lengths[index]                           # scalar
            A = self.A_amplitudes[:, index:index+1, :]             # (k, 1, ports)
            B = self.B_amplitudes[:, index:index+1, :]

        # Fractional grid -> physical position x = frac * L, then the node-A phases.
        x_phys = gp[np.newaxis, np.newaxis, :] * L                 # (., links, grid)
        phase_A = np.exp(1j * beta * x_phys)                       # (k, links, grid)
        phase_B = np.exp(-1j * beta * x_phys)

        # G[k, link, x, port] = A·e^{jβx} + B·e^{-jβx}  (B already carries e^{jβL}).
        G = (A[:, :, np.newaxis, :] * phase_A[:, :, :, np.newaxis]
             + B[:, :, np.newaxis, :] * phase_B[:, :, :, np.newaxis])
        return G