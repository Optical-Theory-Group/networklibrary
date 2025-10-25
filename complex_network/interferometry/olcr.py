import numpy as np
import matplotlib.pyplot as plt
from complex_network.networks.network import Network # type: ignore
from scipy.signal import hilbert, savgol_filter
from collections import deque
from complex_network.components.link import Link # type: ignore
from typing import Tuple
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os
from tqdm import tqdm
from scipy.signal import czt

class OLCR:
    def __init__(self,
                 network: Network,
                 input_node: int,
                 measurement_node: int,
                 central_wavelength: float,
                 bandwidth: float,
                 num_wavelength_sample: int,
                 optical_path_length: Tuple[float, float],
                 num_optical_path_length_sample: int,
                 integeration_method: str = 'czt',
                 width_factor: int = 3,
                 use_multi_proc: bool = False,
                 use_gpu: bool = False,
                 make_cache: bool = False,
                 num_workers: int = None):
        
        self.network = network
        self.input_node = input_node
        self.measurement_node = measurement_node
        self.central_lambda = central_wavelength
        self.bandwidth = bandwidth
        self.num_lambda = num_wavelength_sample
        self.opl_start = optical_path_length[0]
        self.opl_end = optical_path_length[1]
        self.num_opl = num_optical_path_length_sample
        self.width_factor = width_factor
        self.use_mp = use_multi_proc
        self.num_workers = num_workers if num_workers is not None else cpu_count()
        self.use_gpu = use_gpu
        self.integeration_method = integeration_method.lower()
        self.make_cache = make_cache

        # precalculate the values that dont depend on the main loop
        self.opls = np.linspace(self.opl_start, self.opl_end, self.num_opl)
        self.num_external_nodes = len(self.network.external_nodes)
        self.input_signal = np.zeros(self.num_external_nodes, dtype=np.complex128)
        self.input_signal[self.input_node] = 1.0 / np.sqrt(2)

        # Calculating the interferogram is an expensive operation,
        # so we will only do it once and store the result
        self._interferogram = None

        # Calculate the wavenumber range
        # Compute sigma_lambda from FWHM (Bandwidth)
        self.sigma_lambda = self.bandwidth / (2 * np.sqrt(2 * np.log(2)))
        
        # Convert center and sigma into k-space
        self.k0 = 2 * np.pi / self.central_lambda
        # dk / d lambda = -2pi / lambda^2  ==  
        # sigma_k = |dk/d lambda| * sigma_lambda  = (2pi / lambda_0^2) * sigma_lambda
        self.sigma_k = (2 * np.pi / self.central_lambda**2) * self.sigma_lambda
        
        # Define the range of k values
        self.k_min = self.k0 - self.width_factor * self.sigma_k
        self.k_max = self.k0 + self.width_factor * self.sigma_k
        self.k = np.linspace(self.k_min, self.k_max, self.num_lambda)

        # Constant spacing
        self.delta_k = (self.k_max - self.k_min) / (self.num_lambda - 1)
        self.delta_l = (self.opl_end - self.opl_start) / (self.num_opl - 1)

        # Calculate the spatial resolution of the interferogram
        self.spatial_resolution = 0.4412712003053032*self.central_lambda**2/self.bandwidth

        # Validate integration method
        if self.integeration_method not in ['euler', 'czt']:
            raise ValueError(f"Invalid integration method: {self.integration_method}. Choose 'euler' or 'czt'.")

    def generate_broadband_source(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a low-coherence broadband source with a Gaussian distribution
        centered in wavenumber space (k = 2pi/lambda) around k0, with a given bandwidth.
        Returns:
            k : array of sampled wavenumbers
            spectrum : normalized Gaussian spectrum in k-space
        """
        
        # Build the Gaussian spectrum i k-space
        spectrum = np.exp(-0.5 * ((self.k - self.k0) / self.sigma_k) ** 2)
        
        # Normalize the spectrum
        spectrum /= np.trapezoid(spectrum, self.k)
        
        return self.k, spectrum

    """ The interferogram calculation is mainly an integral given by:
        I(l) = ∫ G(k) |E_sample(k) + E_reference(k,l)|^2 dk
        where G(k) is the source spectrum, E_sample(k) is the field at the measurement node.

        This integral can be further simplified to:
        I(l) = C0 + 2 Re{ ∫ G(k) E_sample(k) * E_reference^*(k,l) dk }
        where C0 is a constant background term that does not depend on l.
    """

    def _compute_interferogram_euler_serial(self) -> np.ndarray:
        """Performs the calculation of the interferogram where the Integral over k is done using Euler method
        """

        self.interferogram = np.zeros_like(self.opls, dtype=np.float64)
        # We will launch a beam-splitted light through the input node
        self.input_signal[self.input_node] = 1.0 / np.sqrt(2) # TODO : update for arbitrary splitting ratio
        k, gaussian_frequency = self.generate_broadband_source()

        for k_val, gaussian_freq in tqdm(zip(k, gaussian_frequency), total=len(k), desc="Computing interferogram"):
            # Propagate the reference beam to the detector (air propagation)
            E_reference = np.exp(1j * k_val * self.opls) / np.sqrt(2)
            # Propogate the sample signal through the network and get the output field
            E_sample = self.network.get_S_ee(k_val) @ self.input_signal
            # Signal at the measurement node
            E_measurement = E_sample[self.measurement_node]
            # Add the incoherent intensities (low-coherence assumption )
            self.interferogram += gaussian_freq * np.abs(E_measurement + E_reference) ** 2 * self.delta_k

        return self.interferogram

    def _compute_interferogram_euler_parallel(self)-> np.ndarray:
        """Performs the calculation of the interferogram
        by propagating the reference beam and the sample signal through the network."""

        self.interferogram = np.zeros_like(self.opls, dtype=np.float64)
        # We will launch a beam-splitted light through the input node
        self.input_signal[self.input_node] = 1.0 / np.sqrt(2) # TODO : update for arbitrary splitting ratio
        k, gaussian_frequency = self.generate_broadband_source()

        print("Calculating interferogram (Euler Method) parallelly")
        _env_init()  # Initialize environment variables for multiprocessing
        num_chunks = self.num_workers * 2  # Number of chunks to process in parallel

        idx_chunks = np.array_split(np.arange(len(k)), num_chunks)
        chunks = [(k[idx], gaussian_frequency[idx]) for idx in idx_chunks]
        with ProcessPoolExecutor(max_workers=self.num_workers, 
                                    initializer=_worker_init,
                                    initargs=(self.network, self.input_signal, self.measurement_node, self.opls)) as executor:
            for partial in executor.map(_chunk_worker, chunks):
                self.interferogram += partial
        
        return self.interferogram
    
    def _compute_interferogram_euler_gpu(self) -> np.ndarray:
        """Uses both Multiprocessing for loading the matrices and GPU for accelerating the matrix operations."""
        # Placeholder for GPU implementation for bigger networks
        pass
    
    def _compute_interferogram_ft_serial(self) -> np.ndarray:
        """ Compute the interferogram using Fourier Transform method.
            This method uses the CZT algorithm to perform the inverse transform from k-space to l-space.
            The CZT allows flexible selection of both the frequency (or wavenumber) span and the grid spacing
        """
        E_Sample = np.zeros_like(self.k, dtype=np.complex128)
        for idx, k_value in enumerate(tqdm(self.k, desc="Computing Signal")):
            S = self.network.get_S_ee(k_value)
            E_Sample[idx] = (S @ self.input_signal)[self.measurement_node]

        a_r = 1.0 / np.sqrt(2)  # Reference arm amplitude (50/50 beam splitter) TODO: generalize for arbitrary splitting ratio
        # generate the broadband source
        _, gaussian_spectrum = self.generate_broadband_source()

        # The quantity we want to transform (This is only term that varies with l)
        # G(k) = gaussian_spectrum(k) * a_r^* * E_sample
        G = gaussian_spectrum * np.conj(a_r) * E_Sample

        # The CZT parameters ( A, W, M ) are defined as
        W = np.exp(-1j * self.delta_k * self.delta_l)
        A = np.exp(1j * self.opl_start * self.delta_k)

        # Perform the CZT to get the interferogram
        raw_g_czt = czt(G, self.num_opl, W, A)

        # Constant background term that does not depend on l
        C0 = np.sum(gaussian_spectrum * (np.abs(E_Sample)**2 + np.abs(a_r)**2)) * self.delta_k

        # Apply the required phase and scaling
        g_czt = np.exp(-1j * self.k_min * self.opls) * raw_g_czt
        self.interferogram = C0 + 2 * np.real(g_czt*self.delta_k)

        return self.interferogram
    
    def _compute_interferogram_ft_parallel(self) -> np.ndarray:

        """ Compute the interferogram using Fourier Transform method on multiple processors."""

        self.interferogram = np.zeros_like(self.opls, dtype=np.float64)
        # We will launch a beam-splitted light through the input node
        self.input_signal[self.input_node] = 1.0 / np.sqrt(2)  # TODO : update for arbitrary splitting ratio
        k, gaussian_frequency = self.generate_broadband_source()
        print("Calculating interferogram (FT Method) parallelly")
        _env_init()  # Initialize environment variables for multiprocessing

        num_chunks = self.num_workers * 2  # Number of chunks to process in parallel
        idx_chunks = np.array_split(np.arange(len(k)), num_chunks)
        chunks = [(k[idx], gaussian_frequency[idx]) for idx in idx_chunks]
        with ProcessPoolExecutor(max_workers=self.num_workers, 
                                    initializer=_worker_init,
                                    initargs=(self.network, self.input_signal, self.measurement_node, self.opls)) as executor:
            E_sample_chunks = list(executor.map(_chunk_worker_ft, chunks))

        # Combine the results from all chunks
        E_sample = np.concatenate(E_sample_chunks)
        a_r = 1.0 / np.sqrt(2)  # Reference arm amplitude (50/50 beam splitter) TODO: generalize for arbitrary splitting ratio
        # generate the broadband source
        _, gaussian_spectrum = self.generate_broadband_source()
        # The quantity we want to transform (This is only term that varies with l)
        # G(k) = gaussian_spectrum(k) * a_r^* * E_sample
        G = gaussian_spectrum * np.conj(a_r) * E_sample
        # The CZT parameters ( A, W, M ) are defined as
        W = np.exp(-1j * self.delta_k * self.delta_l)
        A = np.exp(1j * self.opl_start * self.delta_k)
        # Perform the CZT to get the interferogram
        raw_g_czt = czt(G, self.num_opl, W, A)
        # Constant background term that does not depend on l
        C0 = np.sum(gaussian_spectrum * (np.abs(E_sample)**2 + np.abs(a_r)**2)) * self.delta_k
        # Apply the required phase and scaling
        g_czt = np.exp(-1j * self.k_min * self.opls) * raw_g_czt
        self.interferogram = C0 + 2 * np.real(g_czt*self.delta_k)
        return self.interferogram

    def _compute_interferogram_ft_gpu(self) -> np.ndarray:
        # Placeholder for GPU implementation for bigger networks
        pass
    
    def _check_nyquist_criterion(self):
        """Check if the sampling satisfies the Nyquist criterion."""
        delta_opl = (self.opl_end - self.opl_start) / (self.num_opl - 1)
        nyquist_limit = np.pi / self.k_max
        
        # If the delta OPL is larger than the Nyquist limit, we warn the user
        if delta_opl > nyquist_limit:
            print(f"Warning: Spatial sampling is below the Nyquist limit")
            print(f"Current delta(OPL): {delta_opl:.2e} m vs Nyquist limit: {nyquist_limit:.2e} m")
            print(f"Minimum required samples: {int((self.opl_end - self.opl_start) / nyquist_limit) + 1}")
            print("Make the sampling at least 2x denser to avoid aliasing or 4x denser for better results.")

    def _compute_interferogram(self) -> np.ndarray:
        """Compute the interferogram using either serial or parallel method based on the use_mp flag."""
        self._check_nyquist_criterion()
        if self.use_mp:
            return self._compute_interferogram_euler_parallel()
        else:
            return self._compute_interferogram_euler_serial()
  
    def get_interferogram(self)-> np.ndarray:
        """ Returns the interferogram, calculating it if it has not been done yet."""
        if self.integeration_method == 'euler':
            if self._interferogram is None:
                self._interferogram = self._compute_interferogram_euler_parallel() if self.use_mp else self._compute_interferogram_euler_serial()
        if self.integeration_method == 'czt':
            if self._interferogram is None:
                self._interferogram = self._compute_interferogram_ft_parallel() if self.use_mp else self._compute_interferogram_ft_serial()
        return self._interferogram
    
    def _compute_envelope(self,
                            smooth_envelope: bool = True,
                            window_size: int| str = "auto") -> np.ndarray:
        
        """ Calculate the envelope of the interferogram using Hilbert transform.
            The envelope is calculated by taking the absolute value of the Hilbert transform
            of the interferogram. The envelope can be smoothed using a uniform filter if required"""

        self.get_interferogram()
        # We need to the remove the background
        signal_actual = self.interferogram - np.mean(self.interferogram)
        # Calculate the envelope using Hilbert transform
        envelope = np.abs(hilbert(signal_actual))
        # Smooth the envelope if required (True by default)
        if smooth_envelope:
            if window_size == "auto":
                # We will use the spatial resolution to determine the window size
                window_size = int(np.ceil(self.spatial_resolution / (self.opls[1] - self.opls[0])))
                # Ensure the window size is odd
                if window_size % 2 == 0:
                    window_size += 1
                
                # Ensure the window size is at least 3
                # This is to avoid issues with the Savitzky-Golay filter 
                if window_size < 3:
                    print("Warning: Window size is too small, setting to 3")
                    window_size = 3
            # Apply Savitzky-Golay filter to smooth the envelope
            envelope = savgol_filter(envelope, window_length=window_size, polyorder=2)
        # restore the vertical offset so that the envelope sits on the signal
        """ This can make spotting difference harder, so we comment it out
             We will leave it because it is useful for some applications"""
        
        # envelope += np.mean(self.interferogram)
        return envelope
    
    def plot_interferogram(self, plot_signal: bool = True,
                           plot_envelope: bool = True,
                           smooth_envelope: bool = True,
                           signal_window: None | tuple[float, float] = None,
                           saveto: str = None) -> None:
        """ Plot the interferogram and its envelope.
            Parameters:
            - plot_signal: If True, plot the interferogram signal.
            - plot_envelope: If True, plot the envelope of the interferogram.
            - smooth_envelope: If True, smooth the envelope using a Savitzky-Golay filter.
            - signal_window: A tuple (min, max) to set the x-axis limits for the signal.
            - saveto: If provided, save the plot to this file path instead of showing it.
        """
        # Figure-size
        plt.figure(figsize=(10, 6))

        # Set the figure size and plot the interferogram
        plt.xlabel('Optical Path Difference (mu m)')
        plt.ylabel('Intensity (a.u.)')
        plt.title('Interferogram')

        interferogram = self.get_interferogram()
        if plot_signal and plot_envelope:
            plt.plot(self.opls * 1e6, self._compute_envelope(smooth_envelope)+np.mean(self.interferogram), label='Envelope', color='red')
            plt.plot(self.opls * 1e6, interferogram, label='Interferogram', color='blue')
        
        elif plot_signal:
            plt.plot(self.opls * 1e6, interferogram, label='Interferogram', color='blue')

        if plot_envelope:
            plt.plot(self.opls * 1e6, self._compute_envelope(smooth_envelope), label='Envelope', color='red',lw=2)

        if signal_window is not None:
            plt.xlim(signal_window)
        else:
            plt.xlim(self.opl_start*1e6, self.opl_end * 1e6)
        if not saveto:
            plt.tight_layout()
            # plt.show()
        else:
            plt.savefig(saveto, dpi=300)
            plt.close()

# Global variables for multiprocessing
_global_network = None
_global_input_signal =None
_global_measurement_node =None
_global_opls = None

def _worker_init(network:Network, input_signal: np.ndarray,
                 measurement_node: int, opls: np.ndarray)-> None:
    "Called once per worker to initialize global variables"

    global _global_network, _global_input_signal, _global_measurement_node, _global_opls
    _global_network = network
    _global_input_signal = input_signal
    _global_measurement_node = measurement_node
    _global_opls = opls

def _chunk_worker(args)-> np.ndarray:
    """Worker processes one chunk of (k, weight) pairs and returns one partial interferogram"""

    k_chunk, gaussian_freq_chunk = args
    delta_k = k_chunk[1] - k_chunk[0]
    partial = np.zeros_like(_global_opls, dtype=np.float64)

    for k_val, gaussian_freq in zip(k_chunk, gaussian_freq_chunk):
        # Propagate the reference beam to the detector (air propagation)
        E_reference = np.exp(1j * k_val * _global_opls) / np.sqrt(2)
        # Propogate the sample signal through the network and get the output field
        E_sample = _global_network.get_S_ee(k_val) @ _global_input_signal
        E_measurement = E_sample[_global_measurement_node]
        # Add the incoherent intensities (low-coherence assumption )
        partial += gaussian_freq * np.abs(E_measurement + E_reference) ** 2 * delta_k

    return partial

def _chunk_worker_ft(args) -> np.ndarray:
    """Worker processes one chunk of k values and returns partial E_sample array"""
    k_chunk, _ = args
    E_sample_chunk = np.zeros_like(k_chunk, dtype=np.complex128)

    for idx, k_value in enumerate(k_chunk):
        S = _global_network.get_S_ee(k_value)
        E_sample_chunk[idx] = (S @ _global_input_signal)[_global_measurement_node]

    return E_sample_chunk

def _env_init():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"