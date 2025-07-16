import numpy as np
import matplotlib.pyplot as plt
from complex_network.networks.network import Network
from scipy.signal import hilbert, savgol_filter
from collections import deque
from complex_network.components.link import Link
from typing import Tuple
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
import os
from tqdm import tqdm

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
                 width_factor: int = 3,
                 use_multi_proc: bool = False,
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

        # precalculate the values that dont depend on the main loop
        self.opls = np.linspace(self.opl_start, self.opl_end, self.num_opl)
        self.num_external_nodes = len(self.network.external_nodes)
        self.input_signal = np.zeros(self.num_external_nodes, dtype=np.complex128)
        self.input_signal[self.input_node] = 1.0 / np.sqrt(2)

        # Calculating the interferogram is an expensive operation,
        # so we will only do it once and store the result
        self._interferogram = None

    def generate_broadband_source(self)-> Tuple[np.ndarray, np.ndarray]:
        """ Generate an low-coherence broadband source with a Gaussian distribution
            centered at the central wavelength with a given bandwidth.
            The source is defined in the frequency space, and the bandwidth is defined"""
        
        # Gaussian standard deviation in wavelength
        self.sigma_lambda = self.bandwidth / (2 * np.sqrt(2 * np.log(2)))
        self.k_min = 2 * np.pi / (self.central_lambda + self.width_factor * self.sigma_lambda)
        self.k_max = 2 * np.pi / (self.central_lambda - self.width_factor * self.sigma_lambda)
        self.k = np.linspace(self.k_min, self.k_max, self.num_lambda)
        self.lambda_array = 2 * np.pi / self.k
        # Gaussian distribution for wavelength
        self.gaussian_wavelength = np.exp(-0.5 * ((self.lambda_array - self.central_lambda)
                                                  / (self.width_factor * self.sigma_lambda)) ** 2)
        # Convert the gaussian to frequency space using Jacobian |d lambda/ dk| = 2.pi / k^2
        self.gaussian_frequency = self.gaussian_wavelength * (2 * np.pi / self.k ** 2)
        # Normalize the gaussian
        self.gaussian_frequency /= np.sum(self.gaussian_frequency)

        return self.k, self.gaussian_frequency

    def _compute_interferogram(self)-> np.ndarray:
        """Performs the calculation of the interferogram
        by propagating the reference beam and the sample signal through the network."""

        self.interferogram = np.zeros_like(self.opls, dtype=np.float64)
        # We will launch a beam-splitted light through the input node
        self.input_signal[self.input_node] = 1.0 / np.sqrt(2)
        k, gaussian_frequency = self.generate_broadband_source()

        # Check for Nyquist criterion
        delta_opl = (self.opl_end - self.opl_start)/(self.num_opl - 1)
        nyquist_limit = np.pi/self.k_max
        # If the delta OPL is larger than the Nyquist limit, we warn the user
        if delta_opl > nyquist_limit:
            print(f"Warning: Spatial sampling is below the Nyquist limit")
            print(f"Current delta(OPL): {delta_opl:.2e} m vs Nyquist limit: {nyquist_limit:.2e} m")
            print(f"Minimum required samples: {int((self.opl_end - self.opl_start) / nyquist_limit) + 1}")
            print("Make the sampling atleast 2x denser to avoid aliasing or 4x denser for better results.")

        # Multiprocessing support
        if self.use_mp:
            print("Calculating interferogram parallelly")
            _env_init()  # Initialize environment variables for multiprocessing
            num_chunks = self.num_workers * 2  # Number of chunks to process in parallel

            idx_chunks = np.array_split(np.arange(len(k)), num_chunks)
            chunks = [(k[idx], gaussian_frequency[idx]) for idx in idx_chunks]
            with ProcessPoolExecutor(max_workers=self.num_workers, 
                                     initializer=_worker_init,
                                     initargs=(self.network, self.input_signal, self.measurement_node, self.opls)) as executor:
                for partial in executor.map(_chunk_worker, chunks):
                    self.interferogram += partial
        else:
            for k_val, gaussian_freq in tqdm(zip(k, gaussian_frequency), total=len(k), desc="Computing interferogram Sequentially"):
                # Propagate the reference beam to the detector (air propagation)
                E_reference = np.exp(1j * k_val * self.opls) / np.sqrt(2)
                # Propogate the sample signal through the network and get the output field
                E_sample = self.network.get_S_ee(k_val) @ self.input_signal
                # Signal at the measurement node
                E_measurement = E_sample[self.measurement_node]
                # Add the incoherent intensities (low-coherence assumption )
                self.interferogram += gaussian_freq * np.abs(E_measurement + E_reference) ** 2

        return self.interferogram
  
    def get_interferogram(self)-> np.ndarray:
        """ Returns the interferogram, calculating it if it has not been done yet."""
        if self._interferogram is None:
            self._interferogram = self._compute_interferogram()
        return self._interferogram
    
    def _compute_envelope(self,
                            smooth_envelope: bool = True,
                            window_size: int = 9):
        
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
            envelope = savgol_filter(envelope, window_length=window_size,polyorder=2)
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
            plt.plot(self.opls * 1e6, self._compute_envelope(smooth_envelope), label='Envelope', color='red')

        if signal_window is not None:
            plt.xlim(signal_window)
        else:
            plt.xlim(0, self.max_opl * 1e6)
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
    partial = np.zeros_like(_global_opls, dtype=np.float64)

    for k_val, gaussian_freq in zip(k_chunk, gaussian_freq_chunk):
        # Propagate the reference beam to the detector (air propagation)
        E_reference = np.exp(1j * k_val * _global_opls) / np.sqrt(2)
        # Propogate the sample signal through the network and get the output field
        E_sample = _global_network.get_S_ee(k_val) @ _global_input_signal
        E_measurement = E_sample[_global_measurement_node]
        # Add the incoherent intensities (low-coherence assumption )
        partial += gaussian_freq * np.abs(E_measurement + E_reference) ** 2

    return partial

def _env_init():
    os.environ["OMP_NUM_THREADS"] = "1"  # Disable OpenMP in workers to avoid conflicts
    os.environ["MKL_NUM_THREADS"] = "1"  # Disable MKL in workers to avoid conflicts
    os.environ["OPT_NUM_THREADS"] = "1"  # Disable OPT in workers to avoid conflicts