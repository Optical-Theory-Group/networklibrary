"""Use generate_scattering_ensemble to generate an ensemble of scattering matrices for a given network configuration."""
import multiprocessing as mp
import h5py
import numpy as np
import time
import logging
from typing import Tuple
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_factory import generate_network
from tqdm import tqdm
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')


# Worker function to compute scattering matrix for a given network.
def compute_scattering_matrix(args: Tuple[int, float, str, NetworkSpec]) -> Tuple[int, np.ndarray]:
    """
    Compute the scattering matrix for a given network.
    
    Parameters:
        args (tuple): Contains (idx, k0, network_arguments) where:
                      - idx (int): Used to set the random seed.
                      - k0 (complex): Wavenumber for the scattering matrix (passed to get_S_ee).
                      - matrix_type (str): Type of scattering matrix ('ee'-external-external, 'ei'-external-internal, 'full'-full scattering matrix).
                      - network_arguments: NetworkSpec object that defines the network.
    
    Returns:
        tuple: (idx, matrix) where 'matrix' is the computed scattering matrix.
    """
    try:
        idx, k0, matrix_type, network_arguments = args
        network_arguments.random_seed = idx  # Override the seed with idx.

        network = generate_network(spec=network_arguments)

        matrix_funcs = {
            'ee': network.get_S_ee,
            'ie': network.get_S_ie,
            'full': network.get_network_matrix
        }

        if matrix_type not in matrix_funcs:
            raise ValueError(f"Invalid matrix type: {matrix_type}")

        return idx, matrix_funcs[matrix_type](k0)
    
    except Exception as e:
        logging.error(f"Error in task {idx}: {e}")
        return idx, None

def generate_scattering_ensemble(
    hdf5_filename: str, 
    total_tasks: int,
    k0: complex,
    matrix_type: str,
    network_config: NetworkSpec,
    num_workers: int = None
) -> None:
    """
    Generate ensemble of scattering matrices using multiprocessing.
        -hdf5_filename (str) : Path to the HDF5 file
        -total_tasks (int) : Number of scattering matrices to generate
        -k0 (complex) : Wavenumber for the scattering matrix
        -matrix_type (str) : Type of scattering matrix ('ee', 'ei', 'full')
        -network_config (NetworkSpec) : Network configuration
        -num_workers (int) : Number of worker processes (default: number of CPU cores)

    """
    num_workers = num_workers or mp.cpu_count()

    # We want to see if the file already exists, and give the option to wheather delete it or overwrite it
    if os.path.exists(hdf5_filename):
        while True:
            logging.warning(f"File {hdf5_filename} already exists. Do you want to delete it? or write on it?")
            user_input = input("Type 'd' to delete the file, or 'w' to write on it: ")
            if user_input == 'd':
                os.remove(hdf5_filename)
                logging.info(f"File {hdf5_filename} has been deleted.")
                break
            elif user_input == 'w':
                logging.info(f"Writing on file {hdf5_filename}.")
                break
            else:
                logging.error("Invalid input. Please type 'd' or 'w'.")

   
    tasks = [(i, k0, matrix_type, network_config) for i in range(total_tasks)]

    with h5py.File(hdf5_filename, 'a') as h5file:
        scattering_group = h5file.require_group('S_'+matrix_type)
        
        # Pre-check to avoid regenerating existing matrices
        existing_matrices = set(scattering_group.keys())
        tasks = [task for task in tasks if f'matrix_idx={task[0]}' not in existing_matrices]

        logging.info(f"Generating {len(tasks)} scattering matrices. {len(existing_matrices)} already exist in the file.")

        with mp.Pool(num_workers) as pool:
            with tqdm(total=len(tasks), desc="Writing Scattering Matrices to HDF5 file") as pbar:
                for idx, matrix in pool.imap_unordered(compute_scattering_matrix, tasks):
                    if matrix is not None:
                        dataset_name = f'matrix_idx={idx}'
                        scattering_group.create_dataset(dataset_name, data=matrix, compression='gzip', compression_opts=4)
                        pbar.update(1)

    logging.info("Scattering matrix ensemble generation complete.")