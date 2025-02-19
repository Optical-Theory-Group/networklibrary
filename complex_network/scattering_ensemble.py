import multiprocessing as mp
import h5py
import numpy as np
import logging
from typing import Tuple
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_factory import generate_network
from tqdm import tqdm
import os
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def init_worker():
    """Ensure each worker process limits internal threading."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

def compute_scattering_matrix(args: Tuple[int, float, str, NetworkSpec]) -> Tuple[int, np.ndarray]:
    """Worker function that computes the scattering matrix for a given seed value
    Parameters:
        args: Tuple containing the following elements:
            idx: Index of the task which is used as the random seed
            k0: complex Wavenumber
            matrix_type: Type of matrix to compute ('ee', 'ie', or 'full')
            network_arguments: NetworkSpec object containing network parameters
        Returns:
            Tuple containing the following elements:
            idx: Index of the task
            matrix: Scattering matrix (numpy array)"""
    
    try:
        idx, k0, matrix_type, network_arguments = args
        network_arguments.random_seed = idx
        
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
        logging.error(f"Error in task {idx}: {e}", exc_info=True)
        return idx, None

def generate_scattering_ensemble(
    hdf5_filename: str, 
    total_tasks: int,
    k0: complex,
    matrix_type: str,
    network_config: NetworkSpec,
    num_workers: int = None
) -> None:
    """Generate ensemble of scattering matrices using multiprocessing.
    Parameters:
        hdf5_filename: Path to the HDF5 file where the ensemble will be stored
        total_tasks: Number of scattering matrices to generate
        k0: Complex wavenumber
        matrix_type: Type of matrix to compute ('ee', 'ie', or 'full')
        network_config: NetworkSpec object containing network parameters
        num_workers: Number of worker processes to use (default: half of available CPUs or 1 if only 1 CPU)
        
        returns: None (results are stored in the HDF5 file)"""
    num_workers = num_workers or max(1, mp.cpu_count() // 2)

    # Initialize or open HDF5 file
    with h5py.File(hdf5_filename, 'a') as h5file:
        # create the metadata of the ensemble that defines the network properties 
        # using the _set_attributes helper function
        _set_attributes(h5file,network_config)

        # Create or get group for scattering matrices
        scattering_group = h5file.require_group(f'S_{matrix_type}')
        
        # Create datasets for matrix shapes and completion tracking
        if 'matrix_shapes' not in scattering_group:
            shape_dtype = np.dtype([('idx', 'i4'), ('rows', 'i4'), ('cols', 'i4')])
            scattering_group.create_dataset('matrix_shapes', (total_tasks,), dtype=shape_dtype)
            scattering_group.create_dataset('completed', (total_tasks,), dtype=bool, data=np.zeros(total_tasks, dtype=bool))
        
        completed = scattering_group['completed']
        matrix_shapes = scattering_group['matrix_shapes']
        
        # Determine which tasks need to be computed
        existing_indices = np.where(completed[:])[0]
        tasks = [(i, k0, matrix_type, network_config) for i in range(total_tasks) if i not in existing_indices]
        
        logging.info(f"Generating {len(tasks)} matrices. {len(existing_indices)} already exist.")
        
        # Process tasks
        with mp.Pool(num_workers, initializer=init_worker) as pool:
            chunksize = max(1, len(tasks) // (num_workers * 5))
            with tqdm(total=len(tasks), desc="Processing") as pbar:
                for idx, matrix in pool.imap_unordered(compute_scattering_matrix, tasks, chunksize=chunksize):
                    if matrix is not None:
                        # Store matrix shape
                        matrix_shapes[idx] = (idx, matrix.shape[0], matrix.shape[1])
                        
                        # Create or get dataset for this matrix
                        matrix_name = f'matrix_{idx}'
                        if matrix_name in scattering_group:
                            del scattering_group[matrix_name]  # Replace if exists
                        
                        # Store the matrix with compression
                        scattering_group.create_dataset(
                            matrix_name,
                            data=matrix,
                            compression='gzip',
                            compression_opts=1
                        )
                        
                        # Mark as completed
                        completed[idx] = True
                        pbar.update(1)

    logging.info("Ensemble generation complete.")

# define a helper function that sets the metadata to the hdf5 file
def _set_attributes(h5file,network_config):
    """saves the network configuation attribuets to the hdf5 file"""
    all_vars = network_config.__dict__.copy()
    # Pop the material object from the network_config as it is not json serializable
    material_obj = all_vars.pop('material', None)
    #pop the random seed as it is not needed in the metadata
    all_vars.pop('random_seed',None)

    #Pop the objects inside the material object that are not json serializable
    material_vars = material_obj.__dict__.copy()
    material_vars.pop('_n',None)
    material_vars.pop('_dn',None)
    for key, value in material_vars.items():
        h5file.attrs['material_'+key] = value

    for key, value in all_vars.items():
        h5file.attrs[key] = json.dumps(value)

    return None