"""It is noted that this code running on linux systems may not ensure the conditions passed in the init_worker function.
   This can cause the code to run slower than expected due to processing overhead. You can manually set the threads using the following commands:
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   export OPENBLAS_NUM_THREADS=1
   export NUMBA_NUM_THREADS=1
   export VECLIB_MAXIMUM_THREADS=1
   
   or run the set_threads.sh script in the terminal to set the threads to 1."""


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
from typing import Callable, Tuple, Union
# import h5py_blosc

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

def init_worker():
    """Ensure each worker process limits internal threading."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

def compute_scattering_matrix(args: Tuple[int, complex, str, NetworkSpec]) -> Tuple[int, np.ndarray]:
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
	        'rt': network.get_RT_matrix
        }
        
        if matrix_type not in matrix_funcs:
            raise ValueError(f"Invalid matrix type: {matrix_type}")
            
        return idx, matrix_funcs[matrix_type](k0)
    
    except Exception as e:
        logging.error(f"Error in task {idx}: {e}", exc_info=True)
        return idx, None

def multiproc_worker(
    hdf5_filename: str, 
    total_tasks: int,
    k0: complex,
    matrix_type: str,
    network_config: NetworkSpec,
    num_workers: int = None,
    compression_opts: int = 4
) -> None:
    """Generate ensemble of scattering matrices using multiprocessing.
    Parameters:
        hdf5_filename: Path to the HDF5 file where the ensemble will be stored
        total_tasks: Number of scattering matrices to generate
        k0: Complex wavenumber
        matrix_type: Type of matrix to compute ('ee', 'ie')
        network_config: NetworkSpec object containing network parameters
        num_workers: Number of worker processes to use (default: half of available CPUs or 1 if only 1 CPU)
        
        returns: None (results are stored in the HDF5 file)"""
    num_workers = num_workers or max(1, mp.cpu_count()-2)

    # Initialize or open HDF5 file
    with h5py.File(hdf5_filename, 'a') as h5file:
        # create the metadata of the ensemble that defines the network properties 
        # using the _set_attributes helper function
        _set_attributes(h5file,network_config,k0)

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
        
        logging.info(
    f"Generating {len(tasks)} matrices. {len(existing_indices)} already exist. \n"
    f"ne:{network_config.num_external_nodes} ni:{network_config.num_internal_nodes} "
    f"network_type:{network_config.network_type} network_shape:{network_config.network_shape} "
    f"matrix_type:{matrix_type}")
        
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
                            compression_opts=compression_opts
                        )
                        
                        # Mark as completed
                        completed[idx] = True
                        pbar.update(1)

    logging.info("Ensemble generation complete.")

# Function to generate the ensemble of scattering matrices
def generate_scattering_ensemble(
    hdf5_filename: str,
    total_tasks: int,
    k0: complex,
    matrix_type: str,
    network_config: NetworkSpec,
    num_workers: int = None,
    compression_opts: int = 4
    ) -> None:
    """Generate ensemble of scattering matrices using multiprocessing."""
    if matrix_type not in ('ee', 'ie','rt' ,'full'):
        raise ValueError(f"Invalid matrix type: {matrix_type}")
    # define a helper function that sets the metadata to the hdf5 file
    if matrix_type == 'full':
        multiproc_worker(hdf5_filename, total_tasks, k0, 'ee', network_config, num_workers,compression_opts)
        multiproc_worker(hdf5_filename, total_tasks, k0, 'ie', network_config, num_workers,compression_opts)
    elif matrix_type in ('ee', 'ie','rt'):
        multiproc_worker(hdf5_filename, total_tasks, k0, matrix_type, network_config, num_workers,compression_opts)
    else:
        raise ValueError(f"Invalid matrix type: {matrix_type}")
    
    return None

#_______________________________________________________________________________________________________________________
# We will write function that computes some value from the scattering matrices and stores the results in the HDF5 file

def compute_value_worker(args: Tuple[int, complex, str, NetworkSpec]):
    """Worker function to compute values from scattering matrices."""
    idx, k0, quantity_name, network_config = args
    try:
        # Re-seed the network configuration
        network_config.random_seed = idx
        
        # Generate the network and compute matrix
        network = generate_network(spec=network_config)

        network_shape = network_config.network_shape
        if network_shape == 'slab':
            matrix = network.get_RT_matrix(k0)
        elif network_shape == 'circular':
            matrix = network.get_S_ee(k0)
        else:
            raise ValueError(f"Invalid network shape: {network_shape}")

        """Compute the Inverse Participation Ratio (IPR) of the scattering matrix."""
        n = matrix.shape[0]

        if network_shape == 'slab':
            if quantity_name == 'transmission_IPR':
                t = matrix[np.ix_(range(n//2,n), range(n//2))]
                prob = np.conj(t).T@t
            elif quantity_name == 'reflection_IPR':
                r = matrix[np.ix_(range(n//2), range(n//2))]
                prob = np.conj(r).T@ r
            elif quantity_name == 'full_IPR':
                prob = np.conj(matrix).T@matrix 
            else:
                raise ValueError(f"Invalid quantity name: {quantity_name}")
            
        elif network_shape == 'circular':
            quantity_name = 'full_IPR'
            prob = np.conj(matrix).T@matrix

        eigenvalues, eigenvectors = np.linalg.eig(prob)
        ipr = np.sum(np.abs(eigenvectors) ** 4, axis=0)/np.sum(np.abs(eigenvectors) ** 2, axis=0)

        
        return idx, eigenvalues,ipr
    except Exception as e:
        logging.error(f"Error computing value for seed {idx}: {e}", exc_info=True)
        return idx, None, None

#@TODO: Make it so that whenever it finds an error and stops, it will not stop the whole process but continue with the next task
def compute_ipr_ensemble(
    hdf5_filename: str,
    total_tasks: int,
    k0: complex,
    network_config: NetworkSpec,
    quantity_name: str,
    compression_opts: int = 4,
    num_workers: int = None
) -> None:
    """Optimized version with multiprocessing improvements."""
    
    with h5py.File(hdf5_filename, 'a') as h5file:
        _set_attributes(h5file, network_config, k0)
        ipr_group = h5file.require_group(f'{quantity_name}')
        eigenvalues_group = h5file.require_group(f'eigenvalues')
        
        # Initialize tracking datasets
        if 'completed' not in ipr_group:
            ipr_group.create_dataset(
                'completed', 
                (total_tasks,), 
                dtype=bool, 
                data=np.zeros(total_tasks, dtype=bool)
            )
        completed = ipr_group['completed']
        
        # Identify remaining tasks
        existing_indices = np.where(completed[:])[0]
        tasks = [(i, k0, quantity_name, network_config) 
                for i in range(total_tasks) if i not in existing_indices]
        
        logging.info(f"Computing {len(tasks)}/{total_tasks} {quantity_name} values")

        # Configure parallel processing
        num_workers = num_workers or max(1, mp.cpu_count()-2)
        chunksize = max(1, len(tasks) // (num_workers * 5))
        
        with mp.Pool(num_workers, initializer=init_worker) as pool:
            with tqdm(total=len(tasks), desc=f"Processing {quantity_name}") as pbar:
                for idx, ipr, eigenvalues in pool.imap_unordered(
                    compute_value_worker, tasks, chunksize=chunksize
                ):
                    if ipr is not None:
                        # Store results
                        dataset_name = f'value_{idx}'
                        if dataset_name in ipr_group:
                            del ipr_group[dataset_name]
                            
                        if np.isscalar(ipr):
                            ipr_group.create_dataset(dataset_name, data=ipr)
                        else:
                            ipr_group.create_dataset(
                                dataset_name,
                                data=ipr,
                                compression='gzip',
                                compression_opts=compression_opts
                            )
                        # Update completion status
                        completed[idx] = True

                    if eigenvalues is not None:
                        dataset_name = f'eigenvalues_{idx}'
                        if dataset_name in eigenvalues_group:
                            del eigenvalues_group[dataset_name]
                            
                        if np.isscalar(eigenvalues):
                            eigenvalues_group.create_dataset(dataset_name, data=eigenvalues)
                        else:
                            eigenvalues_group.create_dataset(
                                dataset_name,
                                data=eigenvalues,
                                compression='gzip',
                                compression_opts=compression_opts
                            )
                        # Update completion status
                        completed[idx] = True
                    pbar.update(1)

    logging.info(f"Stored {quantity_name} results in {hdf5_filename}")


#___________________Helper Function_______________________
def _set_attributes(h5file,network_config,k0):
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
          
    h5file.attrs['k0'] = k0

    return None
