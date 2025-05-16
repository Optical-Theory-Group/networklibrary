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
from typing import Callable, Tuple, Union, Any, Dict
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
            matrix_shapes = scattering_group.create_dataset(
                'matrix_shapes',
                shape=(total_tasks,),
                maxshape=(None,),
                chunks=True,
                dtype=shape_dtype
            )
            scattering_group.create_dataset(
                'completed',
                shape=(total_tasks,),
                maxshape=(None,),
                chunks=True,
                dtype=bool,
                data=np.zeros(total_tasks, dtype=bool)
            )
        else:
            matrix_shapes = scattering_group['matrix_shapes']
            completed = scattering_group['completed']
            # Optionally resize if total_tasks is increased.
            if matrix_shapes.shape[0] < total_tasks:
                matrix_shapes.resize((total_tasks,))
                completed.resize((total_tasks,))
        
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

def compute_ipr_worker(args: Tuple[int, complex, NetworkSpec]) -> Tuple[int, Dict[str, Tuple[Any, Any]]]:
    """
    Worker function that computes eigenvalues and IPR for all required quantities.
    
    For slab networks:
      - Computes the scattering matrix only once via network.get_RT_matrix(k0)
      - From that matrix, computes:
          • 'full': the full scattering matrix,
          • 'reflection': extracted from the top-left block,
          • 'transmission': extracted from the bottom-left block.
      - (Optionally, you can compute an RT matrix using the helper convert_to_RT function.)
    
    For circular networks:
      - Only the full scattering matrix is computed via network.get_S_ee(k0).
    
    Returns a tuple of the seed index and a dictionary mapping each quantity name to a tuple (eigenvalues, IPR).
    """
    idx, k0, network_config = args
    try:
        network_config.random_seed = idx
        network = generate_network(spec=network_config)
        network_shape = network_config.network_shape
        results = {}

        # Get the scattering matrix
        matrix = network.get_S_ee(k0)
        
        if network_shape == 'slab':
            # Convert the scattering matrix to an RT matrix
            matrix = convert_to_RT(matrix, network)
            n = matrix.shape[0]
            
            # Full quantity
            full_prob = matrix
            eigvals_full, eigvecs_full = np.linalg.eig(full_prob)
            ipr_full = np.sum(np.abs(eigvecs_full) ** 4, axis=0) / (np.sum(np.abs(eigvecs_full) ** 2, axis=0) ** 2)
            results['full'] = (eigvals_full, ipr_full)
            
            # Reflection quantity: use top-left block
            r = matrix[np.ix_(range(n//2), range(n//2))]
            prob_r = np.conj(r).T @ r
            eigvals_r, eigvecs_r = np.linalg.eig(prob_r)
            ipr_r = np.sum(np.abs(eigvecs_r) ** 4, axis=0) / (np.sum(np.abs(eigvecs_r) ** 2, axis=0) ** 2)
            results['reflection_intensity'] = (eigvals_r, ipr_r)

            r = matrix[np.ix_(range(n//2), range(n//2))]
            eigvals_r, eigvecs_r = np.linalg.eig(r)
            ipr_r = np.sum(np.abs(eigvecs_r) ** 4, axis=0) / (np.sum(np.abs(eigvecs_r) ** 2, axis=0) ** 2)
            results['reflection_amplitude'] = (eigvals_r, ipr_r)
            
            # Transmission quantity: use bottom-left block
            t = matrix[np.ix_(range(n//2, n), range(n//2))]
            prob_t = np.conj(t).T @ t
            eigvals_t, eigvecs_t = np.linalg.eig(prob_t)
            ipr_t = np.sum(np.abs(eigvecs_t) ** 4, axis=0) / (np.sum(np.abs(eigvecs_t) ** 2, axis=0) ** 2)
            results['transmission_intensity'] = (eigvals_t, ipr_t)

            t = matrix[np.ix_(range(n//2, n), range(n//2))]
            eigvals_t, eigvecs_t = np.linalg.eig(t)
            ipr_t = np.sum(np.abs(eigvecs_t) ** 4, axis=0) / (np.sum(np.abs(eigvecs_t) ** 2, axis=0) ** 2)
            results['transmission_amplitude'] = (eigvals_t, ipr_t)
            
        elif network_shape == 'circular':
            eigvals, eigvecs = np.linalg.eig(matrix)
            ipr = np.sum(np.abs(eigvecs) ** 4, axis=0) / (np.sum(np.abs(eigvecs) ** 2, axis=0) ** 2)
            results['full'] = (eigvals, ipr)
        else:
            raise ValueError(f"Invalid network shape: {network_shape}")
        
        return idx, results
    except Exception as e:
        logging.error(f"Error computing value for seed {idx}: {e}", exc_info=True)
        return idx, {}

#@TODO: Make it so that whenever it finds an error and stops, it will not stop the whole process but continue with the next task
def compute_ipr_ensemble(
    hdf5_filename: str,
    total_tasks: int,
    k0: complex,
    network_config: NetworkSpec,
    compression_opts: int = 4,
    num_workers: int = None
) -> None:
    """
    Compute and store eigenvalues and IPR for all seeds in a single pass.
    
    For slab networks, the following quantities are computed:
      • 'full'
      • 'reflection_intensity'
      • 'reflection_amplitude'
      • 'transmission_intensity'
      • 'transmission_amplitude'
    
    For circular networks, only 'full' is computed.
    
    Each computed quantity is stored in its own HDF5 group (one for eigenvalues and one for IPR).
    """
    with h5py.File(hdf5_filename, 'a') as h5file:
        _set_attributes(h5file, network_config, k0)
        network_shape = network_config.network_shape
        
        # Determine which quantities to compute.
        if network_shape == 'slab':
            quantity_names = ['full', 'reflection_intensity','reflection_amplitude', 'transmission_intensity','transmission_amplitude']
        elif network_shape == 'circular':
            quantity_names = ['full']
        else:
            raise ValueError(f"Invalid network shape: {network_shape}")
        
        # Prepare groups for each computed quantity.
        groups = {}
        for q in quantity_names:
            groups[q] = {
                'IPR': h5file.require_group(f'{q}_IPR'),
                'eigenvalues': h5file.require_group(f'{q}_eigenvalues')
            }
        
        # Create (or open) a completion dataset to track computed seeds.
        if 'completed' not in h5file:
            completed_ds = h5file.create_dataset(
                'completed',
                shape=(total_tasks,),
                maxshape=(None,),
                dtype=bool,
                chunks=True,  # Enable chunking
                data=np.zeros(total_tasks, dtype=bool)
            )
        else:
            completed_ds = h5file['completed']
            # Resize if necessary:
            if completed_ds.shape[0] < total_tasks:
                completed_ds.resize((total_tasks,))

        
        # Identify remaining tasks.
        completed_indices = np.where(completed_ds[:])[0]
        tasks = [(i, k0, network_config) for i in range(total_tasks) if i not in completed_indices]
        
        logging.info(f"Computing {len(tasks)}/{total_tasks} tasks")
        
        num_workers = num_workers or max(1, mp.cpu_count()-2)
        chunksize = max(1, len(tasks) // (num_workers * 50))
        
        with mp.Pool(num_workers, initializer=init_worker) as pool:
            with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                for idx, results in pool.imap_unordered(compute_ipr_worker, tasks, chunksize=chunksize):
                    if results:
                        for q, (eigvals, ipr) in results.items():
                            ds_ipr = groups[q]['IPR']
                            ds_eig = groups[q]['eigenvalues']
                            ds_name_ipr = f'value_{idx}'
                            ds_name_eig = f'eigenvalues_{idx}'
                            
                            # If the dataset already exists for this seed, remove it.
                            if ds_name_ipr in ds_ipr:
                                del ds_ipr[ds_name_ipr]
                            if ds_name_eig in ds_eig:
                                del ds_eig[ds_name_eig]
                            
                            # Store the IPR and eigenvalues with optional compression.
                            if np.isscalar(ipr):
                                ds_ipr.create_dataset(ds_name_ipr, data=ipr)
                            else:
                                ds_ipr.create_dataset(ds_name_ipr, data=ipr,
                                                      compression='gzip', compression_opts=compression_opts)
                            if np.isscalar(eigvals):
                                ds_eig.create_dataset(ds_name_eig, data=eigvals)
                            else:
                                ds_eig.create_dataset(ds_name_eig, data=eigvals,
                                                      compression='gzip', compression_opts=compression_opts)
                        completed_ds[idx] = True
                    pbar.update(1)
    logging.info(f"Stored results in {hdf5_filename}")

def internal_intensity_ipr_worker(args: Tuple[int, complex, NetworkSpec]) -> Tuple[int, Dict[str, Tuple[Any]]]:
    """
    Worker function that computes IPR for internal field distributions of eigen modes.
    Distribution of energy in the internal links of the network.
    Returns a tuple of the seed index and a dictionary mapping the quantity name to the IPR.
    """
    idx, k0, network_config = args
    try:
        network_config.random_seed = idx
        network = generate_network(spec=network_config)

        # Get the scattering matrix
        matrix = network.get_S_ee(k0)

        # Compute the eigenmodes
        _, vec = np.linalg.eig(matrix)
        
        Intensity_IPR = np.zeros(vec.shape[1])
        for i in range(vec.shape[1]):
            vector = vec[:, i]
            energy_denisty = network.get_all_link_energy_densities(k0,vector)

            # Compute the IPR of the internal field distribution
            ipr = np.sum(np.abs(energy_denisty) ** 4) / (np.sum(np.abs(energy_denisty) ** 2) ** 2)
            Intensity_IPR[i] = ipr

        # Select quantity name based on network shape
        if network_config.network_shape == 'slab':
            quantity = 'slab_full'
        elif network_config.network_shape == 'circular':
            quantity = 'circ_full'
        else:
            raise ValueError(f"Invalid network shape: {network_config.network_shape}")

        # Return in a dictionary to match the multiproc loop expectation.
        return idx, {quantity: (Intensity_IPR,)}
            
    except Exception as e:
        logging.error(f"Error computing value for seed {idx}: {e}", exc_info=True)
        return idx, None


def internal_intensity_ipr(
    hdf5_filename: str,
    total_tasks: int,
    k0: complex,
    network_config: NetworkSpec,
    compression_opts: int = 4,
    num_workers: int = None
) -> None:
    """
    Compute and store eigenvalues and IPR for all seeds in a single pass.
    
    For slab networks, the following quantities are computed:
      • 'slab_full'
    
    For circular networks, only 'circ_full' is computed.
    
    Each computed quantity is stored in its own HDF5 group (one for eigenvalues and one for IPR).
    """
    with h5py.File(hdf5_filename, 'a') as h5file:
        _set_attributes(h5file, network_config, k0)
        network_shape = network_config.network_shape
        
        # Determine which quantities to compute.
        if network_shape == 'slab':
            quantity_names = ['slab_full']
        elif network_shape == 'circular':
            quantity_names = ['circ_full']
        else:
            raise ValueError(f"Invalid network shape: {network_shape}")
        
        # Prepare groups for each computed quantity.
        groups = {}
        for q in quantity_names:
            groups[q] = {'IPR': h5file.require_group(f'{q}_internal_IPR')}
        
        # Create (or open) a completion dataset to track computed seeds.
        if 'completed' not in h5file:
            completed_ds = h5file.create_dataset(
                'completed',
                shape=(total_tasks,),
                maxshape=(None,),
                dtype=bool,
                chunks=True,  # Enable chunking
                data=np.zeros(total_tasks, dtype=bool)
            )
        else:
            completed_ds = h5file['completed']
            # Resize if necessary:
            if completed_ds.shape[0] < total_tasks:
                completed_ds.resize((total_tasks,))
        
        # Identify remaining tasks.
        completed_indices = np.where(completed_ds[:])[0]
        tasks = [(i, k0, network_config) for i in range(total_tasks) if i not in completed_indices]
        
        logging.info(f"Computing {len(tasks)}/{total_tasks} tasks")
        
        num_workers = num_workers or max(1, mp.cpu_count() - 2)
        chunksize = max(1, len(tasks) // (num_workers * 50))
        
        with mp.Pool(num_workers, initializer=init_worker) as pool:
            with tqdm(total=len(tasks), desc="Processing tasks") as pbar:
                for idx, results in pool.imap_unordered(internal_intensity_ipr_worker, tasks, chunksize=chunksize):
                    if results:
                        # Iterate through the returned dictionary.
                        for q, (ipr,) in results.items():
                            ds_ipr = groups[q]['IPR']
                            ds_name_ipr = f'value_{idx}'
                            
                            # If the dataset already exists for this seed, remove it.
                            if ds_name_ipr in ds_ipr:
                                del ds_ipr[ds_name_ipr]
                            
                            # Store the IPR with optional compression.
                            if np.isscalar(ipr):
                                ds_ipr.create_dataset(ds_name_ipr, data=ipr)
                            else:
                                ds_ipr.create_dataset(ds_name_ipr, data=ipr,
                                                      compression='gzip', compression_opts=compression_opts)
                        completed_ds[idx] = True
                    pbar.update(1)
    logging.info(f"Stored results in {hdf5_filename}")
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

def convert_to_RT(matrix,network):
    """Calculate the reflection and transmission matrix of the scattering matrix which is valid for slab geometries
            Convert the scattering matrix to the slab scattering matrix.
            The slab scattering matrix is defined as the scattering matrix of a slab of material
            but the terms are organized as [[r,t']
                                            [t,r']]
            First column: Response to an incoming wave from the left r (reflection from the left) t (transmission from left to right).
            Second column: Response to an incoming wave from the right r (reflection from the right) t (transmission from right to left).

            The left nodes have +ve coordinates and the right nodes have -ve coordinates."""

    external_scattering_map = network.external_scattering_map
    port_to_node = {v: k for k, v in external_scattering_map.items()}
    
    left_ports = []
    right_ports = []

    # Sort the ports into left and right
    for port in port_to_node:
        node = network.get_node(port_to_node[port])
        if node.position[0] > 0:
            left_ports.append(port)
        else:
            right_ports.append(port)

    left = np.array(left_ports)
    right = np.array(right_ports)

    # Extract submatrices using ix_ to handle index arrays correctly
    r = matrix[np.ix_(left, left)] if left.size else np.empty((0, 0))
    t_prime = matrix[np.ix_(left, right)] if left.size and right.size else np.empty((left.size, right.size))
    t = matrix[np.ix_(right, left)] if right.size and left.size else np.empty((right.size, left.size))
    r_prime = matrix[np.ix_(right, right)] if right.size else np.empty((0, 0))
    
    # Construct the block matrix using numpy's block function
    block_matrix = np.block([[r, t_prime], [t, r_prime]])
    
    return block_matrix

# --- Function to calculate energy density stored in a fiber ---
def calculate_fiber_energy_density(inwave, outwave, length, k):
    """
    Calculate energy stored in fiber using:
    Energy = (r1^2 + r2^2) * l_i + 2*r1*r2/k * cos(k*length + theta1-theta2) * sin(k*l_i)
    
    where:
    - r1, theta1: amplitude and phase of inwave
    - r2, theta2: amplitude and phase of outwave
    - l_i: fiber length
    - k: wavenumber
    """
    # Extract amplitude and phase
    r1 = np.abs(inwave)
    r2 = np.abs(outwave)
    theta1 = np.angle(inwave)
    theta2 = np.angle(outwave)
    
    # Calculate energy components
    amplitude_term = (r1**2 + r2**2)
    interference_term = 2 * r1 * r2 * np.cos(k*length + theta1 - theta2) * np.sin(k * length)/k
    
    # Total energy
    energy_density = amplitude_term + interference_term/length
    return energy_density
