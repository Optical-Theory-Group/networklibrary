# Importing the necessary libraries
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_factory import generate_network
# Importing the scattering ensemble function
from complex_network.scattering_ensemble import generate_scattering_ensemble

# define the network parameters using the network spec data class
network_type = 'delaunay'
network_shape = 'slab'
random_seed = 0
num_seed_nodes = None
network_size = (200e-6,200e-6)
external_size = 250e-6
external_offset = 0.0

if __name__ == '__main__':
    for j in [10]:
        num_external_nodes = j
        for i in [20]:
            num_internal_nodes = i
            network_spec = NetworkSpec(network_type=network_type,
                                    network_shape=network_shape,
                                    random_seed=random_seed,
                                    num_internal_nodes=num_internal_nodes,
                                    num_external_nodes=num_external_nodes,
                                    num_seed_nodes=num_seed_nodes,
                                    network_size=network_size,
                                    external_size=external_size,
                                    external_offset=external_offset)
            # We can generate the ensemble of scatetring matrices and write them to a hdf5 file using the scattering ensemble function
            # hdf5_filename = f'/home/baruva/scattering_data/{network_type[0:3]}_{network_shape[0:4]}/scattering_ensemble_{network_type[0:3]}_{network_shape[0:4]}_{num_internal_nodes}_{num_external_nodes}.h5'
            hdf5_filename = f'scattering_ensemble_{network_type[0:3]}_{network_shape[0:4]}_{num_internal_nodes}_{num_external_nodes}1.h5'
            # name of the hdf5 file to write the scattering matrices to
            total_tasks = 1000                        # number of scattering matrices to generate
            k0 = 1.0                                   # wavenumber of the incident field
            matrix_type = 'full'                         # type of scattering matrix to generate (ee, ei or full)
            network_config = network_spec              # network configuration to use
            num_workers = 60                           # number of workers to use for parallel processing (None uses all available cores)
            generate_scattering_ensemble(hdf5_filename=hdf5_filename,
                                        total_tasks=total_tasks,
                                        k0=k0,
                                        matrix_type=matrix_type,
                                        network_config=network_config,
                                        num_workers=num_workers,
                                        compression_opts=1)