# Structure of the HDF5 File

The HDF5 file created by the scattering_ensemble.py module serves as a structured container for storing scattering matrices and their associated metadata including network properties. These attributes are extracted from the `NetworkSpec` object and stored in JSON format to ensure compatibility with HDF5. Non-serializable objects are removed before storage.


## File Attributes
The root level of the HDF5 file contains attributes that store metadata about the network and material properties used in generating the scattering matrices. These attributes include:
- `external_offset`: Offset parameter for external nodes.
- `external_size`: Size parameter for external nodes.
- `fully_connected`: Specifies whether the network is fully connected.
- `k0` : Complex wavenumber.
- `material_B`: Coefficients related to material properties.
- `material_C`: Additional material coefficients.
- `material_default_wave_param`: Default wave parameter used in computations.
- `material_material`: Type of material used (e.g., "glass").
- `network_shape`: The geometric shape of the network (e.g., "circular").
- `network_size`: The overall size of the network.
- `network_type`: Network topology (e.g., "delaunay").
- `node_S_mat_params`: JSON-encoded parameters for node scattering matrices.
- `node_S_mat_type`: Type of scattering matrix model used (e.g., "COE").
- `num_external_nodes`: Number of external nodes in the network.
- `num_internal_nodes`: Number of internal nodes in the network.
- `num_seed_nodes`: Number of seed nodes used in the network.

## Data Organization
The HDF5 file organizes the scattering matrices within specific groups.

### Scattering Matrix Groups (`S_<matrix_type>`)
Each type of scattering matrix (e.g., `S_ee`, `S_ie`) is stored in a dedicated group named `S_<matrix_type>`, where `<matrix_type>` refers to the type of matrix stored. `ee` - scattering from external nodes to external nodes, `ie` - scattering from internal nodes to external nodes.
Each group contains the following datasets:

1. **`matrix_shapes` Dataset**
   - Stores the shape of each scattering matrix as they could be different.
   - Data type: Structured NumPy dtype `('idx', 'i4'), ('rows', 'i4'), ('cols', 'i4')`
   - Purpose: Tracks the dimensions of matrices for efficient retrieval.

2. **`completed` Dataset**
   - Boolean array indicating whether a scattering matrix has been successfully stored.
   - Purpose: Avoids redundant computations by tracking completed tasks.

3. **Scattering Matrices (`matrix_<idx>`)**
   - Each computed scattering matrix is stored as a separate dataset named `matrix_<idx>`. `<idx>` refers to the random seed used for file generation.
   - Data type: NumPy array containing complex values.
   - Compression: Gzip with compression level 1.
   - Purpose: Stores the computed scattering matrices for each network configuration.
