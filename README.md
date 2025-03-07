# complex_network: Python library for complex, optical networks.

This library provides a Python platform for generating and analysing random photonic networks.

Please refer to the `examples` directory for demonstrations of common use cases.

## 📂 Library structure
```
complex_network/                     
├── components/                      
│   ├── component.py                 # Base class for network components
│   ├── link.py                      # Waveguides that connect nodes
│   └── node.py                      # Junction points in the network
├── materials/                       
│   ├── dielectric.py                # Dielectric material properties
│   ├── material.py                  # Base class for optical material properties
│   └── refractive_index.py          # Refractive index dispersion
├── networks/                        
│   ├── network_factory.py           # Network generation methods
│   ├── network_perturbator.py       # Network perturbation methods
│   ├── network_spec.py              # Pre-generation network parameters
│   ├── network.py                   # Core network class
│   └── pole_calculator.py           # Methods for analysing scattering resonances
├── scattering_matrices/             
│   ├── link_matrix.py               # Methods for link scattering matrices
│   └── node_matrix.py               # Methods for node scattering matrices
├── ensemble.py                      
├── Scattering_ensemble.py           # Generating ensemble of scattering matrices stored in HDF5 format                
└── utils.py                         
```

## 🔧 Package versions
See `environment.yml` for a list of required packages.
