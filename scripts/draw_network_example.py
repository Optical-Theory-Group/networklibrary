# -*- coding: utf-8 -*-
"""
Created on Mon 18 Oct 2021

@author: Matthew Foreman

Draw an example of different networks and save them to the output folder

"""

import numpy as np
import os
import matplotlib.pyplot as plt

from complexnetworklibrary.network import Network

wavelength = 1050e-9
k = (2 * np.pi) / wavelength
n = 1

SM_type = 'isotropic_unitary'
seed = np.random.randint(2 ** 10)
print(seed)
qualifier = ''

seed = 2 + 4408
output_root = '../output/network_examples/'
# Check if the directory exists, and create it if it doesn't
if not os.path.exists(output_root):
    os.makedirs(output_root)


cases = ['delaunay', 'delaunay slab', 'voronoi', 'voronoi slab', 'buffon', 'buffon slab']

for case in cases:
    print('Generating {} type network'.format(case))
    if case == 'delaunay':
        # delaunay circular
        network_type = 'delaunay'
        network_spec = {'internal_nodes': 60,
                        'exit_nodes': 20,
                        'network_size': 100e-6,
                        'exit_size': 110e-6,
                        'wavenumber': k,
                        'refractive_index': n,
                        'shape': 'circular'}
    elif case == 'delaunay slab':
        # delaunay slab
        network_type = 'delaunay'
        network_spec = {'internal_nodes': 100,
                        'exit_nodes': 10,
                        'network_size': (100e-6, 20e-6),
                        'exit_size': 110e-6,
                        'left_exit_fraction': 0.5,
                        'wavenumber': k,
                        'refractive_index': n,
                        'shape': 'slab'}
    elif case == 'voronoi':
        # voronoi circular
        network_type = 'voronoi'
        network_spec = {'seed_nodes': 100,
                        'network_size': 100e-6,
                        'exit_size': 110e-6,
                        'wavenumber': k,
                        'refractive_index': n,
                        'shape': 'circular'}
    elif case == 'voronoi slab':
        # voronoi slab
        network_type = 'voronoi'
        network_spec = {'seed_nodes': 30,
                        'network_size': (100e-6, 20e-6),
                        'left_exit_fraction': 0.5,
                        'exit_nodes': 20,
                        'exit_size': 110e-6,
                        'wavenumber': k,
                        'refractive_index': n,
                        'curvature_factor': 10,
                        'shape': 'slab'}
    elif case == 'buffon':
        # buffon
        network_type = 'buffon'
        network_spec = {'lines': 20,
                        'network_size': 100e-6,
                        'wavenumber': k,
                        'refractive_index': n,
                        'fully_connected': True,
                        'shape': 'circular',
                        }
    elif case == 'buffon slab':
        # buffon slab
        network_type = 'buffon'
        network_spec = {'lines': 20,
                        'network_size': (100e-6, 20e-6),
                        'wavenumber': k,
                        'refractive_index': n,
                        'fully_connected': True,
                        'shape': 'slab',
                        }
    else:
        raise ValueError

    node_spec = {'scat_mat_type': SM_type,
                 'scat_loss': 0,
                 'subunitary_factor': 1
                 }

    network = Network(network_type,
                      network_spec,
                      node_spec,
                      seed_number=seed)  # CREATE A NETWORK OBJECT

    network.draw('')

    if 'exit_size' in network_spec.keys():
        exit_size = 1.1 * network_spec['exit_size']
    else:
        exit_size = 1.1 * (network_spec['network_size'][0] if isinstance(network_spec['network_size'], tuple) \
                               else network_spec['network_size'])

    plt.savefig(os.path.join(output_root, "network_{}{}_example.png".format(case, qualifier)), dpi=300)
    plt.savefig(os.path.join(output_root, "network_{}{}_example.pdf".format(case, qualifier)), dpi=300)
