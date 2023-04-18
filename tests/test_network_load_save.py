# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:25:19 2022

Test network load/save functions

"""

import numpy as np
import matplotlib.pyplot as plt
import os

# Import the Network class from the complexnetworklibrary.network module
from complexnetworklibrary.network import Network

# Define some parameters to create a network object
wavelength = 500e-9
k = (2 * np.pi) / wavelength
n = 1
seed = 3

# Define the type and specifications for the network to be created
network_type = 'delaunay'
network_spec = {'internal_nodes': 30,
                'exit_nodes': 10,
                'network_size': 100e-6,
                'exit_size': 110e-6,
                'wavenumber': k,
                'refractive_index': n,
                'shape': 'circular'}

# Define some specifications for the nodes in the network
node_spec = {'Smat_type': 'isotropic_unitary',
             'scat_loss': 0}

# Create a new network object with the specified parameters and specifications
network = Network(network_type,
                  network_spec,
                  node_spec,
                  seed_number=seed)

# Define a filepath to save the network object to
filepath = '../data/test_network.h5'

# Get the directory name from the filepath
datadir = os.path.dirname(filepath)

# Check if the directory exists, and create it if it doesn't
if not os.path.exists(datadir):
    os.makedirs(datadir)

# Save the network object to the filepath
network.save_network(filepath)

print('-------------------------------------------------------')
print('-------------------------------------------------------')

# Create a new network object by loading the saved file
newnetwork = Network(filename=filepath)

# Draw the original and loaded network object for a visual comparison
network.draw()
newnetwork.draw()
