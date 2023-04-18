# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:25:19 2022

@author: mforeman

Test network load/save functions

"""
import numpy as np
import matplotlib.pyplot as plt
import os

from complexnetworklibrary.network import Network

wavelength = 500e-9
k = (2 * np.pi) / wavelength
n = 1
seed = 3

# delaunay circular
network_type = 'delaunay'
network_spec = {'internal_nodes': 30,
                'exit_nodes': 10,
                'network_size': 100e-6,
                'exit_size': 110e-6,
                'wavenumber': k,
                'refractive_index': n,
                'shape': 'circular'}

node_spec = {'Smat_type': 'isotropic_unitary',
             'scat_loss': 0}

network = Network(network_type,
                  network_spec,
                  node_spec,
                  seed_number=seed)  # CREATE A NETWORK OBJECT

network.draw()
filepath = '../data/test_network.h5'
datadir = os.path.dirname(filepath)
# make data folder if they dont exist
if not os.path.exists(datadir):
    os.makedirs(datadir)

network.save_network(filepath)

print('-------------------------------------------------------')
print('-------------------------------------------------------')
print('-------------------------------------------------------')
print('-------------------------------------------------------')
newnetwork = Network(filename=filepath)
newnetwork.draw()
