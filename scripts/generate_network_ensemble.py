#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 14 14:08:40 2022

@author: Matthew Foreman

Generate ensembles of networks and store for later analysis
"""
import numpy as np
import os
from complexnetworklibrary.ensemble import NetworkEnsemble

# %% Setup physical parameters
wavelength = 1050e-9
k = (2 * np.pi) / wavelength
n = 1

case = 'voronoi slab'

# network specification
if case == 'delaunay':
    # delaunay slab
    network_type = 'delaunay'
    network_spec = {'internal_nodes': 350,
                    'exit_nodes': 50,
                    'network_size': (100e-6, 30e-6),
                    'exit_size': 110e-6,
                    'left_exit_fraction': 0.5,
                    'wavenumber': k,
                    'refractive_index': n,
                    'shape': 'slab'}
elif case == 'buffon slab':
    # buffon slab
    network_type = 'buffon'
    network_spec = {'lines': 30,
                    'network_size': (100e-6, 30e-6),
                    'wavenumber': k,
                    'refractive_index': n,
                    'fully_connected': True,
                    'shape': 'slab',
                    }
elif case == 'voronoi slab':
    # voronoi slab
    network_type = 'voronoi'
    network_spec = {'seed_nodes': 100,
                    'exit_nodes': 50,
                    'network_size': (100e-6, 30e-6),
                    'exit_size': 110e-6,
                    'left_exit_fraction': 0.5,
                    'wavenumber': k,
                    'refractive_index': n,
                    'shape': 'slab'}

# scales = [0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
# for length_scal_factor in scales:

# numlines = [25,50,75]
# for nl in numlines:

numseeds = [10]  # [50,100,150,200,250]
for ns in numseeds:
    length_scal_factor = 1
    # delanauy scaling
    # network_spec.update({
    #              'network_size':  (length_scal_factor*100e-6,30e-6),
    #              'internal_nodes': int(length_scal_factor*350),
    #              'exit_size': length_scal_factor*100e-6 + 10e-6,
    #              })

    # # buffon
    # network_spec.update({
    #                 'lines': nl
    #                 })

    # voronoi
    network_spec.update({
        'seed_nodes': ns
    })

    # node specification
    node_spec = {'Smat_type': 'COE',
                 'scat_loss': 0,
                 'subunitary_factor': 1
                 # 'delta':2*np.pi*np.random.rand(40),
                 }

    # ensemble generation
    # runid = 'delaunay_slab_{}_i{}_e{}_dim{}_su{}_lf{}'.format(node_spec['Smat_type'],
    #                                                  network_spec['internal_nodes'],
    #                                                  network_spec['exit_nodes'],
    #                                                  np.array(network_spec['network_size'])*1e6,
    #                                                  node_spec['subunitary_factor'],
    #                                                  length_scal_factor)
    runid = 'debug'

    # runid = 'buffon_slab_{}_e{}_dim{}_su{}_lf{}'.format(node_spec['Smat_type'],
    #                                                  int(network_spec['lines']*2),
    #                                                  np.array(network_spec['network_size'])*1e6,
    #                                                  node_spec['subunitary_factor'],
    #                                                  length_scal_factor)

    # runid = 'voronoi_slab_{}_ns{}_dim{}_su{}_lf{}'.format(
    #     node_spec['Smat_type'],
    #     network_spec['seed_nodes'],
    #     np.array(network_spec['network_size']) * 1e6,
    #     node_spec['subunitary_factor'],
    #     length_scal_factor)

    runid = runid.replace(' ', '')
    nrealisations = 200#10000
    nbatchsize = 50

    # # fudge since parallel hdf5 doesnt seem to work on my Windows
    # if os.name == 'nt':
    #     nbatchsize = 1

    nbatch = int(nrealisations / nbatchsize)

    initial_seed = 0
    savefilename = 'data/{}/p{}.h5'.format(runid, runid)

    # make data folders if they don't exist
    datadir = 'data/{}'.format(runid)
    outputdir = 'output/{}'.format(runid)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

    # %% Generate and save the ensemble
    # writemode = 'full'
    # writemode = 'sm'
    writemode = 'network'
    generator = NetworkEnsemble(savefilename,
                                writemode,
                                network_type,
                                network_spec, node_spec,
                                nbatch, nbatchsize, initial_seed)
