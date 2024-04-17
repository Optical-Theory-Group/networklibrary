# -*- coding: utf-8 -*-
"""
Created on Thu Feb  2 16:08:02 2023

@author: mforeman
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:31:03 2023

@author: mforeman

Generates a single random network and calculates the |S| complex spectrum 
as a specified node is perturbed

"""
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FFMpegFileWriter
from matplotlib.collections import LineCollection

from complex_network.networks.network import Network
from complex_network.util import update_progress

# #################################################
# %% define simulation parameters
# #################################################
wavelength = 500e-9
k=(2*np.pi)/wavelength
n=1
seed = 65565

case ='delaunay'
if case == 'delaunay':
    # delaunay slab
    network_type = 'delaunay'
    network_spec = {'internal_nodes': 80,
                    'exit_nodes': 30,
                    # 'network_size': (100e-6),
                    # 'exit_size': 110e-6,
                    'wavenumber': k,
                    'refractive_index': n,
                    'shape':'circular'
                    }
    
    node_number = 9 # which node do we peturb?
    lmin = 700e-9
    lmax = 701e-9
    Qmin = 250# 25000
    kimax = 0*k/Qmin
    kimin = 0#-7000#k/Qmin
 
# node specification

scattering_loss = 0
node_spec = {'S_mat_type': 'unitary_cyclic',
             'scat_loss': scattering_loss,
             # leave 'delta' out to get random phases across all nodes
             }

# #################################################
# %%parameters for data files
# #################################################
size0 = 100e-6
runid = 'scale_network_{}_{}_i{}_e{}_dim{}'.format(network_type,
                                        node_spec['S_mat_type'],
                                        network_spec['internal_nodes'],
                                        network_spec['exit_nodes'],
                                        size0*1e6)
outputfiles = True

# make data folders if they dont exist
datadir = 'data/{}'.format(runid)
outputdir = 'output/{}'.format(runid)
if not os.path.exists(datadir):
    os.makedirs(datadir)
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

# #################################################
# %% Construct coordinates in complex plane
# #################################################
kmin = (2*np.pi/lmax) + 1j*kimin
kmax = (2*np.pi/lmin) + 1j*kimax
Nkr = 1500
Nki = 1#150

krmin = np.real(kmin)
kimin = np.imag(kmin)

krmax = np.real(kmax)
kimax = np.imag(kmax)

# #####################################################################
# %% Set up properties for tuning properties of a single node
# #####################################################################
scale_factors = np.logspace(0,-2,50) # scalefactors to scan over

# Initialise arrays to store results
scal_scan_results = {}
detS = np.zeros( (Nkr, Nki, len(scale_factors)),dtype=np.float64)

# #################################################
# %% Tune node and calculate detS landscape and
#    numerically find poles
# #################################################
indmin = 0
indmax = 1#len(scale_factors)

for index in range(indmin,indmax):
    update_progress((index-indmin) / (indmax-indmin), '{}/{}'.format((index-indmin),(indmax-indmin)))
    length_scal_factor = 10**(scale_factors[index])
    
    # #################################################
    # %% Initialise the network
    # #################################################
    network_spec.update({
                  'network_size':  length_scal_factor*size0,
                  'exit_size': length_scal_factor*1.1*size0,
                  })
    network = Network(network_type,network_spec,node_spec,seed_number=seed)   #CREATE A NETWORK OBJECT

    input_amp = [0]* network_spec['exit_nodes']
    input_amp[0] = 1
    internal_nodes = network.internal_nodes
    external_nodes = network.exit_nodes
    network.initialise_network(node_spec,input_amp)   # input my scattering matrix and the input amplitudes

    results = {}
    
    results['parameter'] = scale_factors[index]
    results['index'] = index

    # # %% -- calculate determinant landscape
    kr,ki,detS_temp,kpeaks_temp = network.calc_det_S(kmin,kmax,Nkr,Nki,
                                                      progress_bar_text='{}/{}'.format((index-indmin),(indmax-indmin)))
    detS[:,:,index] = detS_temp
    results['detS'] = detS_temp

    # store result
    scal_scan_results[scale_factors[index]] = results
    with open('data/{}/{}_index_results_{}.pkl'.format(runid,runid,index), 'wb') as fpkl:
        dill.dump(results,fpkl)

    # %% Plot detS landscape
    fig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(2*np.pi/kr.T * 1e9,(detS_temp))
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel('|S|')
    plt.xlim(lmin * 1e9,lmax * 1e9)
    
    plt.subplot(2,1,2)
    plt.plot(2*np.pi/kr.T * 1e9,np.log10(detS_temp))
    
    plt.xlabel("Wavelength (nm)")
    plt.ylabel('log(|S|)')
    plt.xlim(lmin * 1e9,lmax * 1e9)
    
    plt.show()
    
    plt.savefig("output/{}/{}_int_{}_ext_{}_loss_{}_lf_{}.png".format(runid,index,internal_nodes,external_nodes,scattering_loss,10**length_scal_factor),dpi=300)
    # fig.clear()
    # plt.close()


