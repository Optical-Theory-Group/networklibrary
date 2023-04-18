# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 08:42:57 2023

@author: mforeman
"""

import numpy as np
import os
import dill
from copy import deepcopy
from complexnetworklibrary.util import update_progress

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
                    'network_size': (100e-6),
                    'exit_size': 110e-6,
                    'wavenumber': k,
                    'refractive_index': n,
                    'shape':'circular'
                    }
    
    node_number = 9 # which node do we peturb?
    lmin = 700e-9
    lmax = 701e-9
    Qmin = 250# 25000
    kimax = 0*k/Qmin
    kimin = -6000#k/Qmin
 
# node specification

scattering_loss = 0
node_spec = {'Smat_type': 'unitary_cyclic',
             'scat_loss': scattering_loss,
             # leave 'delta' out to get random phases across all nodes
             }
length_scal_factor = 1

# #################################################
# %%parameters for data files
# #################################################

runid1 = 'perturb_node_{}_{}_i{}_e{}_dim{}_lf{}'.format(network_type,
                                        node_spec['Smat_type'],
                                        network_spec['internal_nodes'],
                                        network_spec['exit_nodes'],
                                        np.array(network_spec['network_size'])*1e6,
                                        length_scal_factor)

runid2 = 'perturb_node_{}_{}_i{}_e{}_dim{}_lf{}_v2'.format(network_type,
                                        node_spec['Smat_type'],
                                        network_spec['internal_nodes'],
                                        network_spec['exit_nodes'],
                                        np.array(network_spec['network_size'])*1e6,
                                        length_scal_factor)

runid3 = 'perturb_node_{}_{}_i{}_e{}_dim{}_lf{}_merge'.format(network_type,
                                        node_spec['Smat_type'],
                                        network_spec['internal_nodes'],
                                        network_spec['exit_nodes'],
                                        np.array(network_spec['network_size'])*1e6,
                                        length_scal_factor)

# make data folders if they dont exist
for runid in [runid1,runid2,runid3]:
    datadir = 'data/{}'.format(runid)
    outputdir = 'output/{}'.format(runid)
    if not os.path.exists(datadir):
        os.makedirs(datadir)
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)

# #################################################
# %% Loop over loading data from run 1 and 2, 
#    copying fine peaks from run 2 to run 1,
#    and then saving as a run 3 for safety
# #################################################
indmin = 0
indmax = 150



# %%
for index in range(indmin,indmax):
    update_progress((index-indmin) / (indmax-indmin), '{}/{}   '.format((index-indmin),(indmax-indmin)))
    
    filename1 = 'data/{}/{}_index_results_{}.pkl'.format(runid1,runid1,index)
    filename2 = 'data/{}/{}_index_results_{}.pkl'.format(runid2,runid1,index)
    filename3 = 'data/{}/{}_index_results_{}.pkl'.format(runid3,runid3,index)
    
    with open(filename1, 'rb') as fpkl1:
        detS_results = dill.load(fpkl1)
        
    with open(filename2, 'rb') as fpkl2:
        kpeak_results = dill.load(fpkl2)
        
    results = deepcopy(detS_results)
    results['kpeaks_fine'] = kpeak_results['kpeaks_fine']
    results["dkanal_startpoint"] = kpeak_results["kpeaks_fine"]
    with open('data/{}/{}_index_results_{}.pkl'.format(runid3,runid3,index), 'wb') as fpkl3:
        dill.dump(results,fpkl3)
        # print(results['kpeaks_fine'])
                
        