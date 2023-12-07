# -*- coding: utf-8 -*-
"""
Created on Mon Nov 21 15:31:22 2022

@author: mforeman

Performs statistical analysis of network ensemble as created using generate_network_ensemble.py
Plots accompanying graphs from preprocessed datafile.
"""

import os

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from complexnetworklibrary._dict_hdf5 import (
    load_dict_from_hdf5, recursively_load_dict_contents_from_group,
    save_dict_to_hdf5)
from complexnetworklibrary.network import Network
from complexnetworklibrary.util import update_progress

obj = os.scandir('../data/')
dirs = [entry.name for entry in obj if entry.is_dir()]

# %% Datafile selection / read or load selection

process_or_load = input('(P) Process datafile or (L) load preprocessed data? :')

if process_or_load.upper() == 'P':
    for i, entry in enumerate(dirs):
        print("[{}] - {} - {}".format(i, 'x' if os.path.exists('data/{}/proc_data_{}.h5'.format(entry, entry)) else ' ',
                                      entry))

    inputstr = (input("Please enter index (or comma separated indexes) of dataset(s) above to analyse:"))
    strlist = inputstr.split(',')
    dataindices = [int(s) for s in strlist]

    for dataindex in dataindices:
        if dataindex not in range(0, len(dirs)):
            raise ValueError('Incorrect value given')

        runid = dirs[dataindex]
        datafile = 'data/{}/p{}.h5'.format(runid, runid)

        # make output/data folder if it doesnt exist
        datadir = 'data/{}'.format(runid)
        outputdir = 'output/{}'.format(runid)
        if not os.path.exists(datadir):
            os.makedirs(datadir)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        print('----------------------------------------------')
        print('--------- Processing {}.h5 dataset ----------'.format(dirs[dataindex]))
        print('----------------------------------------------')

        with h5py.File(datafile, 'r') as h5file:
            realisations = [key for key in h5file.keys()]

            nreals = len(realisations)
            print('Total {} realisations'.format(nreals))

            # get first data realisation and extract network/node specifications
            data0keys = [key for key in h5file[realisations[0]]['NETWORK']]
            if 'network_spec' in data0keys:
                ns_path = '{}/NETWORK/network_spec'.format(realisations[0])
                network_spec = recursively_load_dict_contents_from_group(h5file, ns_path)

                print('----------------------------------------------')
                print('Network specification is:')
                print(network_spec)
            if 'node_spec' in data0keys:
                ns_path = '{}/NETWORK/node_spec'.format(realisations[0])
                node_spec = recursively_load_dict_contents_from_group(h5file, ns_path)

                print('----------------------------------------------')
                print('Node specification is:')
                print(node_spec)
            if 'network_type' in data0keys:
                nt_path = '{}/NETWORK/network_type'.format(realisations[0])
                network_type = h5file[nt_path][()]

            # initialise data arrays
            all_eigs = []
            all_dets = []
            all_unitary = []
            all_reciprocal = []
            all_timerev = []
            all_teigLR = []
            all_teigRL = []
            all_reigLL = []
            all_reigRR = []
            all_int_nodes = []
            all_ext_nodes = []

            for ii, rid in enumerate(realisations):
                update_progress(ii / nreals, 'Analysing SM {}/{}'.format(ii, nreals))

                # get scattering matrix and network properties
                nw_path = '{}/NETWORK/'.format(rid)
                nw = recursively_load_dict_contents_from_group(h5file, nw_path)
                sm = nw['scattering_matrix']
                smnodes = nw['sm_node_order']

                exitids = nw['exit_ids']  # same elemnts as smnodes but potentially different order
                exitpos = nw['exit_positions']

                # sort positions of exit nodes to match smnodes
                smeinds = [list(exitids).index(eid) for eid in smnodes]
                smpos = exitpos[smeinds]

                leftids = [eid for ii, eid in enumerate(smnodes) if smpos[ii][0] < 0]
                rightids = [eid for ii, eid in enumerate(smnodes) if smpos[ii][0] >= 0]
                leftindex = [ii for ii, eid in enumerate(smnodes) if smpos[ii][0] < 0]
                rightindex = [ii for ii, eid in enumerate(smnodes) if smpos[ii][0] >= 0]

                # extract t,t',r,r' blocks (form matches Rotter review)
                # S = [ r  t' ]
                #     [ t  r' ]
                [LI1, LI2] = np.meshgrid(leftindex, leftindex)
                [RI1, RI2] = np.meshgrid(rightindex, rightindex)
                [RI3, LI3] = np.meshgrid(rightindex, leftindex)
                [LI4, RI4] = np.meshgrid(leftindex, rightindex)

                r = sm[LI1, LI2]
                rp = sm[RI1, RI2]
                t = sm[RI3, LI3]  # left to right
                tp = sm[LI4, RI4]  # right to left

                # do svd of each block to get transmission/reflection eigenvalues
                ur, sr, vrh = np.linalg.svd(r)
                urp, srp, vrph = np.linalg.svd(rp)
                ut, st, vth = np.linalg.svd(t)
                utp, stp, vtph = np.linalg.svd(tp)
                ### BEWARE // ORDERING OF SINGULAR VECTORS/VALUES DIFFERS

                # check scattering matrix symmetries (unitarity, reciprocity and time reversal)
                unit = np.allclose(sm @ sm.conj().T, np.eye(sm.shape[0]))
                reciprocal = np.allclose(sm - sm.T, np.zeros(sm.shape[0]))
                timerev = np.allclose(sm @ sm.conj(), np.eye(sm.shape[0]))

                # scattering matrix eigenvalues
                eigs = np.linalg.eigvals(sm)

                # scattering matrix determinant
                det = np.linalg.det(sm)

                # add to result store
                all_eigs.append(eigs)
                all_dets.append(det)
                all_unitary.append(unit)
                all_reciprocal.append(reciprocal)
                all_timerev.append(timerev)
                all_teigLR.append(st)
                all_teigRL.append(stp)
                all_reigLL.append(sr)
                all_reigRR.append(srp)
                all_int_nodes.append(nw['internal_nodes'])
                all_ext_nodes.append(nw['exit_nodes'])

        # convert to numpy arrays
        all_eigs = np.array(all_eigs)
        all_dets = np.array(all_dets)
        all_teigLR = np.array(all_teigLR)
        all_teigRL = np.array(all_teigRL)
        all_reigLL = np.array(all_reigLL)
        all_reigRR = np.array(all_reigRR)

        # Save processed data
        data_to_store = {'eigenvalues': all_eigs,
                         'det_sm': all_dets,
                         'tau_LR': all_teigLR,
                         'tau_RL': all_teigRL,
                         'rho_LL': all_reigLL,
                         'rho_RR': all_reigRR,
                         'internal_nodes': all_int_nodes,
                         'exit_nodes': all_ext_nodes,
                         'unitary': all_unitary,
                         'reciprocal': all_reciprocal,
                         'timereversal': all_timerev,
                         'network_spec': network_spec,  # NB only saves last value but all should be the same
                         'node_spec': node_spec,  # NB only saves last value but all should be the same
                         'network_type': network_type,  # NB only saves last value but all should be the same
                         }

        savefilename = 'data/{}/proc_data_{}.h5'.format(runid, runid)
        save_dict_to_hdf5(data_to_store, savefilename)
elif process_or_load.upper() == 'L':
    preprocess_list = [runid for runid in dirs if os.path.exists('data/{}/proc_data_{}.h5'.format(runid, runid))]

    for i, entry in enumerate(preprocess_list):
        print("[{}] {}".format(i, entry))

    inputstr = (input("Please enter index (or comma separated indexes) of dataset(s) above to analyse:"))
    strlist = inputstr.split(',')
    dataindices = [int(s) for s in strlist]

    total_trans = []

    for dataindex in dataindices:
        # dataindex = int(input("Please enter   index of dataset above to load:"))
        if dataindex not in range(0, len(preprocess_list)):
            raise ValueError('Incorrect value given')

        runid = preprocess_list[dataindex]
        datafile = 'data/{}/proc_data_{}.h5'.format(runid, runid)

        # make outputfolder if it doesnt exist
        outputdir = 'output/{}'.format(runid)
        if not os.path.exists(outputdir):
            os.makedirs(outputdir)

        loadeddata = load_dict_from_hdf5(datafile)

        all_eigs = loadeddata['eigenvalues']
        all_dets = loadeddata['det_sm']
        all_teigLR = loadeddata['tau_LR']
        all_teigRL = loadeddata['tau_RL']
        all_reigLL = loadeddata['rho_LL']
        all_reigRR = loadeddata['rho_RR']
        all_int_nodes = loadeddata['internal_nodes']
        all_ext_nodes = loadeddata['exit_nodes']
        all_unitary = loadeddata['unitary']
        all_reciprocal = loadeddata['reciprocal']
        all_timerev = loadeddata['timereversal']
        network_spec = loadeddata['network_spec']
        node_spec = loadeddata['node_spec']
        network_type = loadeddata['network_type']
        seed = 0

        total_trans.append(np.mean([np.mean(tau) for tau in all_teigLR]))

        plt.close('all')
        # %% PLOT AN EXAMPLE NETWORK
        network = Network(network_type,
                          network_spec,
                          node_spec,
                          seed_number=seed)

        network.draw('')
        plt.title('Example network')
        plt.savefig("output/{}/example_network_{}.pdf".format(runid, runid), format='pdf')

        # %% PLOT NODE NUMBER DISTRIBUTIONS

        plt.figure('node numbers')
        plt.subplot(1, 2, 1)
        sns.histplot((all_int_nodes), discrete=True, binrange=(min(all_int_nodes) - 5, max(all_int_nodes) + 5))
        plt.title('Internal node distribution')
        plt.subplot(1, 2, 2)
        sns.histplot((all_ext_nodes), discrete=True, binrange=(min(all_ext_nodes) - 5, max(all_ext_nodes) + 5))
        plt.title('Exit node distribution')
        plt.savefig("output/{}/node_distribution_{}.pdf".format(runid, runid), format='pdf')

        # %%  PLOT EIGENVALUE DISTRIBUTION
        df = pd.DataFrame()
        df['real'] = np.real(all_eigs.flatten())
        df['imag'] = np.imag(all_eigs.flatten())

        sns.jointplot(data=df, x='real', y='imag', kind='hex', xlim=(-1.1, 1.1), ylim=(-1.1, 1.1))
        figabs = plt.gcf()
        figabs.canvas.set_window_title('eigenvalue magnitude')
        plt.savefig("output/{}/SM_ev_dist_{}.pdf".format(runid, runid), format='pdf')

        figang = plt.figure('eigenvalue phase')
        figang.clear()
        sns.histplot(np.angle(all_eigs.flatten()) / np.pi, bins=40)
        plt.savefig("output/{}/SM_ev_phase_{}.pdf".format(runid, runid), format='pdf')

        # %% PLOT TRANSMISSION/REFLECTION EIGENVALUE DISTRIBUTION

        figtreigs = plt.figure('transmission/reflection eigenvalues')

        plt.subplot(2, 2, 1)
        sns.histplot(all_reigLL.flatten(), bins=40)
        plt.title(r'$\rho_{LL}$')

        plt.subplot(2, 2, 2)
        sns.histplot(all_teigRL.flatten(), bins=40)
        plt.title(r'$\tau_{LR}$')

        plt.subplot(2, 2, 3)
        sns.histplot(all_teigLR.flatten(), bins=40)
        plt.title(r'$\tau_{LR}$')

        plt.subplot(2, 2, 4)
        sns.histplot(all_reigRR.flatten(), bins=40)
        plt.title(r'$\rho_{RR}$')
        plt.savefig("output/{}/t_r_evs_{}.pdf".format(runid, runid), format='pdf')

    if len(total_trans) > 1:
        fig_totaltrans = plt.figure('total transmission')
        plt.plot(total_trans)
else:
    raise ValueError('Incorrect value given. Choose either "P" or "L".')
