# -*- coding: utf-8 -*-
"""
Created on Fri Nov 11 13:20:30 2022

@author: mforeman

Class for generating, storing and retrieving network ensemble data

"""

# setup code logging
import logging
import multiprocessing as mp
from datetime import datetime, timedelta

import h5py

import logconfig
from complex_network.networks.network import Network

from ._dict_hdf5 import (load_dict_from_hdf5,
                         recursively_save_dict_contents_to_group)
from .util import update_progress

logconfig.setup_logging()
logger = logging.getLogger(__name__)


class NetworkEnsemble:
    global resultstosave
    global progresscount
    global totalrealisations
    global starttime
    resultstosave = {}
    progresscount = 0

    def __init__(self, savetofilename: str, writemode: str, network_type: str, network_spec: dict, node_spec: dict,
                 nbatch: int = 1000, nbatchsize: int = 1, initial_seed:int = 0) -> None:
        """
            Upon initialising this class an ensemble of complex networks with the specified network and node properties
            is generated. Refer to network.py and node.py documentation for further details on network_type,network_spec
            and node_spec inputs. Generated ensemble data is saved to a hdf5 file. Data saved is the dictionary output
            from a call to the Network.network_to_dict() method. If called on an existing data file, data is appended 
            to existing realisations.

            Parameters
            ----------
                savetofilename: str
                    Specify the name of the hdf5 file where results are saved
                writemode: str
                    Output from Network.network_to_dict() will be pruned depending on value chosen according to:
                        - 'sm':  only stores network scattering matrix and relevant node ids/order information
                        - 'network': stores the 'NETWORK' results
                        - otherwise the entire output dictionary is stored
                network_type: str
                    Specify the type of network to be generated
                network_spec: dict
                    Specify the network type and parameters
                node_spec: dict
                    Specify the node model
                nbatch: int
                    Specify how many parallel batches of realisations
                nbatchsize: int
                    Specify the number of realisations to run in each parallel batch
                initial_seed: int
                    Set the random seed for the first realisation
        """
        global resultstosave
        global progresscount
        global totalrealisations
        global starttime
        # TO DO - verify that supplied parameters match those for data saved in an existing hdf5

        resultstosave = {}
        progresscount = 0

        h5py.get_config().track_order = True
        totalrealisations = (nbatch * nbatchsize)

        with h5py.File(savetofilename, 'a', driver=None) as h5file:
            logging.info("Data will be saved to {}".format(h5file))
            existing_realisations = [int(k) for k in h5file.keys()]
            starttime = datetime.now()  # time.time()

            for b in range(0, nbatch):
                if nbatchsize > 1:
                    update_progress(progresscount / totalrealisations, '{}/{}'.format(progresscount, totalrealisations))
                    pool = mp.Pool(mp.cpu_count())
                    seedstorun = [initial_seed + b * nbatchsize + bb for bb in range(0, nbatchsize) if
                                  initial_seed + b * nbatchsize + bb not in existing_realisations]
                    skipped = nbatchsize - len(seedstorun)
                    progresscount += skipped
                    if skipped > 0: print('\n Skipping {} realisations that already exist.'.format(skipped))
                    poolresult = [pool.apply_async(self.single_run, args=(seed, network_type, network_spec, node_spec),
                                                   callback=self.collect_result) for seed in seedstorun]
                    pool.close()
                    pool.join()
                else:
                    for bb in range(0, nbatchsize):
                        update_progress(progresscount / totalrealisations,
                                        '{}/{}'.format(progresscount, totalrealisations))
                        i = b * nbatchsize + bb

                        if (initial_seed + i) not in existing_realisations:
                            networkdictresult = self.single_run(initial_seed + i, network_type, network_spec, node_spec)
                            self.collect_result(networkdictresult)
                        else:
                            progresscount += 1
                            print('\n Realisation {} already exists - skipping'.format(initial_seed + i))

                self.save_results(h5file, writemode)

    def save_results(self, h5file, writemode):
        global resultstosave
        towrite_all = {}
        for seednum, networkdictresult in resultstosave.items():
            if writemode == 'sm':
                towrite = {"NETWORK":
                    {
                        "scattering_matrix": networkdictresult['NETWORK']['scattering_matrix'],
                        "sm_node_order": networkdictresult['NETWORK']['sm_node_order'],
                        "exit_ids": networkdictresult['NETWORK']['exit_ids'],
                        "exit_positions": networkdictresult['NETWORK']['exit_positions'],
                    }
                }
            # exit node ids and positions
            elif writemode == 'network':
                towrite = {"NETWORK": networkdictresult['NETWORK'] }
            else:
                towrite = networkdictresult

            towrite_all.update({'{}'.format(networkdictresult['NETWORK']['seed_number']): towrite})

        recursively_save_dict_contents_to_group(h5file, '/', towrite_all)
        resultstosave = {}

    def single_run(self, seed_number, network_type, network_spec, node_spec):
        network = Network(network_type, network_spec, node_spec, seed_number)
        smd, node_order = network.scattering_matrix_direct()
        networkdict = network.network_to_dict()
        return networkdict

    @staticmethod
    def collect_result(result):
        global resultstosave
        global progresscount
        global totalrealisations
        global starttime

        runtime = datetime.now() - starttime  # time.time() - starttime
        runtime -= timedelta(microseconds=runtime.microseconds)
        progresscount += 1
        remaintime = (runtime / progresscount) * (totalrealisations - progresscount)

        update_progress(progresscount / totalrealisations,
                        '{}/{}. \n Run time {}. \n Est. time remaining  {}. \n'.format(progresscount, totalrealisations,
                                                                                       runtime, remaintime))
        resultstosave.update({'{}'.format(result['NETWORK']['seed_number']): result})

    @staticmethod
    def errorCallback(exception):
        print('Error Thrown: ')
        print(exception)
