# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 17:25:19 2022

Test network load/save functions

"""

import unittest
import numpy as np
import os
import time
import random

# Import the Network class from the complexnetworklibrary.network module
from complexnetworklibrary.network import Network
from complexnetworklibrary.util import compare_dict

class NetworkTestCase(unittest.TestCase):
    def setUp(self):
        # Define some parameters to create a network object
        wavelength = 500e-9
        k = (2 * np.pi) / wavelength
        n = 1
        seed = 3

        # Define the type and specifications for the network to be created
        network_type = 'empty'
        network_spec = {}#{'wavenumber': 1,
                        # 'refractive_index':1}

        # Define some specifications for the nodes in the network
        node_spec = {'Smat_type': 'isotropic_unitary',
                     'scat_loss': 0}

        # Create a new network object with the specified parameters and specifications
        self.network = Network(network_type,
                               network_spec,
                               node_spec,
                               seed_number=seed)

        for i in range(1, 4):
            self.network.add_node(number=i,position=(random.uniform(0, 1), random.uniform(0, 1)))
        for i in range(4, 10):
            self.network.add_node(number=i,position=(random.uniform(0, 1), random.uniform(0, 1)))

        # Connect the nodes in the first component
        self.network.add_connection(node1=1, node2=2, distance=1)
        self.network.add_connection(node1=1, node2=3, distance=1)
        self.network.add_connection(node1=2, node2=3, distance=1)

        # Connect the nodes in the second component
        self.network.add_connection(node1=4, node2=5, distance=1)
        self.network.add_connection(node1=4, node2=6, distance=1)
        self.network.add_connection(node1=5, node2=6, distance=1)
        self.network.add_connection(node1=7, node2=8, distance=1)
        self.network.add_connection(node1=7, node2=9, distance=1)
        self.network.add_connection(node1=8, node2=9, distance=1)
        self.network.draw()

    def test_network_load_save(self):
        # Define a filepath to save the network object to
        filepath = '../data/test_network.h5'

        # Get the directory name from the filepath
        datadir = os.path.dirname(filepath)

        # Check if the directory exists, and create it if it doesn't
        if not os.path.exists(datadir):
            os.makedirs(datadir)

        # Save the network object to the filepath
        self.network.save_network(filepath)

        for ii in range(0, 2):  # pause due to file syncing
            print(ii)
            time.sleep(1)

        # Create a new network object by loading the saved file
        newnetwork = Network(filename=filepath)
        os.remove(filepath)

        d1 = self.network.network_to_dict()
        d2 = newnetwork.network_to_dict()

        self.assertTrue(compare_dict(d1,d2))


if __name__ == '__main__':
    unittest.main(verbosity=2)
