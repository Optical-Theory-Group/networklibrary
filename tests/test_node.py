# setup code logging
import logging
import unittest

import numpy as np

import complex_network._logconfig as _logconfig
from complex_network.components.node import NODE
from complex_network.utils import compare_dict

_logconfig.setup_logging()
logger = logging.getLogger(__name__)

class NodeTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Setting up unit tests")
        self.test_node_dict = {
            'number': 1,
            'position': (.3, -3),
            'node_type': 'internal',
            'n_connect': 4,
            'sorted_connected_nodes': [2, 3, 1, 0],
            'Smat_type': 'test',
            'scat_loss': 1,
            'Smat_params': None,
            'inwave': {0: 1, 1: 1, 2: 1, 3: 1},
            'outwave': {0: 0, 1: 0, 2: 0, 3: 0},
            'inwave_np': [1, 1, 1, 1],
            'outwave_np': [0, 0, 0, 0],
            'S_mat': np.identity(4),
            'iS_mat': np.identity(4)
        }

        node = NODE(number = 1, pos = (.3, -3), nodetype = 'internal')
        node.n_connect = 4
        node.sorted_connected_nodes = [2, 3, 1, 0]
        node.Smat_type = 'test'
        node.scat_loss = 1
        node.Smat_params = None
        node.inwave = {0:1,1:1,2:1,3:1}
        node.outwave = {0: 0, 1: 0, 2: 0, 3: 0}
        node.inwave_np = [1, 1, 1, 1]
        node.outwave_np = [0, 0, 0, 0]
        node.S_mat = np.identity(4)
        node.iS_mat = np.identity(4)
        self.node = node

    def test_NodeType(self):
        """
        Test restriction on node_type works correctly
        """
        logging.info("Beginning unittest: NodeType")
        node = NODE(number = 1, pos = (.3, -3), nodetype = 'internal')

        node.node_type = 'exit'
        node.node_type = 'internal'
        self.assertTrue(isinstance(node.node_type, str))
        with self.assertRaises(Exception):
            node.node_type = 'wrong_type'

    def test_dict_to_node(self):
        """
        Test initialisation of a node from a dictionary
        """
        logging.info("Beginning unittest: dict_to_node")
        newnode = NODE(nodedict=self.test_node_dict)
        newdict = newnode.node_to_dict()

        self.assertTrue(compare_dict(self.test_node_dict,newdict))

    def test_node_to_dict(self):
        """
        Test conversion of a node to a dictionary
        """
        logging.info("Beginning unittest: node_to_dict")
        newdict = self.node.node_to_dict()

        self.assertTrue(compare_dict(self.test_node_dict,newdict))

    def test_update(self):
        """
        TO DO - WRITE THIS UNIT TEST
        Returns
        -------

        """
        logging.warning("Empty unittest: update")
        self.assertEqual(True, True)

if __name__ == '__main__':
    unittest.main(verbosity=2)