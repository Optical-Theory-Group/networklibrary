# setup code logging
import logging
import unittest

import numpy as np

import logconfig
from complexnetworklibrary.components.link import LINK
from complexnetworklibrary.util import compare_dict

logconfig.setup_logging()
logger = logging.getLogger(__name__)

class LinkTestCase(unittest.TestCase):
    def setUp(self):
        logging.info("Setting up unit tests")
        self.test_link_dict = {
            'node1': 1,
            'node2': 2,
            'link_type': 'exit',
            'distance': 1,
            'n': 1.0,
            'k': 1,
            'inwave': np.array([0 + 0j, 0 + 0j]),
            'outwave': np.array([0 + 0j, 0 + 0j]),
            'S_mat': np.array([[0, 1 + 0j],
                               [1 + 0j, 0]]),
            'iS_mat': np.array([[0, 1 + 0j],
                                [1 + 0j, 0]])
        }

        link = LINK()
        link.node1 = 1
        link.node2 = 2
        link.link_type = 'exit'
        link.distance = 1
        link.n = 1.0
        link.k = 1.0
        link.inwave = np.array([0 + 0j, 0 + 0j])
        link.outwave = np.array([0 + 0j, 0 + 0j])
        link.S_mat = np.array([[0, 1 + 0j],
                          [1 + 0j, 0]])
        link.iS_mat = np.array([[0, 1 + 0j],
                           [1 + 0j, 0]])
        self.link = link

    def test_LinkType(self):
        """
        Test restriction on node_type works correctly
        """
        logging.info("Beginning unittest: LinkType")
        link = LINK()

        link.link_type = 'exit'
        link.link_type = 'internal'
        self.assertTrue(isinstance(link.link_type, str))
        with self.assertRaises(Exception):
            link.link_type = 'wrong_type'

    def test_dict_to_link(self):
        """
        Test initialisation of a link from a dictionary
        """
        logging.info("Beginning unittest: dict_to_link")
        newlink = LINK(linkdict=self.test_link_dict)
        newdict = newlink.link_to_dict()

        self.assertTrue(compare_dict(self.test_link_dict, newdict))

    def test_link_to_dict(self):
        """
        Test conversion of a link to a dictionary
        """
        logging.info("Beginning unittest: link_to_dict")
        newdict = self.link.link_to_dict()
        self.assertTrue(compare_dict(self.test_link_dict, newdict))

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
