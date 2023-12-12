# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:38:08 2020

@author: Matthew Foreman

Class file for Network object. Inherits from NetworkGenerator class

"""

# setup code logging
import logging
from copy import deepcopy
from typing import Any
import warnings
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from scipy import optimize
from scipy.linalg import dft, null_space

import logconfig

from ._dict_hdf5 import load_dict_from_hdf5, save_dict_to_hdf5
from ._generator import NetworkGenerator
from complexnetworklibrary.components.link import Link
from complexnetworklibrary.components.node import Node
from .util import detect_peaks, plot_colourline, update_progress

logconfig.setup_logging()
logger = logging.getLogger(__name__)


class Network:
    def __init__(
        self,
        nodes: dict[int, Node],
        links: dict[int, Link],
        data: dict[str, Any] | None = None,
    ) -> None:
        self.reset_values(data)
        self.node_dict = nodes
        self.link_dict = links
        self.reset_fields()

    @property
    def nodes(self):
        return list(self.node_dict.values())

    @property
    def num_nodes(self):
        return len(self.nodes)

    @property
    def exit_nodes(self):
        return [node for node in self.nodes if node.node_type == "exit"]

    @property
    def num_exit_nodes(self):
        return len(self.exit_nodes)

    @property
    def exit_node_indices(self):
        return [node.index for node in self.exit_nodes]

    @property
    def internal_nodes(self):
        return [node for node in self.nodes if node.node_type == "internal"]

    @property
    def num_internal_nodes(self):
        return len(self.internal_nodes)

    @property
    def links(self):
        return list(self.link_dict.values())

    @property
    def num_links(self):
        return list(self.links)

    @property
    def exit_links(self):
        return [link for link in self.links if link.node_type == "exit"]

    @property
    def num_exit_links(self):
        return list(self.exit_links)

    @property
    def internal_links(self):
        return [link for link in self.links if link.node_type == "internal"]

    @property
    def num_internal_links(self):
        return list(self.internal_links)

    @property
    def connections(self):
        """Alias for links"""
        return self.links

    def get_node(self, index: int | str) -> Node:
        """Returns the node with the specified index identifying number"""
        return self.node_dict[str(index)]

    def get_link(self, index: int | str) -> Link:
        """Returns the link with the specified index identifying number"""
        return self.link_dict[str(index)]

    def reset_values(self, data: dict[str, Any] | None = None) -> None:
        """Reset values of network to defaults or those in provided data"""
        default_values = self.get_default_values()
        if data is not None:
            default_values.update(data)
        for key, value in default_values.items():
            setattr(self, key, value)

    def reset_fields(self) -> None:
        """Reset the values of the incident and outgoing fields to be zero"""
        # Reset all node and link values
        for node in self.nodes:
            node.reset_fields()
        for link in self.links:
            link.reset_fields()

        # Set up keys
        for node in self.exit_nodes:
            self.inwave[str(node.index)] = 0 + 0j
            self.outwave[str(node.index)] = 0 + 0j

        # Set up np arrays
        self.inwave_np = np.zeros(len(self.inwave.keys()), dtype=np.complex128)
        self.outwave_np = np.zeros(
            len(self.inwave.keys()), dtype=np.complex128
        )

    @staticmethod
    def get_default_values() -> dict[str, Any]:
        """Default values for the network"""
        default_values: dict[str, Any] = {
            "node_dict": {},
            "link_dict": {},
            "inwave": {},
            "outwave": {},
            "inwave_np": np.zeros(0, dtype=np.complex128),
            "outwave_np": np.zeros(0, dtype=np.complex128),
            "S_mat": np.zeros(0, dtype=np.complex128),
            "iS_mat": np.zeros(0, dtype=np.complex128),
        }
        return default_values

    def update_wave_parameters(
        self,
        n: float | complex | None = None,
        k0: float | complex | None = None,
    ) -> None:
        """Update n and k0 throughout the network"""
        if n is None and k0 is None:
            return
        for link in self.links:
            if n is not None:
                link.n = n
            if k0 is not None:
                link.k0 = k0

    def update_link_S_matrices(
        self,
        n: float | complex | None = None,
        k0: float | complex | None = None,
    ) -> None:
        """Update link scattering matrices throughout network"""
        self.update_wave_parameters(n, k0)
        for link in self.links:
            link.update_S_matrices()

    # -------------------------------------------------------------------------
    # Scattering methods
    # -------------------------------------------------------------------------

    def get_S_matrix_iterative(
        self,
        n: float | complex | None = None,
        k0: float | complex | None = None,
        direction: str = "forward",
        max_num_steps: int = 10**5,
        tolerance: float = 1e-8,
        verbose: bool = True,
    ) -> np.ndarray:
        """Calculate the network scattering matrix iteratively"""
        # Update network with given wave parameters
        self.update_link_S_matrices(n, k0)

        matrix = np.zeros(
            (self.num_exit_nodes, self.num_exit_nodes), dtype=np.complex128
        )
        if verbose:
            node_pbar = tqdm(total=self.num_exit_nodes, desc="Exit nodes")

        # Loop over exit nodes
        for i in range(self.num_exit_nodes):
            incident_field = np.zeros(self.num_exit_nodes, dtype=np.complex128)
            incident_field[i] = 1.0 + 0j

            scattered_field = self.scatter_iterative(
                incident_field, direction, max_num_steps, tolerance, verbose
            )

            matrix[:, i] = scattered_field
            if verbose:
                node_pbar.update(1)

        # Store matrix in network
        if direction == "forward":
            self.S_mat = matrix
            self.iS_mat = np.linalg.inv(matrix)
        else:
            self.iS_mat = matrix
            self.S_mat = np.linalg.inv(matrix)

        return matrix

    def scatter_iterative(
        self,
        incident_field: np.ndarray | None = None,
        direction: str = "forward",
        max_num_steps: int = 10**5,
        tolerance: float = 1e-8,
        verbose: bool = False,
    ) -> np.ndarray:
        """Scatter the incident field through the network"""
        # Reset fields throughout the network
        self.reset_fields()

        # Set incident field if it is given
        if incident_field is not None:
            self.set_incident_field(incident_field, direction)

        # Scatter recursively
        has_converged = False

        if verbose:
            total_pbar = tqdm(total=max_num_steps, desc="Steps")
            conv_pbar = tqdm(total=1, desc="Convergence")

        for i in range(max_num_steps):
            before = np.copy(self.get_outgoing_fields(direction))
            self.scatter_step(direction)
            after = np.copy(self.get_outgoing_fields(direction))

            # Update progress bar
            if verbose:
                total_pbar.update(1)

            # Scattered field has not yet reached the exit nodes
            if np.isclose(np.linalg.norm(after), 0.0):
                continue

            # Check for convergence
            diff = np.linalg.norm(after - before)
            if verbose:
                frac = min(np.log10(diff) / np.log10(tolerance), 1.0)
                conv_pbar.update(frac - conv_pbar.n)

            if diff <= tolerance:
                has_converged = True
                if verbose:
                    print(f"Converged after {i} steps.")
                break

        # Close pbars
        if verbose:
            conv_pbar.close()
            total_pbar.close()

        # Throw warning if field has not converged
        if not has_converged:
            warnings.warn(
                "Max number of steps has been reached, but field "
                "has not converged! Consider increasing the maximum number of "
                "steps.",
                category=UserWarning,
            )

        return after

    def set_incident_field(
        self, incident_field: np.ndarray, direction: str = "forward"
    ) -> None:
        """Sets the incident field to the inwaves/outwaves"""
        # Check size
        if len(incident_field) != self.num_exit_nodes:
            raise ValueError(
                f"Incident field has incorrect size. "
                f"It should be of size {self.num_exit_nodes}."
            )

        # Set values to nodes and network dictionaries
        for i, exit_node in enumerate(self.exit_nodes):
            if direction == "forward":
                self.inwave[str(exit_node.index)] = incident_field[i]
                exit_node.inwave["-1"] = incident_field[i]
                exit_node.inwave_np[0] = incident_field[i]
            elif direction == "backward":
                self.outwave[str(exit_node.index)] = incident_field[i]
                exit_node.outwave["-1"] = incident_field[i]
                exit_node.outwave_np[0] = incident_field[i]

        # Set values to network
        if direction == "forward":
            self.inwave_np = incident_field
        if direction == "backward":
            self.outwave_np = incident_field

    def get_outgoing_fields(self, direction: str = "forward") -> np.ndarray:
        """Get the current outgoinf field on the basis of the given
        direction"""
        if direction == "forward":
            return self.outwave_np
        else:
            return self.inwave_np

    def update_outgoing_fields(self, direction: str = "forward") -> None:
        """Update the fields from the exit nodes and put them into the network
        inwave/outwaves"""
        for i, node in enumerate(self.exit_nodes):
            if direction == "forward":
                self.outwave[str(node.index)] = node.outwave["-1"]
                self.outwave_np[i] = node.outwave["-1"]
            if direction == "backward":
                self.inwave[str(node.index)] = node.inwave["-1"]
                self.inwave_np[i] = node.inwave["-1"]

    def scatter_step(self, direction: str = "forward") -> None:
        """Perform one step of scattering throughout the network"""

        # Scatter at nodes
        for node in self.nodes:
            node.update(direction)

        # Transfer fields to links
        self.nodes_to_links(direction)

        # Scatter in links.
        for link in self.links:
            link.update(direction)

        # Transfer fields to nodes
        self.links_to_nodes(direction)

        # Update network outgoing fields
        self.update_outgoing_fields(direction)

    def nodes_to_links(self, direction: str = "forward") -> None:
        """Transfer fields from nodes to links in given direction"""
        # Give outwaves of nodes to links
        for link in self.links:
            node_one_index, node_two_index = link.node_indices
            node_one_index = str(node_one_index)
            node_two_index = str(node_two_index)
            node_one = self.get_node(node_one_index)
            node_two = self.get_node(node_two_index)

            # Loop through link inwaves
            # Note, note inwave/outwaves are to OTHER NODES, NOT LINKS
            if direction == "forward":
                link.inwave[node_one_index] = node_one.outwave[node_two_index]
                link.inwave[node_two_index] = node_two.outwave[node_one_index]
                link.inwave_np[0] = link.inwave[node_one_index]
                link.inwave_np[1] = link.inwave[node_two_index]

            elif direction == "backward":
                link.outwave[node_one_index] = node_one.inwave[node_two_index]
                link.outwave[node_two_index] = node_two.inwave[node_one_index]
                link.outwave_np[0] = link.outwave[node_one_index]
                link.outwave_np[1] = link.outwave[node_two_index]

    def links_to_nodes(self, direction: str = "forward") -> None:
        """Transfer fields from links to nodes in given direction"""
        # Give outwaves of links to nodes
        for link in self.links:
            node_one_index, node_two_index = link.node_indices
            node_one_index = str(node_one_index)
            node_two_index = str(node_two_index)
            node_one = self.get_node(node_one_index)
            node_two = self.get_node(node_two_index)

            # Loop through link inwaves
            # Note, note inwave/outwaves are to OTHER NODES, NOT LINKS
            if direction == "forward":
                node_one.inwave[node_two_index] = link.outwave[node_one_index]
                node_two.inwave[node_one_index] = link.outwave[node_two_index]

                # Get the indices appropriate for the numpy arrays that
                # correspond to the correct wave
                np_one = node_one.sorted_connected_nodes.index(
                    int(node_two_index)
                )
                np_two = node_two.sorted_connected_nodes.index(
                    int(node_one_index)
                )
                node_one.inwave_np[np_one] = node_one.inwave[node_two_index]
                node_two.inwave_np[np_two] = node_two.inwave[node_one_index]

            elif direction == "backward":
                node_one.outwave[node_two_index] = link.inwave[node_one_index]
                node_two.outwave[node_one_index] = link.inwave[node_two_index]

                # Get the indices appropriate for the numpy arrays that
                # correspond to the correct wave
                np_one = node_one.sorted_connected_nodes.index(
                    int(node_two_index)
                )
                np_two = node_two.sorted_connected_nodes.index(
                    int(node_one_index)
                )
                node_one.outwave_np[np_one] = node_one.outwave[node_two_index]
                node_two.outwave_np[np_two] = node_two.outwave[node_one_index]

    ###########################################################################
    # %% network initialisation and recursive functions
    ###########################################################################

    # def initialise_network(
    #     self,
    #     node_spec: Dict,
    #     input_amp: Union[Iterable[Union[float, complex]], None] = None,
    #     output_amp: Union[Iterable[Union[float, complex]], None] = None,
    # ) -> None:
    #     """
    #     Sets the scattering matrix condition for each node

    #     Parameters
    #     ----------
    #     node_spec : dictionary
    #         Dictionary defining properties of node.

    #     input_amp : [float], optional
    #         List of input amplitudes for all input/exit nodes. The default is None.
    #         (used for forward recursive algorithm)
    #     output_amp : [float], optional
    #         List of output amplitudes for all input/exit nodes. The default is None.
    #         (used for backwards recursive algorithm)

    #     Returns
    #     -------
    #     None.

    #     """

    #     if "scat_loss" in node_spec.keys():
    #         self.scat_loss = node_spec["scat_loss"]
    #     if input_amp is not None:
    #         self.input = input_amp
    #     if output_amp is not None:
    #         self.output = output_amp

    #     # loop over each node
    #     for node in self.nodes:
    #         # ####    MAKE INWAVE AND OUTWAVE DICTIONARY  #######
    #         sorted_connected_nodes = node.sorted_connected_nodes
    #         node.inwave = {n: (0 + 0j) for n in sorted_connected_nodes}
    #         node.outwave = {n: (0 + 0j) for n in sorted_connected_nodes}
    #         if self.scat_loss != 0 and node.node_type == "internal":
    #             for u in sorted_connected_nodes:
    #                 # node.outwave.update({'loss%u'%u:(0+0j) })
    #                 node.outwave["loss%u" % u] = 0 + 0j

    #         # INITIALISE THE NODE SCATTERING MATRIX
    #         self.initialise_node_S_mat(node.number, **node_spec)
    #         # node.init_S_mat(**node_spec)

    #     # ###### initialising input/output node amplitudes  ##########
    #     if input_amp is not None:
    #         self.reset_inputs(input_amp)

    #     if output_amp is not None:
    #         self.reset_outputs(output_amp)

    # def initialise_node_S_mat(
    #     self, nodeid: int, S_mat_type: str, scat_loss: float, **kwargs
    # ) -> None:
    #     """
    #     Initialise scattering matrix of node

    #     Parameters
    #     ----------
    #     nodeid : int
    #         ID number of node to initialise
    #     S_mat_type : str
    #         Specifies type of scattering matrix to use.
    #             'identity': SM is set to identity matrix - complete reflection at each input
    #             'permute_identity' : permuted identity matrix - rerouting to next edge
    #             'random': random SM. Each element takes a value in [0,1)
    #             'isotropic_unitary': unitary isotropic SM, implemented through DFT matrix of correct dimension
    #             'random_unitary': random unitary SM
    #             'COE' : drawn from COE
    #             'CUE' : drawn from CUE
    #             'unitary_cyclic': unitary cyclic SM constructed through specifying phases of eigenvalues using 'delta'
    #                                 kwarg
    #             'to_the_lowest_index': reroutes all energy to connected node of lowest index
    #             'custom' : Set a custom scattering matrix. Requires kwarg 'S_mat' to be set

    #     scat_loss : float
    #         Specify scattering loss parameter for node, i.e. fraction of power lost
    #     **kwargs : Keyword arguments
    #         Extra keyword arguments required for specified type of scattering matrix:
    #             For S_mat_type == 'custom':
    #                 kwargs['S_mat'] defines custom scattering matrix
    #             For S_mat_type == 'unitary_cyclic':
    #                 kwargs['delta'] is a vector define phase of eigenvalues of scattering matrix

    #     Returns
    #     -------
    #     None.

    #     """
    #     supported_S_mats = [
    #         "identity",
    #         "permute_identity",
    #         "random",
    #         "isotropic_unitary",
    #         "random_unitary",
    #         "COE",
    #         "CUE",
    #         "unitary_cyclic",
    #         "to_the_lowest_index",
    #         "custom",
    #     ]

    #     if S_mat_type not in supported_S_mats:
    #         raise ValueError(
    #             "Specified scattering matrix type is invalid. Please choice one from {}".format(
    #                 supported_S_mats
    #             )
    #         )

    #     node: Node = self.get_node(nodeid)

    #     node.S_mat_type = S_mat_type
    #     node.scat_loss = scat_loss
    #     node.S_mat_params = kwargs

    #     if scat_loss != 0 and node.node_type == "internal":
    #         node.inwave_np = np.array([0 + 0j] * (2 * node.num_connect))
    #         node.outwave_np = np.array([0 + 0j] * (2 * node.num_connect))
    #     else:
    #         node.inwave_np = np.array([0 + 0j] * node.num_connect)
    #         node.outwave_np = np.array([0 + 0j] * node.num_connect)

    #     # scattering matrix for exit node is identity
    #     if node.node_type == "exit":
    #         node.S_mat = np.identity(node.num_connect, dtype=np.complex_)
    #         node.iS_mat = np.identity(node.num_connect, dtype=np.complex_)
    #         return

    #     # scattering matrix for internal node
    #     if node.S_mat_type == "identity":
    #         node.S_mat = np.identity(
    #             node.num_connect, dtype=np.complex_
    #         )  # identity matrix
    #     elif node.S_mat_type == "uniform_random":
    #         node.S_mat = np.random.rand(
    #             node.num_connect, node.num_connect
    #         )  # RANDOM SCATTEING MATRIX (nXn)
    #     elif node.S_mat_type == "isotropic_unitary":
    #         node.S_mat = (1 / node.num_connect) ** 0.5 * dft(node.num_connect)
    #     elif node.S_mat_type == "CUE":
    #         gamma = (
    #             1
    #             if "subunitary_factor" not in kwargs.keys()
    #             else kwargs["subunitary_factor"]
    #         )
    #         x = (
    #             np.identity(1, dtype=np.complex_)
    #             if node.num_connect == 1
    #             else stats.unitary_group.rvs(node.num_connect)
    #         )
    #         node.S_mat = gamma * x
    #     elif node.S_mat_type == "COE":
    #         gamma = (
    #             1
    #             if "subunitary_factor" not in kwargs.keys()
    #             else kwargs["subunitary_factor"]
    #         )
    #         x = (
    #             np.identity(1, dtype=np.complex_)
    #             if node.num_connect == 1
    #             else stats.unitary_group.rvs(node.num_connect)
    #         )
    #         node.S_mat = gamma * (x.T @ x)
    #     elif node.S_mat_type == "permute_identity":
    #         mat = np.identity(node.num_connect, dtype=np.complex_)
    #         inds = [
    #             (i - 1) % node.num_connect for i in range(0, node.num_connect)
    #         ]
    #         node.S_mat = mat[:, inds]
    #     elif node.S_mat_type == "custom":
    #         mat = kwargs["S_mat"]
    #         # dimension checking
    #         if mat.shape != (node.num_connect, node.num_connect):
    #             raise RuntimeError(
    #                 "Supplied scattering matrix is of incorrect dimensions: "
    #                 "{} supplied, {} expected".format(
    #                     mat.shape, (node.num_connect, node.num_connect)
    #                 )
    #             )
    #         else:
    #             node.S_mat = mat
    #     elif node.S_mat_type == "unitary_cyclic":
    #         if "delta" in kwargs.keys():
    #             ll = np.exp(1j * kwargs["delta"][0 : node.num_connect])
    #         else:
    #             ll = np.exp(1j * 2 * np.pi * np.random.rand(node.num_connect))
    #         s = np.matmul((1 / node.num_connect) * dft(node.num_connect), ll)
    #         node.S_mat = np.zeros(
    #             shape=(node.num_connect, node.num_connect), dtype=np.complex_
    #         )
    #         for jj in range(0, node.num_connect):
    #             node.S_mat[jj, :] = np.concatenate(
    #                 (
    #                     s[(node.num_connect - jj) : node.num_connect],
    #                     s[0 : node.num_connect - jj],
    #                 )
    #             )

    #     # define inverse scattering matrix
    #     node.iS_mat = np.linalg.inv(node.S_mat)

    #     # ###  INTRODUCE INCOHERENT SCATTERING LOSS   #########
    #     if scat_loss != 0:
    #         S11 = (np.sqrt(1 - scat_loss**2)) * node.S_mat
    #         S12 = np.zeros(
    #             shape=(node.num_connect, node.num_connect), dtype=np.complex_
    #         )
    #         S21 = np.zeros(
    #             shape=(node.num_connect, node.num_connect), dtype=np.complex_
    #         )
    #         S22 = scat_loss * np.identity(node.num_connect, dtype=np.complex_)

    #         S_mat_top_row = np.concatenate((S11, S12), axis=1)
    #         S_mat_bot_row = np.concatenate((S21, S22), axis=1)
    #         node.S_mat = np.concatenate((S_mat_top_row, S_mat_bot_row), axis=0)

    #         iS11 = node.iS_mat / np.sqrt(1 - scat_loss**2)
    #         iS12 = np.zeros(
    #             shape=(node.num_connect, node.num_connect), dtype=np.complex_
    #         )
    #         iS21 = np.zeros(
    #             shape=(node.num_connect, node.num_connect), dtype=np.complex_
    #         )
    #         iS22 = scat_loss * node.iS_mat / np.sqrt(1 - scat_loss**2)

    #         iS_mat_top_row = np.concatenate((iS11, iS12), axis=1)
    #         iS_mat_bot_row = np.concatenate((iS21, iS22), axis=1)
    #         node.iS_mat = np.concatenate(
    #             (iS_mat_top_row, iS_mat_bot_row), axis=0
    #         )

    # def get_node_amplitudes(self, node_type: str = "all") -> np.ndarray:
    #     """
    #     Returns a vector of all mode amplitudes in the network for specified node types

    #     Returns
    #     -------
    #     None.

    #     """

    #     coeffs = []
    #     for node in self.nodes:
    #         if node_type == "all":
    #             inampls = [x for x in node.inwave.values()]
    #             outampls = [x for x in node.outwave.values()]
    #             ampls = inampls + outampls
    #             coeffs += ampls
    #         else:
    #             if node.node_type != node_type:
    #                 inampls = [x for x in node.inwave.values()]
    #                 outampls = [x for x in node.outwave.values()]
    #                 ampls = inampls + outampls
    #                 coeffs += ampls

    #     return np.array(coeffs)

    # def update_network(self, direction: str = "forward") -> None:
    #     """
    #     Main update function for doing iterative calculation. Algorithm does the following
    #     1. do the propagation through all waveguides using LINK.update()
    #     2. map the modesout from each connection to the modesin for each node
    #     3. apply scattering at each node using Node.update()
    #     4. map the modesout from each node to the modesin for each connection

    #     Parameters
    #     ----------
    #     direction : str, optional
    #         ['forward'|'backward'] Specifies direction in which recursive algorithm is being run.
    #         The default is 'forward'.

    #     Returns
    #     -------
    #     None.

    #     """

    #     for link in self.links:
    #         connected_nodes = [
    #             self.get_node(link.node1),
    #             self.get_node(link.node2),
    #         ]
    #         link.update(connected_nodes, direction)
    #         # link.update(self.nodes)
    #         # link.scattering_matrix(self.nodes)
    #     for node in self.nodes:
    #         node.update(self.scat_loss, direction)
    #     pass

    # def run_network(
    #     self,
    #     n_iterations: int = 10000,
    #     converge: bool = True,
    #     period: int = 200,
    #     threshold: float = 0.0001,
    #     conv_nodes: str = "all",
    #     direction: str = "forward",
    # ):
    #     total_var = 1

    #     if not converge:
    #         for niter in range(1, n_iterations + 1):
    #             # update_progress(niter/n_iterations,'Iterating network...'.format(total_var))
    #             self.update_network(direction)  # updates and iterate

    #     else:
    #         new_coeffs = self.get_node_amplitudes(conv_nodes)
    #         coeffs = np.zeros((period, len(new_coeffs)), dtype=np.complex128)
    #         coeffs[0, :] = new_coeffs
    #         for niter in range(1, n_iterations):
    #             update_progress(
    #                 threshold / abs(total_var - threshold),
    #                 "Iteration {}/{} : variance {}".format(
    #                     niter, n_iterations, total_var
    #                 ),
    #             )
    #             self.update_network(direction)  # updates and iterate
    #             coeffs[niter % period, :] = self.get_node_amplitudes()

    #             # check convergence
    #             if niter % period == 0:
    #                 # print('***********************************')
    #                 total_var = sum(stats.tstd(np.abs(coeffs))) / sum(
    #                     np.mean(np.abs(coeffs), axis=0)
    #                 )

    #                 if total_var < threshold:
    #                     update_progress(
    #                         threshold / abs(total_var - threshold),
    #                         "Iteration {}/{} : variance {}".format(
    #                             niter, n_iterations, total_var
    #                         ),
    #                     )
    #                     # print ('Converged after ',niter,' iterations')
    #                     break

    # #####################################
    # # %% network reset functions
    # #####################################

    # def reset_network(
    #     self,
    #     k: Union[complex, float, None] = None,
    #     input_amp: Union[np.ndarray[complex, float], None] = None,
    #     output_amp: Union[np.ndarray[complex, float], None] = None,
    # ):
    #     """
    #     Resets the specified properties of the network

    #     Parameters
    #     ----------
    #     k : float, optional
    #         Complex wavenumber for propagation along edges = k_0 n. The default is None.
    #     input_amp : [float], optional
    #         List of input mode amplitudes at input/output nodes. The default is None.
    #     output_amp : [float], optional
    #         List of output mode amplitudes at input/output nodes. The default is None.

    #     Returns
    #     -------
    #     None.

    #     """
    #     for node in self.nodes:
    #         for i in node.inwave:
    #             # node.inwave.update({i:0 +0j})
    #             node.inwave[i] = 0 + 0j
    #         for y in node.outwave:
    #             # node.outwave.update({y:0 +0j})
    #             node.outwave[y] = 0 + 0j

    #         node.inwave_np = 0 * node.inwave_np
    #         node.outwave_np = 0 * node.outwave_np

    #     if k is not None:
    #         self.k = k
    #         for link in self.links:
    #             link.reset_link(link.distance, k, link.n)

    #     if input_amp is not None:
    #         self.reset_inputs(input_amp)

    #     if output_amp is not None:
    #         self.reset_outputs(output_amp)

    # def reset_inputs(self, input_amp: np.ndarray[np.complex64]):
    #     """
    #     Helper function to reset input amplitudes of input/output nodes.

    #     Parameters
    #     ----------
    #     input_amp : np.ndarray(np.complex64)
    #         List of input mode amplitudes at input/output nodes. .

    #     Returns
    #     -------
    #     None.

    #     """
    #     self.input = input_amp
    #     input_amps_iterate = iter(input_amp)

    #     for node in self.nodes:
    #         if node.node_type == "exit":
    #             for i in node.inwave:
    #                 # node.inwave.update({i:0 +0j})
    #                 # node.outwave.update({i:next(input_amps_iterate)})
    #                 node.inwave[i] = 0 + 0j
    #                 try:
    #                     node.outwave[i] = next(input_amps_iterate)
    #                 except Exception as e:
    #                     logging.ERROR(e)
    #                     pass

    # def reset_outputs(self, output_amp):
    #     """
    #     Helper function to reset output amplitudes of input/output nodes.

    #     Parameters
    #     ----------
    #     output_amp : [float]
    #         List of output mode amplitudes at input/output nodes. .

    #     Returns
    #     -------
    #     None.

    #     """
    #     self.output = output_amp
    #     output_amps_iterate = iter(output_amp)

    #     for node in self.nodes:
    #         if node.node_type == "exit":
    #             for i in node.inwave:
    #                 # node.outwave.update({i:0 +0j})
    #                 # node.inwave.update({i:next(output_amps_iterate)})
    #                 node.outwave[i] = 0 + 0j
    #                 node.inwave[i] = next(output_amps_iterate)

    # #####################################
    # # %% network analysis functions
    # #####################################

    # def find_pole(
    #     self,
    #     k0: complex,
    #     method: str = "CG",
    #     opts: Union[dict, None] = None,
    #     bounds: Union[tuple, None] = None,
    # ) -> complex:
    #     """
    #     Finds poles of the scattering matrix in the complex k plane (actually search
    #     is based on finding minima of inverse scattering matrix logarithmic determinant).

    #     Parameters
    #     ----------
    #     k0 : float
    #         initial start point.
    #     method : string, optional
    #         Search algorithm (see optimize.minimize documentation). The default is 'CG'.
    #     opts :
    #         Search algorithm options (see optimize.minimize documentation). The default is None.
    #     bounds : tuple of bounds, optional
    #         Bounds on search region (see optimize.minimize documentation).

    #     Returns
    #     -------
    #     kmin
    #         Complex wavenumber defining position of pole.

    #     """
    #     opt_network = deepcopy(self)
    #     optout = optimize.minimize(
    #         lambda kk: self._find_pole_helper(kk[0], kk[1], opt_network),
    #         np.array([k0.real, k0.imag]),
    #         method=method,
    #         options=opts,
    #         bounds=bounds,
    #     )
    #     if not optout["success"]:
    #         print(optout["message"])
    #     # return minimum
    #     minimum = optout["x"]
    #     return minimum[0] + 1j * minimum[1]

    # @staticmethod
    # def _find_pole_helper(
    #     kr: float, ki: float, opt_network: "Network"
    # ) -> complex:
    #     """
    #     Helper function for find_poles function

    #     Parameters
    #     ----------
    #     kr : float
    #         real part of wavenumber.
    #     ki : float
    #         imaginary part of wavenumber.
    #     opt_network : Network
    #         instance of Network being used for minimization.

    #     Returns
    #     -------
    #     det
    #         logarithmic determinant of network scattering matrix at specified wavenumber.

    #     """
    #     k = kr + 1j * ki
    #     opt_network.reset_network(k=k)
    #     ism, _ = opt_network.inverse_scattering_matrix_direct()
    #     # sm,_ = opt_network.scattering_matrix_direct()
    #     # try:
    #     # det = np.abs(np.linalg.det(ism))
    #     sign, det = np.linalg.slogdet(ism)
    #     # sign, det = (np.linalg.slogdet(sm)) # logarithm and sign of determinant
    #     # if np.isnan(det):
    #     #     sm,_ = opt_network.scattering_matrix_direct()

    #     return det  # -sign*det # 1/det

    # # This method calculates determinant of scattering matrix of the network object.
    # # Input parameters:
    # # - self: An instance of Network class.
    # # - kmin: Minimum wavenumber value to calculate determinant.
    # # - kmax: Maximum wavenumber value to calculate determinant.
    # # - npr: Number of points in the real axis.
    # # - npi: Number of points in the imaginary axis.
    # # - takeabs: Flag to determine whether to take the absolute value of determinant or not.
    # # - progress_bar_text: Text to show in the progress bar.
    # def calc_det_S(
    #     self,
    #     kmin: complex,
    #     kmax: complex,
    #     npr: int,
    #     npi: Union[int, None] = None,
    #     takeabs: bool = True,
    #     progress_bar_text: str = "",
    # ):
    #     """
    #     This method calculates determinant of scattering matrix of the network object.

    #     Inputs:
    #         kmin: complex
    #             Minimum wavenumber value to calculate determinant.
    #         kmax: complex
    #             Maximum wavenumber value to calculate determinant.
    #         npr: int
    #             Number of points in the real axis.
    #         npi: int, optional
    #             Number of points in the imaginary axis. DEFAULT = npr
    #         takeabs: bool, optional
    #             Flag to determine whether to take the absolute value of determinant or not. DEFAULT = True
    #         progress_bar_text: str, optional
    #             Text to show in the progress bar.

    #     Returns:
    #         KR, KI: numpy ndarrays giving meshgrid of real and imaginary k coordinates
    #         detS:   abs(|SM|) or |SM| found at each position on k grid
    #         kpeaks: coarse approximation of positions of peaks in detS
    #     """
    #     # If npi is not given, we by default set it equal to npr.
    #     if npi is None:
    #         npi = npr

    #     # Save the original value of k.
    #     korig = self.k

    #     # Get real and imaginary parts of kmin and kmax and create a meshgrid of points for real/imaginary axes.
    #     krmin = kmin.real
    #     kimin = kmin.imag
    #     krmax = kmax.real
    #     kimax = kmax.imag
    #     kr = np.linspace(krmin, krmax, npr)
    #     ki = np.linspace(kimin, kimax, npi)
    #     KR, KI = np.meshgrid(kr, ki)

    #     # Initialise array to store determinant values for each complex wavenumber
    #     if takeabs is True:
    #         detS = np.zeros((npr, npi))
    #     else:
    #         detS = np.zeros((npr, npi)) + 1j * np.zeros((npr, npi))

    #     # Iterate over the grid and calculate the determinant of the scattering matrix for each point.
    #     for ii in range(0, npr):
    #         k_real = kr[ii]
    #         for jj in range(0, npi):
    #             # Update the progress bar and working value of k.
    #             update_progress(
    #                 (ii * npr + jj) / npr**2, status=progress_bar_text
    #             )
    #             k = k_real + 1j * ki[jj]

    #             # Reset the network with the new k value.
    #             self.reset_network(k=k)

    #             # Calculate the scattering matrix and node order and store SM determinant into array.
    #             sm, node_order = self.scattering_matrix_direct()
    #             if takeabs is True:
    #                 detS[ii, jj] = abs(np.linalg.det(sm))
    #             else:
    #                 detS[ii, jj] = np.linalg.det(sm)

    #     # Reset the network back to the original k value.
    #     self.reset_network(k=korig)

    #     # Find peaks of detS.
    #     peak_inds = detect_peaks(abs(detS))

    #     # Create an array with the wavenumbers of the peaks.
    #     kpeaks = np.array(
    #         [
    #             kr[peak_inds[i][0]] + 1j * ki[peak_inds[i][1]]
    #             for i in range(0, len(peak_inds))
    #         ]
    #     )

    #     return KR, KI, detS, kpeaks

    # def calc_det_ImSP(self, kmin, kmax, npr, npi=None, takeabs=True):
    #     if npi is None:
    #         npi = npr
    #     korig = self.k

    #     krmin = np.real(kmin)
    #     kimin = np.imag(kmin)

    #     krmax = np.real(kmax)
    #     kimax = np.imag(kmax)

    #     kr = np.linspace(krmin, krmax, npr)
    #     ki = np.linspace(kimin, kimax, npi)
    #     KR, KI = np.meshgrid(kr, ki)

    #     if takeabs is True:
    #         detS = np.zeros((npr, npi))
    #     else:
    #         detS = np.zeros((npr, npi)) + 1j * np.zeros((npr, npi))

    #     for ii in range(0, npr):
    #         k_real = kr[ii]
    #         for jj in range(0, npi):
    #             update_progress((ii * npr + jj) / npr**2)
    #             k_imag = ki[jj]

    #             k = k_real + 1j * k_imag
    #             self.reset_network(k=k)
    #             S11, P11 = self.get_S11_P11()
    #             sm = np.eye(S11.shape[0]) - S11 @ P11
    #             if takeabs is True:
    #                 detS[ii, jj] = abs(np.linalg.det(sm))
    #             else:
    #                 detS[ii, jj] = np.linalg.det(sm)

    #     self.reset_network(k=korig)

    #     peak_inds = detect_peaks(abs(detS))
    #     kpeaks = np.array(
    #         [
    #             kr[peak_inds[i][0]] + 1j * ki[peak_inds[i][1]]
    #             for i in range(0, len(peak_inds))
    #         ]
    #     )

    #     return KR, KI, detS, kpeaks

    # def calc_input_energy(self, impedance=1.0):
    #     """
    #     Calculates the total power input into the network based on exit node amplitudes.
    #     Power given is expressed per unit cross-sectional area of input connections

    #     Parameters
    #     ----------
    #     impedance : float, optional
    #         wave impedance of connections

    #     """

    #     input_power = 0
    #     for node in self.nodes:
    #         if node.node_type == "exit":
    #             input_power += np.sum(
    #                 np.abs(np.array(list(node.outwave.values()))) ** 2
    #             )  # note energy going into network is going out from the exit node

    #     return input_power / impedance

    # def calc_output_energy(self, impedance=1.0):
    #     """
    #     Calcualtes the total power ouptut from the network based on exit node amplitudes.
    #     Power given is expressed per unit cross-sectional area of input connections

    #     Parameters
    #     ----------
    #     impedance : float, optional
    #         wave impedance of connections

    #     """
    #     output_power = 0
    #     for node in self.nodes:
    #         if node.node_type == "exit":
    #             output_power += np.sum(
    #                 np.abs(np.array(list(node.inwave.values()))) ** 2
    #             )

    #     return output_power / impedance

    # # def calc_internal_energy(self, epsr=1):
    # #     """
    # #     Calculates the total energy density within the network.
    # #     Energy density given is expressed per unit cross-sectional area of connections
    # #
    # #     Parameters
    # #     ----------
    # #     epsr : float, optional
    # #         relative permittivity of connections
    # #     """
    # #     Utot = 0
    # #     eps0 = 1  # 8.854187812*1e-12
    # #
    # #     for connection in self.links:
    # #         d = connection.distance
    # #         Ap = connection.inwave[0]
    # #         Am = connection.inwave[1]
    # #         Apc = np.conjugate(Ap)
    # #         Amc = np.conjugate(Am)
    # #
    # #         kr = np.real(self.k)
    # #         ki = np.imag(self.k)
    # #
    # #         if ki == 0:
    # #             UintPP = Ap * Apc * d
    # #             UintMM = Am * Amc * d
    # #             UintPM = Ap * Amc * (np.sin(kr * d)) / kr
    # #             UintMP = Am * Apc * (np.sin(kr * d)) / kr
    # #         else:
    # #             UintPP = Ap * Apc * (1 - np.exp(-2 * d * ki)) / (2 * ki)
    # #             UintMM = Am * Amc * (1 - np.exp(-2 * d * ki)) / (2 * ki)
    # #             UintPM = Ap * Amc * (np.sin(kr * d)) * np.exp(-d * ki) / kr
    # #             UintMP = Am * Apc * (np.sin(kr * d)) * np.exp(-d * ki) / kr
    # #
    # #         U1 = 2 * 0.5 * eps0 * epsr * (UintPP + UintMM + UintPM + UintMP)  # twice since we assume Ue = Um
    # #
    # #         Utot += U1
    # #
    # #     return Utot

    # ###########################
    # # %%  network properties
    # ###########################

    # def laplacian_matrix(
    #     self,
    # ):
    #     """
    #     Returns Laplacian matrix for network
    #     https://en.wikipedia.org/wiki/Laplacian_matrix

    #     """
    #     A = self.adjacency_matrix()
    #     D = self.degree_matrix()
    #     return D - A

    # def degree_matrix(
    #     self,
    # ):
    #     """
    #     Returns degree matrix for network
    #     https://en.wikipedia.org/wiki/Degree_matrix

    #     """
    #     deg = np.zeros((self.total_nodes, self.total_nodes))

    #     # given we have no control over what no numbers have been assigned to each node
    #     # we first retrieve a list of node IDs, then sort them, then loop over each node
    #     # and fill out degree matrix

    #     # get list of node IDs
    #     IDs = [0] * self.total_nodes
    #     for index, node in enumerate(self.nodes):
    #         IDs[index] = node.number

    #     # sort nodes
    #     sorted_nodes = sorted(IDs)

    #     # construct adjacency matrix
    #     for n_id in sorted_nodes:
    #         node = self.get_node(n_id)
    #         index = sorted_nodes.index(n_id)
    #         deg[index, index] = node.degree

    #     return deg

    # def adjacency_matrix(
    #     self,
    # ):
    #     """
    #     Returns adjacency matrix for network
    #     https://en.wikipedia.org/wiki/Adjacency_matrix

    #     """
    #     adj = np.zeros((self.total_nodes, self.total_nodes))

    #     # given we have no control over what numbers have been assigned to each node
    #     # we first retrieve a list of node IDs, then sort them, then loop over each node
    #     # and fill out adjacency matrix according to connected nodes

    #     # get list of node IDs
    #     IDs = [0] * self.total_nodes
    #     for index, node in enumerate(self.nodes):
    #         IDs[index] = node.number

    #     # sort nodes
    #     sorted_nodes = sorted(IDs)

    #     # construct adjacency matrix
    #     for jj, index in enumerate(sorted_nodes):
    #         node = self.get_node(index)
    #         connected = node.sorted_connected_nodes
    #         for connected_index in connected:
    #             adj[jj, sorted_nodes.index(connected_index)] = 1

    #     return adj

    # def fiedler(
    #     self,
    # ):
    #     """
    #     Returns Fiedler value or algebraic connectivity for network
    #     https://en.wikipedia.org/wiki/Algebraic_connectivity
    #     """

    #     L = self.laplacian_matrix()
    #     eigensystem = np.linalg.eig(L)
    #     eigenvalues = eigensystem[0]
    #     eigenvectors = eigensystem[1]
    #     sorted_eigenvalues = sorted(eigenvalues)

    #     f = sorted_eigenvalues[1]
    #     fv = eigenvectors[list(eigenvalues).index(f)]
    #     return f, fv

    # def get_components(
    #     self,
    # ):
    #     """
    #     Generates a list of network objects for each component in the parent Network class.

    #     Returns
    #     -------
    #     subnetworks : [Network]
    #         List of network objects for each component in the parent Network class.

    #     """
    #     size, components = self.connected_component_nodes()

    #     subnetworks = []
    #     for component in components:
    #         subnetwork = self.construct_subnetwork(component)
    #         subnetworks.append(subnetwork)

    #     return subnetworks

    # def connected_components(
    #     self,
    # ):
    #     """
    #     Returns number of connected components of network
    #     """
    #     L = self.laplacian_matrix()
    #     ns = null_space(L)
    #     return ns.shape[1]

    # def network_parameters(
    #     self,
    # ):
    #     (
    #         node_degrees,
    #         internal_node_degrees,
    #         exit_node_degrees,
    #     ) = self.degree_distribution(plotflag=False)

    #     parameters = {
    #         "total_nodes": len(self.nodes),
    #         "external_nodes": len(exit_node_degrees),
    #         "internal_nodes": len(internal_node_degrees),
    #         "total_links": len(self.links),
    #         "external_links": len(exit_node_degrees),
    #         "internal_links": len(self.links) - len(exit_node_degrees),
    #         "node_degree_statistics": self.degree_statistics(),
    #         # "node_degree": {"internal": internal_node_degrees,
    #         #                 "external": exit_node_degrees,
    #         #                 "total": node_degrees ,
    #         #                 },
    #         "adjacency_matrix": self.adjacency_matrix(),
    #         "mean_free_path": self.mean_free_path(),
    #         "components": self.connected_components(),
    #     }

    #     return parameters

    # def mean_free_path(
    #     self,
    # ):
    #     """
    #     Returns mean link length excluding exit links
    #     """
    #     distances = [np.nan] * len(self.links)
    #     for index, link in enumerate(self.links):
    #         distances[index] = link.distance

    #     # remove nans
    #     distances_int = [d for d in distances if not np.isnan(d)]
    #     return stats.tmean(distances_int)

    # def degree_distribution(self, plotflag=True, kde=False):
    #     """
    #     Evaluates the distribution of node degrees

    #     Parameters
    #     ----------
    #     plotflag : bool, optional
    #         Set true to plot distributions. The default is True.
    #     kde : TYPE, optional
    #         DESCRIPTION. The default is False.

    #     Returns
    #     -------
    #     node_degrees : list
    #         list of degree for all nodes
    #     internal_node_degrees : list
    #         list of degree for interal nodes
    #     exit_node_degrees : list
    #         list of degree for all exit nodes

    #     """
    #     internal_node_degrees = []
    #     exit_node_degrees = []

    #     for node in self.nodes:
    #         if node.node_type == "internal":
    #             # do we want to check if the node is connected to and exit node and discount those connections?
    #             internal_node_degrees.append(node.degree)
    #         else:
    #             exit_node_degrees.append(node.degree)

    #     node_degrees = internal_node_degrees + exit_node_degrees

    #     if plotflag:
    #         plt.figure()
    #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    #         all_bins = [i for i in range(0, max(node_degrees) + 1)]

    #         sns.distplot(
    #             internal_node_degrees,
    #             ax=ax1,
    #             bins=all_bins,
    #             color="red",
    #             label="internal nodes",
    #             kde=kde,
    #         )
    #         plt.legend()
    #         sns.distplot(
    #             exit_node_degrees,
    #             ax=ax2,
    #             bins=all_bins,
    #             color="green",
    #             label="external nodes",
    #             kde=kde,
    #         )
    #         plt.legend()
    #         sns.distplot(
    #             node_degrees,
    #             ax=ax3,
    #             bins=all_bins,
    #             color="blue",
    #             label="all nodes",
    #             kde=kde,
    #         )
    #         plt.legend()

    #     return node_degrees, internal_node_degrees, exit_node_degrees

    # def degree_statistics(
    #     self,
    # ):
    #     """
    #     Returns dictionary of mean, and standard deviation of node degree distributions

    #     """
    #     (
    #         node_degrees,
    #         internal_node_degrees,
    #         exit_node_degrees,
    #     ) = self.degree_distribution(False)
    #     return {
    #         "internal": {
    #             "mean": stats.tmean(internal_node_degrees),
    #             "std": stats.tstd(internal_node_degrees),
    #             # "median": stats.median(internal_node_degrees),
    #         },
    #         "exit": {
    #             "mean": stats.tmean(exit_node_degrees),
    #             "std": stats.tstd(exit_node_degrees),
    #             # "median": statistics.median(exit_node_degrees),
    #         },
    #         "all": {
    #             "mean": stats.tmean(node_degrees),
    #             "std": stats.tstd(node_degrees),
    #             # "median": statistics.median(node_degrees),
    #         },
    #     }

    # ######################################################
    # # %%  scattering matrix related functions
    # ######################################################

    # def get_S11_P11(self):
    #     """
    #     Returns S11 and P11 matrices
    #     -------
    #     S11,P11 : numpy array
    #         Blocks used in calculation of scattering matrix, or isolated network resonances

    #     """
    #     # get list of node IDs tuples and associated node type
    #     IDs = [0] * self.total_nodes
    #     for index, node in enumerate(self.nodes):
    #         IDs[index] = node.number

    #     # sort nodes
    #     sorted_nodes = sorted(IDs)

    #     # construct big S matrix
    #     total_degree = sum(
    #         [
    #             node.degree
    #             for node in self.nodes
    #             if node.node_type == "internal"
    #         ]
    #     )
    #     Sdim = total_degree if self.scat_loss == 0 else 2 * total_degree
    #     S = np.zeros(shape=(Sdim, Sdim), dtype=np.complex_)

    #     Scoeff_inds = []
    #     Pcoeff_inds = []
    #     Scoeff_type = []  # vector of flags 0: internal a/b coeff,
    #     #                 2: c/d loss node coeff
    #     Pcoeff_type = []  # vector of flags 0: internal a/b coeff,
    #     #                 1: u/v exit node coeff

    #     sindex = 0
    #     for nodeid in sorted_nodes:
    #         node = self.get_node(nodeid)
    #         if node.node_type == "internal":
    #             sdim = node.S_mat.shape[0]
    #             for i in node.sorted_connected_nodes:
    #                 # connected_node = self.get_node(i)
    #                 # connected_node_type = connected_node.node_type
    #                 Scoeff_type.append(0)
    #                 Pcoeff_type.append(0)

    #                 Scoeff_inds.append((node.number, i))
    #                 Pcoeff_inds.append((node.number, i))

    #             if self.scat_loss != 0:
    #                 for i in node.sorted_connected_nodes:
    #                     Scoeff_inds.append((node.number, i))
    #                     Scoeff_type.append(2)

    #             S[sindex : sindex + sdim, sindex : sindex + sdim] = node.S_mat
    #             sindex += sdim

    #         elif node.node_type == "exit":
    #             for i in node.sorted_connected_nodes:
    #                 Pcoeff_inds.append((node.number, i))
    #                 Pcoeff_type.append(1)

    #     # construct small S blocks
    #     s11filter = np.array(
    #         np.zeros(
    #             shape=S.shape,
    #         ),
    #         dtype="bool",
    #     )
    #     for index1 in range(0, len(Scoeff_inds)):
    #         for index2 in range(0, len(Scoeff_inds)):
    #             if (Scoeff_type[index1] == 0) and (Scoeff_type[index2] == 0):
    #                 s11filter[index1, index2] = True

    #     nab = sum(np.array(Scoeff_type) == 0)
    #     # ncd = sum(np.array(Scoeff_type) != 0)
    #     S11 = np.extract(s11filter, S).reshape((nab, nab))

    #     # construct big P matrix
    #     P = np.zeros(
    #         shape=(len(Pcoeff_inds), len(Pcoeff_inds)), dtype=np.complex_
    #     )
    #     for index1, (i, j) in enumerate(Pcoeff_inds):
    #         # node1 = self.get_node(i)
    #         # node2 = self.get_node(j)
    #         # distance = self.calculate_distance(node1.position,node2.position)
    #         link = self.get_link(i, j)
    #         index2 = Pcoeff_inds.index((j, i))
    #         P[index1, index2] = np.exp(1j * link.k * link.distance)
    #     p11filter = np.array(
    #         np.zeros(
    #             shape=P.shape,
    #         ),
    #         dtype="bool",
    #     )

    #     sm_sorted_nodes = []
    #     for index1 in range(0, len(Pcoeff_inds)):
    #         if Pcoeff_type[index1] != 0:
    #             sm_sorted_nodes.append(Pcoeff_inds[index1][0])

    #         for index2 in range(0, len(Pcoeff_inds)):
    #             if (Pcoeff_type[index1] == 0) and (Pcoeff_type[index2] == 0):
    #                 p11filter[index1, index2] = True
    #     mab = sum(np.array(Pcoeff_type) == 0)
    #     # muv = sum(np.array(Pcoeff_type) != 0)
    #     P11 = np.extract(p11filter, P).reshape((mab, mab))

    #     return S11, P11

    # def generate_SP_matrices(self, inverse=False):
    #     """
    #     Calculates the S and P block matrices used for direct computation of the
    #     scattering matrix of the network

    #     Returns
    #     -------
    #     S11,S12,S21,S22,P11,P12,P21,P22 : numpy arrays for blocks of S and P
    #         matrices

    #     """
    #     # get list of node IDs tuples and associated node type
    #     IDs = [0] * self.total_nodes
    #     for index, node in enumerate(self.nodes):
    #         IDs[index] = node.number

    #     # sort nodes
    #     sorted_nodes = sorted(IDs)

    #     # construct big S matrix
    #     total_degree = sum(
    #         [
    #             node.degree
    #             for node in self.nodes
    #             if node.node_type == "internal"
    #         ]
    #     )
    #     Sdim = total_degree if self.scat_loss == 0 else 2 * total_degree
    #     S = np.zeros(shape=(Sdim, Sdim), dtype=np.complex_)

    #     Scoeff_inds = []
    #     Pcoeff_inds = []
    #     Scoeff_type = []  # vector of flags 0: internal a/b coeff,
    #     #                 2: c/d loss node coeff
    #     Pcoeff_type = []  # vector of flags 0: internal a/b coeff,
    #     #                 1: u/v exit node coeff

    #     sindex = 0
    #     for nodeid in sorted_nodes:
    #         node = self.get_node(nodeid)
    #         if node.node_type == "internal":
    #             sdim = node.S_mat.shape[0]
    #             for i in node.sorted_connected_nodes:
    #                 # connected_node = self.get_node(i)
    #                 # connected_node_type = connected_node.node_type
    #                 Scoeff_type.append(0)
    #                 Pcoeff_type.append(0)

    #                 Scoeff_inds.append((node.number, i))
    #                 Pcoeff_inds.append((node.number, i))

    #             if self.scat_loss != 0:
    #                 for i in node.sorted_connected_nodes:
    #                     Scoeff_inds.append((node.number, i))
    #                     Scoeff_type.append(2)

    #             if inverse:
    #                 S[
    #                     sindex : sindex + sdim, sindex : sindex + sdim
    #                 ] = node.iS_mat
    #             else:
    #                 S[
    #                     sindex : sindex + sdim, sindex : sindex + sdim
    #                 ] = node.S_mat
    #             sindex += sdim

    #         elif node.node_type == "exit":
    #             for i in node.sorted_connected_nodes:
    #                 Pcoeff_inds.append((node.number, i))
    #                 Pcoeff_type.append(1)

    #     # construct small S blocks
    #     s11filter = np.array(
    #         np.zeros(
    #             shape=S.shape,
    #         ),
    #         dtype="bool",
    #     )
    #     s12filter = np.array(
    #         np.zeros(
    #             shape=S.shape,
    #         ),
    #         dtype="bool",
    #     )
    #     s21filter = np.array(
    #         np.zeros(
    #             shape=S.shape,
    #         ),
    #         dtype="bool",
    #     )
    #     s22filter = np.array(
    #         np.zeros(
    #             shape=S.shape,
    #         ),
    #         dtype="bool",
    #     )
    #     for index1 in range(0, len(Scoeff_inds)):
    #         for index2 in range(0, len(Scoeff_inds)):
    #             if (Scoeff_type[index1] == 0) and (Scoeff_type[index2] == 0):
    #                 s11filter[index1, index2] = True
    #             elif (Scoeff_type[index1] == 0) and (Scoeff_type[index2] != 0):
    #                 s12filter[index1, index2] = True
    #             elif (Scoeff_type[index1] != 0) and (Scoeff_type[index2] == 0):
    #                 s21filter[index1, index2] = True
    #             elif (Scoeff_type[index1] != 0) and (Scoeff_type[index2] != 0):
    #                 s22filter[index1, index2] = True
    #     nab = sum(np.array(Scoeff_type) == 0)
    #     ncd = sum(np.array(Scoeff_type) != 0)
    #     S11 = np.extract(s11filter, S).reshape((nab, nab))
    #     S12 = np.extract(s12filter, S).reshape((nab, ncd))
    #     S21 = np.extract(s21filter, S).reshape((ncd, nab))
    #     S22 = np.extract(s22filter, S).reshape((ncd, ncd))

    #     # construct big P matrix
    #     P = np.zeros(
    #         shape=(len(Pcoeff_inds), len(Pcoeff_inds)), dtype=np.complex_
    #     )
    #     for index1, (i, j) in enumerate(Pcoeff_inds):
    #         # node1 = self.get_node(i)
    #         # node2 = self.get_node(j)
    #         # distance = self.calculate_distance(node1.position,node2.position)
    #         link = self.get_link(i, j)
    #         index2 = Pcoeff_inds.index((j, i))
    #         if inverse:
    #             P[index1, index2] = np.exp(-1j * link.k * link.distance)
    #         else:
    #             P[index1, index2] = np.exp(1j * link.k * link.distance)

    #         if not np.isfinite(P[index1, index2]):
    #             print(
    #                 "Warning: exponentials in propagation matrix are overflowing."
    #             )
    #     p11filter = np.array(
    #         np.zeros(
    #             shape=P.shape,
    #         ),
    #         dtype="bool",
    #     )
    #     p12filter = np.array(
    #         np.zeros(
    #             shape=P.shape,
    #         ),
    #         dtype="bool",
    #     )
    #     p21filter = np.array(
    #         np.zeros(
    #             shape=P.shape,
    #         ),
    #         dtype="bool",
    #     )
    #     p22filter = np.array(
    #         np.zeros(
    #             shape=P.shape,
    #         ),
    #         dtype="bool",
    #     )

    #     sm_sorted_nodes = []
    #     for index1 in range(0, len(Pcoeff_inds)):
    #         if Pcoeff_type[index1] != 0:
    #             sm_sorted_nodes.append(Pcoeff_inds[index1][0])

    #         for index2 in range(0, len(Pcoeff_inds)):
    #             if (Pcoeff_type[index1] == 0) and (Pcoeff_type[index2] == 0):
    #                 p11filter[index1, index2] = True
    #             elif (Pcoeff_type[index1] == 0) and (Pcoeff_type[index2] != 0):
    #                 p12filter[index1, index2] = True
    #             elif (Pcoeff_type[index1] != 0) and (Pcoeff_type[index2] == 0):
    #                 p21filter[index1, index2] = True
    #             elif (Pcoeff_type[index1] != 0) and (Pcoeff_type[index2] != 0):
    #                 p22filter[index1, index2] = True
    #     mab = sum(np.array(Pcoeff_type) == 0)
    #     muv = sum(np.array(Pcoeff_type) != 0)
    #     P11 = np.extract(p11filter, P).reshape((mab, mab))
    #     P12 = np.extract(p12filter, P).reshape((mab, muv))
    #     P21 = np.extract(p21filter, P).reshape((muv, mab))
    #     P22 = np.extract(p22filter, P).reshape((muv, muv))

    #     return S11, S12, S21, S22, P11, P12, P21, P22, sm_sorted_nodes

    # def scattering_matrix_direct(self):
    #     """
    #     Calculates the scattering matrix of the network using direct solution of the matrix transport equations

    #     Returns
    #     -------
    #     scattering_matrix : numpy array
    #         Scattering matrix of network.
    #     sm_sorted_nodes : [int]
    #         list of node IDs matching order of scattering matrix

    #     """
    #     (
    #         S11,
    #         S12,
    #         S21,
    #         S22,
    #         P11,
    #         P12,
    #         P21,
    #         P22,
    #         sm_sorted_nodes,
    #     ) = self.generate_SP_matrices()

    #     invfac = np.linalg.inv(np.identity(S11.shape[0]) - S11 @ P11)
    #     scattering_matrix = (P21 @ (invfac @ (S11 @ P12))) + P22
    #     # print("P22 is zero: {}".format(np.all(np.abs(P22) == 0.)))
    #     # P22 is only full zero for fully connected nodes

    #     self.scattering_matrix = scattering_matrix
    #     self.sm_node_order = sm_sorted_nodes

    #     return scattering_matrix, sm_sorted_nodes

    # def inverse_scattering_matrix_direct(self):
    #     """
    #     Calculates the inverse scattering matrix of the network using direct solution of the matrix transport equations

    #     Returns
    #     -------
    #     inverse_scattering_matrix : numpy array
    #         Inverse scattering matrix of network.
    #     sm_sorted_nodes : [int]
    #         list of node IDs matching order of scattering matrix

    #     """
    #     (
    #         S11,
    #         S12,
    #         S21,
    #         S22,
    #         P11,
    #         P12,
    #         P21,
    #         P22,
    #         sm_sorted_nodes,
    #     ) = self.generate_SP_matrices(inverse=True)
    #     iS22 = np.linalg.inv(S22)
    #     Q = S11 - (S12 @ (iS22 @ S21))  # schur complement

    #     # iQ = np.linalg.inv(Q)
    #     # iSt11 = iQ
    #     # iSt12 = - iQ @ (S12 @ iS22)
    #     # iSt21 = - iS22 @ (S21 @ iQ)
    #     # fac = S21 @ (iQ @ (S12 @ iS22))
    #     # iSt22 = iS22 @ (np.identity(fac.shape) + fac)

    #     # # block inverse of P
    #     # fac2 = np.linalg.inv(P21 @ (iP11 @ P12))
    #     # Id = np.identity(P22.shape)
    #     # iAB = -iP11 @ P12
    #     # CiA = -P21 @ iP11
    #     # iPt11 = iP11 - (iAB) @ fac2 @ (CiA)
    #     # iPt12 = - iAB @ fac2
    #     # iPt21 = - fac2 @ CiA
    #     # iPt22 = - fac2

    #     # invfac = np.linalg.inv( np.identity(iSt11.shape[0]) - iSt11 @ iPt11   )
    #     # inverse_scattering_matrix =  (iPt21 @ (invfac @ (iSt11 @ iPt12)))  +  iPt22

    #     invfac = np.linalg.inv(np.identity(S11.shape[0]) - Q @ P11)
    #     inverse_scattering_matrix = (P21 @ (invfac @ (Q @ P12))) + P22

    #     self.inverse_scattering_matrix = inverse_scattering_matrix
    #     self.sm_node_order = sm_sorted_nodes

    #     return inverse_scattering_matrix, sm_sorted_nodes

    # def scattering_matrix_recursive(
    #     self, n_iterations=1000, converge=True, period=200, threshold=0.01
    # ):
    #     """
    #     Calculates the scattering matrix of the network using a forward recursive solution of the matrix
    #     transport equations

    #     Parameters
    #     ----------
    #     n_iterations : int, optional
    #         Number of maximum iterations. The default is 1000.
    #     converge : bool, optional
    #         Flag to use convergence check to exit recursive algorithm early.
    #         The default is True.
    #     period : int, optional
    #         Number of iterations between convergence is checked. The default is 200.
    #     threshold : float, optional
    #         Convergence tolerance parameter. Dictates threshold of the standard deviation in the
    #         total fractional change in
    #         the total output power from network.
    #         The default is 0.01.

    #         NB/ This is a legacy convergence test. Better test to implement would be
    #         based on amplitudes at all nodes

    #     Returns
    #     -------
    #     scattering_matrix : numpy array
    #         Scattering matrix of network.
    #     node_order : [int]
    #         list of node IDs matching order of scattering matrix

    #     """
    #     scattering_matrix = np.zeros(
    #         shape=(self.exit_nodes, self.exit_nodes), dtype=np.complex_
    #     )

    #     # excite each input node individually with a wave of unitary amplitude
    #     # and store the output wave amplitude into the correct column of the
    #     # scattering matrix of the entire system
    #     node_order = None
    #     for i in range(self.exit_nodes):
    #         update_progress(
    #             i / self.exit_nodes,
    #             "Calculating SM for I/O node {}...".format(i),
    #         )

    #         exit_out = [0] * self.exit_nodes
    #         exit_out[i] = 1

    #         index1 = 0
    #         self.reset_network(input_amp=exit_out)
    #         self.run_network(n_iterations, converge, period, threshold)

    #         node_order = []
    #         for nodes in self.nodes:
    #             if nodes.node_type == "exit":
    #                 node_order.append(nodes.number)
    #                 for b in nodes.inwave:
    #                     scattering_matrix[index1, i] = nodes.inwave[b]

    #                 index1 += 1

    #     self.scattering_matrix = scattering_matrix
    #     self.sm_node_order = node_order

    #     return scattering_matrix, node_order

    # def inverse_scattering_matrix_recursive(
    #     self, n_iterations=1000, converge=True, period=200, threshold=0.01
    # ):
    #     """
    #     Calculates the inverse scattering matrix of the network using a backward recursive solution of the
    #     matrix transport equations

    #     Parameters
    #     ----------
    #     n_iterations : int, optional
    #         Number of maximum iterations. The default is 1000.
    #     converge : bool, optional
    #         Flag to use convergence check to exit recursive algorithm early.
    #         The default is True.
    #     period : int, optional
    #         Number of iterations between convergence is checked. The default is 200.
    #     threshold : float, optional
    #         Convergence tolerance parameter. Dictates threshold for fractional change in
    #         the total input power from network.
    #         The default is 0.01.

    #         NB/ This is a legacy convergence test. Better test to implement would be
    #             based on amplitudes at all nodes

    #     Returns
    #     -------
    #     inverse_scattering_matrix : numpy array
    #         Inverse of scattering matrix of network.
    #     node_order : [int]
    #         list of node IDs matching order of scattering matrix

    #     """

    #     inverse_scattering_matrix = np.zeros(
    #         shape=(self.exit_nodes, self.exit_nodes), dtype=np.complex_
    #     )

    #     # consider each outpt node individually with a wave of unitary amplitude
    #     # and store the corresponding input  wave amplitude into the correct column of the
    #     # inverse scattering matrix of the entire system
    #     node_order = None
    #     for i in range(self.exit_nodes):
    #         update_progress(
    #             i / self.exit_nodes,
    #             "Calculating SM for I/O node {}...".format(i),
    #         )
    #         exit_out = [0] * self.exit_nodes
    #         exit_out[i] = 1

    #         index1 = 0
    #         self.reset_network(output_amp=exit_out)
    #         self.run_network(
    #             n_iterations,
    #             converge,
    #             period,
    #             threshold,
    #             direction="backwards",
    #         )

    #         node_order = []
    #         for nodes in self.nodes:
    #             if nodes.node_type == "exit":
    #                 node_order.append(nodes.number)
    #                 for b in nodes.outwave:
    #                     inverse_scattering_matrix[index1, i] = nodes.outwave[b]

    #                 index1 += 1

    #     self.scattering_matrix = np.linalg.inv(inverse_scattering_matrix)
    #     self.sm_node_order = node_order
    #     return inverse_scattering_matrix, node_order

    # ##########################
    # # %% Save/Load Functions
    # ##########################

    # def save_network(self, filename):
    #     networkdict = self.network_to_dict()
    #     save_dict_to_hdf5(networkdict, filename)

    #     # with open(filename, 'w+') as f:
    #     #     dump(networkdict, f, indent=4)

    # def load_network(self, filename):
    #     networkdict = load_dict_from_hdf5(filename)
    #     self.dict_to_network(networkdict)

    # def network_to_dict(
    #     self,
    # ):
    #     # save nodes
    #     allnodes = {
    #         i: node.node_to_dict() for i, node in enumerate(self.nodes)
    #     }

    #     # save links
    #     alllinks = {
    #         i: link.link_to_dict() for i, link in enumerate(self.links)
    #     }

    #     # save other network properties
    #     varnames = self.get_default_properties().keys()

    #     networkprops = dict(
    #         (v, eval("self." + v)) for v in varnames if hasattr(self, v)
    #     )

    #     # store exit node data separately
    #     exitpos = [
    #         node.position for node in self.nodes if node.node_type == "exit"
    #     ]
    #     exitids = self.get_exit_node_ids()

    #     networkprops.update(
    #         {
    #             "exit_ids": exitids,
    #             "exit_positions": exitpos,
    #         }
    #     )

    #     networkdict = {
    #         "NODES": allnodes,
    #         "LINKS": alllinks,
    #         "NETWORK": networkprops,
    #     }

    #     return networkdict

    # def dict_to_network(self, networkdict):
    #     self.nodes = []
    #     self.links = []
    #     self.node_indices = []
    #     self.nodenumber_indices = {}

    #     # reset all network attributes
    #     networkprops = networkdict["NETWORK"]
    #     for v, d in self.get_default_properties().items():
    #         if hasattr(self, v):
    #             setattr(self, v, d)

    #     for key, val in networkprops.items():
    #         setattr(self, key, val)

    #     # load nodes
    #     node_dict = networkdict["NODES"]
    #     for js in node_dict.values():
    #         self.add_node(node_dict=js)

    #     # load links
    #     linkdict = networkdict["LINKS"]
    #     for js in linkdict.values():
    #         self.add_connection(linkdict=js)

    #     self.count_nodes()
    #     self.connect_nodes()

    # @staticmethod
    # def get_default_properties() -> dict:
    #     default_values = {
    #         "scat_loss": 0,  # parameter describing fractional scattering loss
    #         "input": None,  # array of input wave amplitudes
    #         "output": None,  # array of output wave amplitudes
    #         "k": 1.0,  # vacuum wavenumber of wave propagating in link
    #         "n": 1.0,  # effective refractive index of link
    #         "total_nodes": 0,  # total number of nodes in network
    #         "internal_nodes": 0,  # number of internal nodes in network
    #         "exit_nodes": 0,  # number of exit nodes in network
    #         "scattering_matrix": None,  # network scattering matrix
    #         "sm_node_order": None,  # list of exit node ids corresponding to SM order
    #         "network_spec": None,  # dictionary specifying network properties
    #         "network_type": None,  # type of network
    #         "node_spec": None,  # dictionary specifying node properties
    #         "seed_number": 0,  # seed number used for network generation
    #     }

    #     return default_values

    # ##########################
    # # %% Plotting Functions
    # ##########################

    def plot(self, plot_mode: str = "", show_indices=False) -> None:
        """Draw network"""
        fig, ax = plt.subplots()

        # Plot links
        for link in self.links:
            node_1_index, node_2_index = link.node_indices
            node_1_pos = self.get_node(node_1_index).position
            node_2_pos = self.get_node(node_2_index).position
            link.plot(ax, node_1_pos, node_2_pos, show_indices)

        # Plot nodes
        for node in self.nodes:
            node.plot(ax, show_indices)
