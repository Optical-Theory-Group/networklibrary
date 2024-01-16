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
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from scipy import optimize
from scipy.linalg import dft, null_space
from matplotlib.patches import FancyArrowPatch
import logconfig

from ._dict_hdf5 import load_dict_from_hdf5, save_dict_to_hdf5
from ._generator import NetworkGenerator
from complex_network import network_factory
from complex_network.components.link import Link
from complex_network.components.node import Node
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

    # -------------------------------------------------------------------------
    # Basic network properties
    # -------------------------------------------------------------------------

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
        return len(list(self.links))

    @property
    def exit_links(self):
        return [link for link in self.links if link.link_type == "exit"]

    @property
    def num_exit_links(self):
        return list(self.exit_links)

    @property
    def internal_links(self):
        return [link for link in self.links if link.link_type == "internal"]

    @property
    def num_internal_links(self):
        return len(list(self.internal_links))

    @property
    def connections(self):
        """Alias for links"""
        return self.links

    # -------------------------------------------------------------------------
    # Network matrix properties
    # -------------------------------------------------------------------------

    @property
    def S_mat(self):
        """The scattering matrix for the exit ports"""
        if hasattr(self, "_S_mat") and self._S_mat is not None:
            return self._S_mat
        else:
            return self.get_S_matrix_direct()

    @S_mat.setter
    def S_mat(self, value: np.ndarray) -> None:
        self._S_mat = value

    @property
    def network_step_matrix(self):
        """The network matrix for one step of iteration"""
        if (
            hasattr(self, "_network_step_matrix")
            and self._network_step_matrix is not None
        ):
            return self._network_step_matrix
        else:
            return self.get_network_step_matrix()

    @property
    def network_matrix(self):
        """The full network matrix"""
        if (
            hasattr(self, "_network_matrix")
            and self._network_matrix is not None
        ):
            return self._network_matrix
        else:
            return self.get_network_matrix()

    # -------------------------------------------------------------------------
    # Basic utility functions
    # -------------------------------------------------------------------------

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
    #  Direct scattering methods
    # -------------------------------------------------------------------------

    def scatter_direct(
        self,
        incident_field: np.ndarray,
        direction: str = "forward",
    ) -> None:
        """Scatter the incident field through the network using the
        network matrix"""

        # Set up the matrix product
        network_matrix = self.network_matrix
        num_exits = self.num_exit_nodes
        incident_vector = np.zeros((len(network_matrix)), dtype=np.complex128)
        incident_vector[num_exits : 2 * num_exits] = incident_field
        outgoing_vector = network_matrix @ incident_vector

        # Reset fields throughout the network and set incident field
        self.reset_fields()
        self.set_network_fields(outgoing_vector)

    def set_network_fields(self, vector: np.ndarray) -> None:
        exit_vector_length = self.num_exit_nodes
        internal_vector_length = 0
        for node in self.internal_nodes:
            internal_vector_length += node.degree

        outgoing_exit = vector[0:exit_vector_length]
        incoming_exit = vector[exit_vector_length : 2 * exit_vector_length]
        outgoing_internal = vector[
            2 * exit_vector_length : 2 * exit_vector_length
            + internal_vector_length
        ]
        incoming_internal = vector[
            2 * exit_vector_length + internal_vector_length :
        ]

        self.set_incident_field(incoming_exit)

        count = 0
        for node in self.exit_nodes:
            # Set outgoing exit values
            value = outgoing_exit[count]
            node_index = node.index
            connected_link_index = node.sorted_connected_links[0]
            connected_link = self.get_link(connected_link_index)

            node.outwave["-1"] = value
            node.outwave_np[0] = value
            node.inwave[str(connected_link)] = value
            node.inwave[1] = value

            connected_link.outwave[str(node_index)] = value
            connected_link.outwave_np[1] = value

            # Set incoming exit values
            value = incoming_exit[count]
            node_index = node.index
            connected_link_index = node.sorted_connected_links[0]
            connected_link = self.get_link(connected_link_index)

            node.inwave["-1"] = value
            node.inwave_np[0] = value
            node.outwave[str(connected_link)] = value
            node.outwave[1] = value

            connected_link.inwave[str(node_index)] = value
            connected_link.inwave_np[1] = value

            count += 1

        # Set internal node values
        count = 0
        for node in self.internal_nodes:
            for i, connected_index in enumerate(node.sorted_connected_nodes):
                incoming_value = incoming_internal[count]
                node.inwave[str(connected_index)] = incoming_value
                node.inwave_np[i] = incoming_value

                outgoing_value = outgoing_internal[count]
                node.outwave[str(connected_index)] = outgoing_value
                node.outwave_np[i] = outgoing_value

                count += 1

        # Set internal link values
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            node_one = self.get_node(node_one_index)
            node_two = self.get_node(node_two_index)

            # Set link fields
            link.inwave[str(node_one_index)] = node_one.outwave[
                str(node_two_index)
            ]
            link.inwave_np[0] = node_one.outwave[str(node_two_index)]
            link.inwave[str(node_two_index)] = node_two.outwave[
                str(node_one_index)
            ]
            link.inwave_np[1] = node_two.outwave[str(node_one_index)]

            # Outwaves
            link.outwave[str(node_one_index)] = node_one.inwave[
                str(node_two_index)
            ]
            link.outwave_np[0] = node_one.inwave[str(node_two_index)]
            link.outwave[str(node_two_index)] = node_two.inwave[
                str(node_one_index)
            ]
            link.outwave_np[1] = node_two.inwave[str(node_one_index)]

        # Remaining exit links values
        for link in self.exit_links:
            exit_index, node_index = link.node_indices
            node = self.get_node(node_index)

            # Set link fields
            link.inwave[str(node_index)] = node.outwave[str(exit_index)]
            link.inwave_np[1] = node.outwave[str(exit_index)]

            link.outwave[str(node_index)] = node.inwave[str(exit_index)]
            link.outwave_np[1] = node.inwave[str(exit_index)]

        self.update_outgoing_fields()

    def get_S_matrix_direct(
        self,
        n: float | complex | None = None,
        k0: float | complex | None = None,
    ) -> np.ndarray:
        """Calculate the network scattering matrix directly"""
        # Update network with given wave parameters
        self.update_link_S_matrices(n, k0)

        network_matrix = self.get_network_matrix()
        num_exit_nodes = self.num_exit_nodes
        S_exit = network_matrix[
            0:num_exit_nodes, num_exit_nodes : 2 * num_exit_nodes
        ]
        self._S_mat = S_exit
        return S_exit

    def get_network_matrix(self) -> np.ndarray:
        """Get the 'infinite' order network matrix"""
        step_matrix = self.get_network_step_matrix()
        lam, v = np.linalg.eig(step_matrix)
        modified_lam = np.where(np.isclose(lam, 1.0 + 0.0 * 1j), lam, 0.0)
        rebuilt = v @ np.diag(modified_lam) @ np.linalg.inv(v)
        self._network_matrix = rebuilt
        return rebuilt

    def get_network_step_matrix(self) -> np.ndarray:
        """The network matrix satisfies

        (O_e)       (0 0     |P_e    0)(O_e)
        (I_e)       (0 1     |0      0)(I_e)
        (---)   =   (-----------------)(---)
        (O_i)       (0 S*P_e | S*P_i 0)(O_i)
        (I_i)_n+1   (0 P_e   | P_i   0)(I_i)_n
        """

        exit_vector_length = self.num_exit_nodes
        internal_vector_length = 0
        for node in self.internal_nodes:
            internal_vector_length += node.degree

        # Maps for dealing with positoins of matrix elements
        (
            internal_scattering_map,
            internal_scattering_slices,
            exit_scattering_map,
        ) = self._get_network_matrix_maps()

        # Get the internal S
        internal_S = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        for node in self.internal_nodes:
            node_index = node.index
            node_S_mat = node.S_mat
            new_slice = internal_scattering_slices[str(node_index)]
            internal_S[new_slice, new_slice] = node_S_mat

        # Get internal P
        internal_P = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            # Wave that is going into node_one
            row = internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor
            # Wave propagating the other way
            internal_P[col, row] = phase_factor

        # Get exit P
        exit_P = np.zeros(
            (exit_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.exit_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]
            row = exit_scattering_map[f"{str(node_two_index)}"]
            col = internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            exit_P[row, col] = phase_factor

        # Build up network matrix
        # First define all the zero matrices to keep things simpler
        z_ee = np.zeros(
            (exit_vector_length, exit_vector_length),
            dtype=np.complex128,
        )
        z_long = np.zeros(
            (exit_vector_length, internal_vector_length), dtype=np.complex128
        )
        z_tall = z_long.T
        z_ii = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        identity = np.identity(exit_vector_length, dtype=np.complex128)

        network_step_matrix = np.block(
            [
                [z_ee, z_ee, exit_P, z_long],
                [z_ee, identity, z_long, z_long],
                [z_tall, internal_S @ exit_P.T, internal_S @ internal_P, z_ii],
                [z_tall, exit_P.T, internal_P, z_ii],
            ]
        )
        self._network_step_matrix = network_step_matrix
        return network_step_matrix

    def _get_network_matrix_maps(
        self,
    ) -> tuple[dict[str, int], dict[str, slice], dict[str, int]]:
        internal_scattering_slices = {}
        internal_scattering_map = {}
        exit_scattering_map = {}
        i = 0

        for node in self.internal_nodes:
            # Update the map. Loop through connected nodes and work out the
            # indices
            start = i
            node_index = node.index
            for new_index in node.sorted_connected_nodes:
                internal_scattering_map[f"{node_index},{new_index}"] = i
                i += 1
            end = i
            internal_scattering_slices[f"{node_index}"] = slice(start, end)

        i = 0
        for node in self.exit_nodes:
            exit_scattering_map[f"{node.index}"] = i
            i += 1

        return (
            internal_scattering_map,
            internal_scattering_slices,
            exit_scattering_map,
        )

    # -------------------------------------------------------------------------
    #  Methods for altering/perturbing the network
    # -------------------------------------------------------------------------

    def set_node_S_matrix(
        self,
        node_index: int,
        S_mat_type: str,
        S_mat_params: dict[str, Any] | None = None,
    ) -> None:
        """Replace the node scattering matrix with a new one as defined by
        the arguments"""
        if S_mat_params is None:
            S_mat_params = {}

        node = self.get_node(node_index)
        size, _ = node.S_mat.shape
        node.S_mat = network_factory.get_S_mat(S_mat_type, size, S_mat_params)
        node.iS_mat = np.linalg.inv(node.S_mat)

    def random_scaled_node_perturbation(
        self, node_index: int, variance: float = 0.01
    ) -> None:
        """Perturb a node's S matrix by multiplying it by a small perturbation
        of the identity matrix"""
        node = self.get_node(node_index)
        S_mat = node.S_mat
        size, _ = S_mat.shape

        is_symmetric = np.allclose(S_mat - S_mat.T, 0.0)
        is_unitary = np.allclose(
            S_mat @ np.conj(S_mat.T) - np.identity(size, dtype=np.complex128),
            0.0,
        )

        # Calculate M = I + dS
        dS = (
            np.random.randn(size, size) + 1j * np.random.randn(size, size)
        ) * np.sqrt(variance)
        M = np.identity(size, dtype=np.complex128) + dS
        new_S_mat = S_mat @ M

        # Symmetrise new S if necessary
        if is_symmetric:
            new_S_mat = (new_S_mat + new_S_mat.T) / 2
        if is_unitary:
            U, s, Vh = np.linalg.svd(new_S_mat)
            new_S_mat = U @ Vh

        node.S_mat = new_S_mat
        node.iS_mat = np.linalg.inv(new_S_mat)

    # -------------------------------------------------------------------------
    # Iterative scattering methods
    # -------------------------------------------------------------------------

    def get_S_matrix_iterative(
        self,
        n: float | complex | None = None,
        k0: float | complex | None = None,
        direction: str = "forward",
        max_num_steps: int = 10**5,
        tolerance: float = 1e-5,
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

            print(f"Scattering {incident_field}")
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
        incident_field: np.ndarray,
        direction: str = "forward",
        max_num_steps: int = 10**5,
        tolerance: float = 1e-5,
        verbose: bool = False,
    ) -> np.ndarray:
        """Scatter the incident field through the network"""
        # Reset fields throughout the network and set incident field
        self.reset_fields()
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

            # Give the code a couple of simulations so that it actually reaches
            # the exit
            if i <= 5:
                continue

            # Check for convergence
            diff = np.linalg.norm(after - before)
            if verbose:
                frac = min(np.log10(diff) / np.log10(tolerance), 1.0)
                conv_pbar.update(frac - conv_pbar.n)

            if diff <= tolerance:
                has_converged = True
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
        """Perform one step of scattering throughout the network.

        This involves scattering once at the nodes and once in the links."""

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

    # -------------------------------------------------------------------------
    # Plotting methods
    # -------------------------------------------------------------------------

    def draw(
        self,
        show_indices: bool = False,
        show_exit_indices: bool = False,
        equal_aspect: bool = False,
    ) -> None:
        """Draw network"""
        fig, ax = plt.subplots()
        if equal_aspect:
            ax.set_aspect("equal")

        # Plot links
        for link in self.links:
            node_1_index, node_2_index = link.node_indices
            node_1_pos = self.get_node(node_1_index).position
            node_2_pos = self.get_node(node_2_index).position
            link.draw(ax, node_1_pos, node_2_pos, show_indices)

        # Plot nodes
        for node in self.nodes:
            node.draw(ax, show_indices, show_exit_indices)

    def plot_fields(
        self,
        highlight_nodes: list[int] | None = None,
        title: str = "Field distribution",
        max_val: float | None = None,
    ) -> None:
        """Show internal field distribution in the network"""
        if highlight_nodes is None:
            highlight_nodes = []

        fig, ax = plt.subplots()
        # ax.set_aspect("equal")
        ax.set_title(title)
        # Get link intensities to define the colormap
        power_diffs = [link.power_diff for link in self.links]
        vmax = max(power_diffs) if max_val is None else max_val
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap = plt.cm.viridis

        # Plot links
        for link in self.links:
            node_1_index, node_2_index = link.node_indices
            node_1_pos = self.get_node(node_1_index).position
            node_2_pos = self.get_node(node_2_index).position
            color = cmap(norm(link.power_diff))
            link.draw(ax, node_1_pos, node_2_pos, color=color)

            length = link.length
            x_1, x_2 = node_1_pos[0], node_2_pos[0]
            y_1, y_2 = node_1_pos[1], node_2_pos[1]
            cx = (x_1 + x_2) / 2
            cy = (y_1 + y_2) / 2
            if link.power_direction == 1:
                dx = x_2 - x_1
                dy = y_2 - y_1
            else:
                dx = x_1 - x_2
                dy = y_1 - y_2
            dx /= 100
            dy /= 100

            ax.quiver(
                cx,
                cy,
                dx,
                dy,
                scale=0.1,
                scale_units="xy",
                angles="xy",
                color=color,
            )

            # arrow = FancyArrowPatch(
            #     (cx, cy),  # Starting point of the arrowhead
            #     (cx + dx, cy + dy),  # Ending point of the arrowhead
            #     arrowstyle='->',  # Arrow style
            #     mutation_scale=1e-6,  # Increase this value for a chunkier arrowhead
            #     color=color,  # Arrow color
            # )

            # # Add the arrow to the axis
            # ax.add_patch(arrow)
            # arrowhead_size = 0.0000005 * length
            # ax.arrow(
            #     cx,
            #     cy,
            #     dx,
            #     dy,
            #     lw=0.0,
            #     head_width=arrowhead_size,
            #     head_length=arrowhead_size,
            #     color=color,
            # )

        # Add the colorbar
        sm = plt.cm.ScalarMappable(norm, cmap)
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical")

        # background color
        bg_color = cmap(0)
        ax.set_facecolor(bg_color)

        # Exit nodes
        for node in self.exit_nodes:
            node.draw(ax, color="white")

        # Highlight nodes
        for node_index in highlight_nodes:
            node = self.get_node(node_index)
            node.draw(ax, color="red")
