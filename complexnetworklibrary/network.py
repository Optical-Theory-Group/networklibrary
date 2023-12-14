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

import logconfig

from ._dict_hdf5 import load_dict_from_hdf5, save_dict_to_hdf5
from ._generator import NetworkGenerator
from complexnetworklibrary import network_factory
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
    # Scattering methods
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

    # -------------------------------------------------------------------------
    # Plotting methods
    # -------------------------------------------------------------------------

    def draw(self, show_indices=False) -> None:
        """Draw network"""
        fig, ax = plt.subplots()
        ax.set_aspect("equal")

        # Plot links
        for link in self.links:
            node_1_index, node_2_index = link.node_indices
            node_1_pos = self.get_node(node_1_index).position
            node_2_pos = self.get_node(node_2_index).position
            link.draw(ax, node_1_pos, node_2_pos, show_indices)

        # Plot nodes
        for node in self.nodes:
            node.draw(ax, show_indices)

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
        ax.set_aspect("equal")
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

            ax.arrow(
                cx,
                cy,
                dx,
                dy,
                lw=0.0,
                head_width=0.05,
                head_length=0.05,
                length_includes_head=False,
                color=color,
            )

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
