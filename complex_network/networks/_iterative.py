"""These are iterative scattering methods that used to belong to network.

These are network class methods and don't make sense in isolation. Don't try to
use them. They are not really of any use anymore since the network calculations
can all be done analytically. This may be deleted in the future. If you want to
use these, paste them back into the network class"""

import numpy as np
from tqdm import tqdm
import warnings

# -------------------------------------------------------------------------
# Iterative scattering methods
# THESE ARE ESSENTIALLY OBSOLETE AT THIS POINT AS EVERYTHING CAN DO BE
# ANALYTICALLY MUCH FASTER. CONSIDER REMOVING?
# -------------------------------------------------------------------------


def get_S_matrix_iterative(
    self,
    k0: float | complex,
    direction: str = "forward",
    max_num_steps: int = 10**5,
    tolerance: float = 1e-5,
    verbose: bool = True,
) -> np.ndarray:
    """Calculate the network scattering matrix iteratively"""
    # Update network with given wave parameters
    self.update_links(k0)

    matrix = np.zeros(
        (self.num_external_nodes, self.num_external_nodes),
        dtype=np.complex128,
    )
    if verbose:
        node_pbar = tqdm(total=self.num_external_nodes, desc="external nodes")

    # Loop over external nodes
    for i in range(self.num_external_nodes):
        incident_field = np.zeros(self.num_external_nodes, dtype=np.complex128)
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
    self._reset_fields()
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
        # the external
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




def get_outgoing_fields(self, direction: str = "forward") -> np.ndarray:
    """Get the current outgoinf field on the basis of the given
    direction"""
    if direction == "forward":
        return self.outwave_np
    else:
        return self.inwave_np




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
            np_one = node_one.sorted_connected_nodes.index(int(node_two_index))
            np_two = node_two.sorted_connected_nodes.index(int(node_one_index))
            node_one.inwave_np[np_one] = node_one.inwave[node_two_index]
            node_two.inwave_np[np_two] = node_two.inwave[node_one_index]

        elif direction == "backward":
            node_one.outwave[node_two_index] = link.inwave[node_one_index]
            node_two.outwave[node_one_index] = link.inwave[node_two_index]

            # Get the indices appropriate for the numpy arrays that
            # correspond to the correct wave
            np_one = node_one.sorted_connected_nodes.index(int(node_two_index))
            np_two = node_two.sorted_connected_nodes.index(int(node_one_index))
            node_one.outwave_np[np_one] = node_one.outwave[node_two_index]
            node_two.outwave_np[np_two] = node_two.outwave[node_one_index]
