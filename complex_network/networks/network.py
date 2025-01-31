"""Main network class module."""

from typing import Any, Callable

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from complex_network.components.link import Link
from complex_network.components.node import Node
from complex_network.scattering_matrices import link_matrix, node_matrix


class Network:
    """Class for storing the nodes and links that form a network.

    The network class also has some additional properties and methods for
    getting nodes and links with particular incides, numbers of different types
    components etc. There are also methods to obtain all of the internal node
    and link scattering matrices that are required for calculating the
    external scattering matrices and other mathematical objects (e.g. Wigner
    Smith operators). The scattering matrix can be calculated directly from the
    network object, but calculations of Wigner Smith operators are in a
    different file to reduce clutter, and since they are related to different
    types of perturbations that are not necessarily intrinsic to the
    network class."""

    def __init__(
        self,
        nodes: dict[int, Node],
        links: dict[int, Link],
        data: dict[str, Any] | None = None,
    ) -> None:
        """Nodes and links are primarily stored in dictionaries whose keys
        are the indices. Don't make a network directly from here; use
        network_factory.py."""
        self._reset_values(data)
        self.node_dict = nodes
        self.link_dict = links
        self._reset_fields()
        self._set_matrix_calc_utils()

    def _reset_values(self, data: dict[str, Any] | None = None) -> None:
        """Set various attributes of the network either to defaults or to
        those in provided in the data parameter.

        This method is mainly just run on init to give the network a bunch of
        default attributes.
        """
        default_values = self._get_default_values()
        if data is not None:
            default_values.update(data)
        for key, value in default_values.items():
            setattr(self, key, value)

    def _reset_fields(self) -> None:
        """Reset the values of the incident and outgoing fields to be zero

        Not particularly useful in isolation, but is used by other methods
        in some scenarios.
        """
        # Reset all node and link values
        for node in self.nodes:
            node.reset_fields()
        for link in self.links:
            link.reset_fields()

        # Set up keys
        for node in self.external_nodes:
            self.inwave[str(node.index)] = 0 + 0j
            self.outwave[str(node.index)] = 0 + 0j

        # Set up np arrays
        self.inwave_np = np.zeros(len(self.inwave.keys()), dtype=np.complex128)
        self.outwave_np = np.zeros(
            len(self.inwave.keys()), dtype=np.complex128
        )

    def _set_matrix_calc_utils(self) -> None:
        """Pre-calculate some useful quantities used in calculating the network
        scattering matrces"""

        internal_vector_length = 0
        for node in self.internal_nodes:
            internal_vector_length += node.degree
        self.internal_vector_length = internal_vector_length

        (
            internal_scattering_map,
            internal_scattering_slices,
            external_scattering_map,
        ) = self._get_network_matrix_maps()
        self.internal_scattering_map = internal_scattering_map
        self.internal_scattering_slices = internal_scattering_slices
        self.external_scattering_map = external_scattering_map

    def _get_network_matrix_maps(
        self,
    ) -> tuple[dict[str, int], dict[str, slice], dict[str, int]]:
        internal_scattering_slices = {}
        internal_scattering_map = {}
        external_scattering_map = {}
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
        for node in self.external_nodes:
            external_scattering_map[f"{node.index}"] = i
            i += 1

        return (
            internal_scattering_map,
            internal_scattering_slices,
            external_scattering_map,
        )

    @staticmethod
    def _get_default_values() -> dict[str, Any]:
        """Default values for the network.

        Used by init to set things up initially, ensuring that certain
        attributes are given default values.
        """
        default_values: dict[str, Any] = {
            "node_dict": {},
            "link_dict": {},
            "inwave": {},
            "outwave": {},
            "inwave_np": np.zeros(0, dtype=np.complex128),
            "outwave_np": np.zeros(0, dtype=np.complex128),
            "get_S": np.zeros(0, dtype=np.complex128),
            "get_S_inv": np.zeros(0, dtype=np.complex128),
        }
        return default_values

    # -------------------------------------------------------------------------
    # Basic network properties
    # -------------------------------------------------------------------------

    @property
    def nodes(self) -> list[Node]:
        """List of all nodes."""
        return list(self.node_dict.values())

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network."""
        return len(self.nodes)

    @property
    def external_nodes(self) -> list[Node]:
        """List of external nodes."""
        return [node for node in self.nodes if node.node_type == "external"]

    @property
    def num_external_nodes(self) -> int:
        """Number of external nodes."""
        return len(self.external_nodes)

    @property
    def external_vector_length(self) -> int:
        """Equivalent to number of external nodes."""
        return self.num_external_nodes

    @property
    def external_node_indices(self) -> list[int]:
        """List of all the indices of the external nodes."""
        return [node.index for node in self.external_nodes]

    @property
    def internal_nodes(self) -> list[Node]:
        """List of all internal nodes."""
        return [node for node in self.nodes if node.node_type == "internal"]

    @property
    def num_internal_nodes(self) -> int:
        """Number of internal nodes in the network."""
        return len(self.internal_nodes)

    @property
    def links(self) -> list[Link]:
        """List of links in the network."""
        return list(self.link_dict.values())

    @property
    def num_links(self) -> int:
        """Number of links in the network."""
        return len(list(self.links))

    @property
    def external_links(self) -> list[Link]:
        """List of all external links in the network."""
        return [link for link in self.links if link.link_type == "external"]

    @property
    def num_external_links(self) -> int:
        """Number of external links in the network."""
        return list(self.external_links)

    @property
    def internal_links(self) -> list[Link]:
        """List of all internal links in the network."""
        return [link for link in self.links if link.link_type == "internal"]

    @property
    def num_internal_links(self) -> int:
        """Number of internal links in the network."""
        return len(list(self.internal_links))

    # -------------------------------------------------------------------------
    # Basic utility functions
    # -------------------------------------------------------------------------

    def get_node(self, index: int | str) -> Node:
        """Returns the node with the specified index"""
        node = self.node_dict.get(str(index), None)
        if node is None:
            raise ValueError(f"Node {index} does not exist.")
        return node

    def get_link(self, index: int | str) -> Link:
        """Returns the link with the specified index"""
        link = self.link_dict.get(str(index), None)
        if link is None:
            raise ValueError(f"Link {index} does not exist.")
        return link

    def get_node_by_position(self, position: np.ndarray) -> Node:
        """Find the node with a given position."""
        for node in self.nodes:
            if np.allclose(node.position, position):
                return node
        raise ValueError("Node not found.")

    def get_link_by_node_indices(self, node_indices: tuple[int, int]) -> Link:
        """Find the link that connects the nodes with given indices."""
        for link in self.links:
            if sorted(link.node_indices) == sorted(node_indices):
                return link
        raise ValueError("Link not found.")

    def reset_dict_indices(self) -> None:
        """Go through the node and link dictionaries and generate a new
        numbering system.

        This is used when, for example, an extra node is added to the
        network."""

        # Set up conversion dictionaries from old to new
        new_node_dict = {}
        new_link_dict = {}

        old_to_new_links = {}
        old_to_new_nodes = {-1: -1}
        i = 0
        for node in self.internal_nodes + self.external_nodes:
            old_to_new_nodes[node.index] = i
            i += 1
        i = 0
        for link in self.internal_links + self.external_links:
            old_to_new_links[link.index] = i
            i += 1

        # Update all the nodes
        for node in self.nodes:
            node.index = old_to_new_nodes[node.index]
            node.sorted_connected_nodes = [
                old_to_new_nodes[j] for j in node.sorted_connected_nodes
            ]
            # Reorder the connected nodes and permute S matrices if required
            sorted_indices = np.argsort(node.sorted_connected_nodes)
            node.sorted_connected_nodes = list(
                np.array(node.sorted_connected_nodes)[sorted_indices]
            )

            # Permute matrix indices
            node.get_S = node_matrix.get_permuted_matrix_closure(
                node, "get_S", sorted_indices
            )
            node.get_S_inv = node_matrix.get_permuted_matrix_closure(
                node, "get_S_inv", sorted_indices
            )
            node.get_dS = node_matrix.get_permuted_matrix_closure(
                node, "get_dS", sorted_indices
            )

            node.sorted_connected_links = sorted(
                [old_to_new_links[j] for j in node.sorted_connected_links]
            )
            node.inwave = {
                str(old_to_new_nodes[int(key)]): value
                for key, value in node.inwave.items()
            }
            node.outwave = {
                str(old_to_new_nodes[int(key)]): value
                for key, value in node.outwave.items()
            }
            new_node_dict[str(node.index)] = node

        # Update all the links
        for link in self.links:
            link.index = old_to_new_links[link.index]
            link.node_indices = (
                old_to_new_nodes[link.node_indices[0]],
                old_to_new_nodes[link.node_indices[1]],
            )
            link.sorted_connected_nodes = [
                old_to_new_nodes[j] for j in link.sorted_connected_nodes
            ]

            link.inwave = {
                str(old_to_new_nodes[int(key)]): value
                for key, value in link.inwave.items()
            }
            link.outwave = {
                str(old_to_new_nodes[int(key)]): value
                for key, value in link.outwave.items()
            }
            new_link_dict[str(link.index)] = link

        self.node_dict = new_node_dict
        self.link_dict = new_link_dict

    # -------------------------------------------------------------------------
    #  Altering the geometry of the network
    # -------------------------------------------------------------------------

    def add_node_to_link(
        self,
        link_index: int,
        fractional_position: float,
        new_get_S: Callable | None = None,
        new_get_S_inv: Callable | None = None,
        new_get_dS: Callable | None = None,
    ) -> None:
        """Add a node to a link with given index at fractional position given
        by s, which ranges from 0 to 1.

        s=0 corresponds to the attached node with lower node index and s=1 is
        the opposite. s=0 and s=1 are not actaully valid options, since
        0 < s < 1.
        """
        is_invalid_fractional_position = (
            not isinstance(fractional_position, float)
            or np.isclose(fractional_position, 0.0)
            or np.isclose(fractional_position, 1.0)
            or fractional_position < 0.0
            or fractional_position > 1.0
        )
        if is_invalid_fractional_position:
            raise ValueError(
                f"fractional_position must be a float in the open interval "
                f"(0,1). "
                f"fractional_position={fractional_position} was given."
            )

        # Give default values if none are passed
        if new_get_S is None:
            new_get_S = Node.get_default_values().get("get_S")
        if new_get_S_inv is None:
            new_get_S_inv = Node.get_default_values().get("get_S_inv")
        if new_get_dS is None:
            new_get_dS = Node.get_default_values().get("get_dS")

        old_link = self.get_link(link_index)
        node_one = self.get_node(old_link.node_indices[0])
        node_two = self.get_node(old_link.node_indices[1])

        # Create new node
        node_position = node_one.position + fractional_position * (
            node_two.position - node_one.position
        )
        new_node = Node(
            self.num_nodes,
            "internal",
            node_position,
            data={
                "get_S": new_get_S,
                "get_S_inv": new_get_S_inv,
                "get_dS": new_get_dS,
                "S_mat_type": "custom",
            },
        )

        # Create two new links
        new_link_one = Link(
            self.num_links, "internal", (node_one.index, new_node.index)
        )
        new_link_two = Link(
            self.num_links + 1, "internal", (new_node.index, node_two.index)
        )

        # Fix the lists of connected nodes and links for the two outside nodes
        # The list of sorted nodes might not be in order anymore, so they need
        # sorting. If they are no longer in order, the scattering matrices need
        # to be permuted too
        for node, other_node, new_link in zip(
            [node_one, node_two],
            [node_two, node_one],
            [new_link_one, new_link_two],
        ):
            replace_index = node.sorted_connected_nodes.index(other_node.index)
            node.sorted_connected_nodes[replace_index] = new_node.index

            sorted_indices = np.argsort(node.sorted_connected_nodes)
            node.sorted_connected_nodes = list(
                np.array(node.sorted_connected_nodes)[sorted_indices]
            )

            # Permute matrix indices if necessary
            if not np.array_equal(sorted_indices, np.sort(sorted_indices)):
                node.get_S = node_matrix.get_permuted_matrix_closure(
                    node, "get_S", sorted_indices
                )
                node.get_S_inv = node_matrix.get_permuted_matrix_closure(
                    node, "get_S_inv", sorted_indices
                )
                node.get_dS = node_matrix.get_permuted_matrix_closure(
                    node, "get_dS", sorted_indices
                )

            node.sorted_connected_links.remove(old_link.index)
            node.sorted_connected_links.append(new_link.index)
            node.sorted_connected_links = sorted(node.sorted_connected_links)
            del node.inwave[str(other_node.index)]
            node.inwave[str(new_node.index)] = 0 + 0j
            del node.outwave[str(other_node.index)]
            node.outwave[str(new_node.index)] = 0 + 0j

        # Set up indices and other properties of the new node
        new_node.sorted_connected_nodes = sorted(
            [node_one.index, node_two.index]
        )
        new_node.sorted_connected_links = sorted(
            [new_link_one.index, new_link_two.index]
        )
        new_node.inwave = {
            str(node_one.index): 0 + 0j,
            str(node_two.index): 0 + 0j,
        }
        new_node.outwave = {
            str(node_one.index): 0 + 0j,
            str(node_two.index): 0 + 0j,
        }
        new_node.inwave_np = np.array([0 + 0j, 0 + 0j])
        new_node.outwave_np = np.array([0 + 0j, 0 + 0j])
        new_node.num_connect = 2

        # Set up new links
        for link, node in zip(
            [new_link_one, new_link_two], [node_one, node_two]
        ):
            link.length = np.linalg.norm(node.position - new_node.position)
            link.sorted_connected_nodes = sorted(link.node_indices)
            link.inwave = {
                str(node.index): 0 + 0j,
                str(new_node.index): 0 + 0j,
            }
            link.outwave = {
                str(node.index): 0 + 0j,
                str(new_node.index): 0 + 0j,
            }
            link.material = old_link.material
            link.n = old_link.n
            link.dn = old_link.dn
            link.get_S = link_matrix.get_propagation_matrix_closure(link)
            link.get_S_inv = (
                link_matrix.get_propagation_matrix_inverse_closure(link)
            )
            link.get_dS = (
                link_matrix.get_propagation_matrix_derivative_closure(link)
            )

        # Add new node and link to network dict
        self.node_dict[str(new_node.index)] = new_node
        self.link_dict[str(new_link_one.index)] = new_link_one
        self.link_dict[str(new_link_two.index)] = new_link_two

        # Delete the old link from the dict
        del self.link_dict[str(link_index)]

        # Update matrix maps and so on
        self.reset_dict_indices()
        self._reset_fields()
        self._set_matrix_calc_utils()

    def add_segment_to_link(
        self,
        link_index: int,
        fractional_positions: tuple[float, float],
        node_S_matrix_type: str = "fresnel",
    ) -> None:
        """Add a segment to a link.

        Initially the segment will have the same refractive index as the
        original link. The nodes will be given fresnel scattering matrices.

        The segment looks like this

        O       |--------|    O     |-----------|     O     |--------|    O
        node_one link_one first_node middle_link second_node link_two node_two

        Apologies for future developers for naming convention here. Go through
        it carefully with an example network."""

        # Sort for consistency and extract ratios
        fractional_positions = np.sort(fractional_positions)
        s1, s2 = fractional_positions

        original_link = self.get_link(link_index)
        node_one_index, node_two_index = original_link.node_indices
        node_one = self.get_node(node_one_index)
        node_two = self.get_node(node_two_index)
        node_one_position = node_one.position
        node_two_position = node_two.position

        # Work out the new node positions
        # This is so we can find them after all the index relabeling
        first_node_position = node_one.position + s1 * (
            node_two.position - node_one.position
        )
        second_node_position = node_one.position + s2 * (
            node_two.position - node_one.position
        )

        # Add first node and find its index
        self.add_node_to_link(link_index, s1)
        first_node_index = self.get_node_by_position(first_node_position).index

        # Find the link that connects the new node and node_two
        # Note: node_two_index may have changed!
        node_two_index = self.get_node_by_position(node_two_position).index
        second_link_index = self.get_link_by_node_indices(
            (first_node_index, node_two_index)
        ).index

        # Add second node and find its index
        # Ratio is its fractional position along the new link
        ratio = (s2 - s1) / (1 - s1)
        self.add_node_to_link(second_link_index, ratio)

        first_node_index = self.get_node_by_position(first_node_position).index
        second_node_index = self.get_node_by_position(
            second_node_position
        ).index
        middle_link = self.get_link_by_node_indices(
            (first_node_index, second_node_index)
        )

        self.update_segment_matrices(middle_link)

    def update_link_matrices(self, link: Link) -> None:
        """Update link scattering matrices (normally used after altering Dn)"""
        link.get_S = link_matrix.get_propagation_matrix_closure(link)
        link.get_S_inv = link_matrix.get_propagation_matrix_inverse_closure(
            link
        )
        link.get_dS = link_matrix.get_propagation_matrix_derivative_closure(
            link
        )

    def update_segment_matrices(self, link: Link) -> None:
        """Update the link and node scattering matrices at a segment.

        This is needed because the node scattering matrices will likely be
        Fresnel matrices, which need to be updated when the link's refractive
        index has changed."""

        # Update link scattering matrices
        self.update_link_matrices(link)

        node_one_index, node_two_index = link.node_indices
        node_one = self.get_node(node_one_index)
        node_two = self.get_node(node_two_index)
        link_one_index = (
            node_one.sorted_connected_links[0]
            if node_one.sorted_connected_links[0] != link.index
            else node_one.sorted_connected_links[1]
        )
        link_two_index = (
            node_two.sorted_connected_links[0]
            if node_two.sorted_connected_links[0] != link.index
            else node_two.sorted_connected_links[1]
        )
        link_one = self.get_link(link_one_index)
        link_two = self.get_link(link_two_index)

        # Update node scattering matrices
        # Note that we need to be careful about the order of the refractive
        # indices on either side of the nodes, i.e. which one is "n1" and which
        # is "n2" within the Fresnel formulas. The order is ultimately
        # determined by the numerical order of sorted_connected_nodes.
        sorted_connected_nodes = node_one.sorted_connected_nodes
        first_link = (
            link_one if sorted_connected_nodes[0] != node_two.index else link
        )
        second_link = link if first_link.index == link_one.index else link_one
        perturbed_link_number = 1 if first_link.index == link.index else 2

        node_one.get_S = node_matrix.get_S_fresnel_closure(
            first_link, second_link
        )
        node_one.get_S_inv = node_matrix.get_S_fresnel_inverse_closure(
            first_link, second_link
        )
        node_one.get_dS = node_matrix.get_S_fresnel_derivative_closure(
            first_link, second_link, perturbed_link_number
        )

        # Same, but the other node. Can maybe do both in a loop but the
        # way in which first_link and second_link are determined is different
        sorted_connected_nodes = node_two.sorted_connected_nodes
        first_link = (
            link_two if sorted_connected_nodes[0] != node_one.index else link
        )
        second_link = link if first_link.index == link_two.index else link_two
        perturbed_link_number = 1 if first_link.index == link.index else 2

        node_two.get_S = node_matrix.get_S_fresnel_closure(
            first_link, second_link
        )
        node_two.get_S_inv = node_matrix.get_S_fresnel_inverse_closure(
            first_link, second_link
        )
        node_two.get_dS = node_matrix.get_S_fresnel_derivative_closure(
            first_link, second_link, perturbed_link_number
        )

    def update_node_scattering_matrix(self, node_index: int, node_S_mat_type: str, node_S_mat_params: dict) -> None:
        """Update the S matrix of a node"""
        node = self.get_node(node_index)

        if node.node_type == "external":
            raise ValueError("External nodes do not have modifiable scattering matrices.")
        elif node.node_type == "internal":
            node.S_mat_params = node_S_mat_params
            node.get_S = node_matrix.get_constant_node_S_closure(
                node_S_mat_type, node.degree, node_S_mat_params
            )
            node.get_S_inv = node_matrix.get_inverse_matrix_closure(node.get_S)
            node.get_dS = node_matrix.get_zero_matrix_closure(node.degree)


    def translate_node(
        self, node_index: int, translation_vector: np.ndarray
    ) -> None:
        """Translate node given by node_index in the plane by
        translation_vector"""

        # Update the node itself
        node = self.get_node(node_index)
        node.position = node.position + translation_vector

        # Update all the connecting links
        for link_index in node.sorted_connected_links:
            link = self.get_link(link_index)

            # Update length
            node_one_index, node_two_index = link.node_indices
            node_one = self.get_node(node_one_index)
            node_two = self.get_node(node_two_index)
            length = np.linalg.norm(node_one.position - node_two.position)
            link.length = length

            # Update links matrices
            link.get_S = link_matrix.get_propagation_matrix_closure(link)
            link.get_S_inv = (
                link_matrix.get_propagation_matrix_inverse_closure(link)
            )
            link.get_dS = (
                link_matrix.get_propagation_matrix_derivative_closure(link)
            )

    # -------------------------------------------------------------------------
    #  Direct scattering methods
    #  These are mainly for setting the internal fields throughout the network
    # -------------------------------------------------------------------------

    def scatter_direct(
        self,
        incident_field: np.ndarray,
        k0: complex,
        method: str = "formula",
    ) -> None:
        """Scatter the incident field through the network using the
        network matrix.

        This method sets all the internal field values and doesn't return
        anything."""
        self._reset_fields()

        # Set up the matrix product
        network_matrix = self.get_network_matrix(k0, method)
        num_externals = self.num_external_nodes
        incident_vector = np.zeros((len(network_matrix)), dtype=np.complex128)
        incident_vector[num_externals : 2 * num_externals] = incident_field
        outgoing_vector = network_matrix @ incident_vector

        # Reset fields throughout the network and set incident field
        self._set_network_fields(outgoing_vector)

    def _set_network_fields(self, vector: np.ndarray) -> None:
        """Set the values of the fields throughout the network using a network
        vector"""
        external_vector_length = self.external_vector_length
        internal_vector_length = self.internal_vector_length

        outgoing_external = vector[0:external_vector_length]
        incoming_external = vector[
            external_vector_length : 2 * external_vector_length
        ]
        outgoing_internal = vector[
            2 * external_vector_length : 2 * external_vector_length
            + internal_vector_length
        ]
        incoming_internal = vector[
            2 * external_vector_length + internal_vector_length :
        ]

        self._set_incident_field(incoming_external)

        count = 0
        for node in self.external_nodes:
            # Set outgoing external values
            value = outgoing_external[count]
            node_index = node.index
            connected_link_index = node.sorted_connected_links[0]
            connected_link = self.get_link(connected_link_index)

            node.outwave["-1"] = value
            node.outwave_np[0] = value
            node.inwave[str(node.sorted_connected_nodes[1])] = value
            node.inwave_np[1] = value

            connected_link.outwave[str(node_index)] = value
            connected_link.outwave_np[1] = value

            # Set incoming external values
            value = incoming_external[count]
            node_index = node.index
            connected_link_index = node.sorted_connected_links[0]
            connected_link = self.get_link(connected_link_index)

            node.inwave["-1"] = value
            node.inwave_np[0] = value
            node.outwave[str(node.sorted_connected_nodes[1])] = value
            node.outwave_np[1] = value

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

        # Remaining external links values
        for link in self.external_links:
            external_index, node_index = link.node_indices
            node = self.get_node(node_index)

            # Set link fields
            link.inwave[str(node_index)] = node.outwave[str(external_index)]
            link.inwave_np[1] = node.outwave[str(external_index)]

            link.outwave[str(node_index)] = node.inwave[str(external_index)]
            link.outwave_np[1] = node.inwave[str(external_index)]

        self._update_outgoing_fields()

    def _set_incident_field(
        self, incident_field: np.ndarray, direction: str = "forward"
    ) -> None:
        """Sets the incident field to the inwaves/outwaves"""
        # Check size
        if len(incident_field) != self.num_external_nodes:
            raise ValueError(
                f"Incident field has incorrect size. "
                f"It should be of size {self.num_external_nodes}."
            )

        # Set values to nodes and network dictionaries
        for i, external_node in enumerate(self.external_nodes):
            if direction == "forward":
                self.inwave[str(external_node.index)] = incident_field[i]
                external_node.inwave["-1"] = incident_field[i]
                external_node.inwave_np[0] = incident_field[i]
            elif direction == "backward":
                self.outwave[str(external_node.index)] = incident_field[i]
                external_node.outwave["-1"] = incident_field[i]
                external_node.outwave_np[0] = incident_field[i]

        # Set values to network
        if direction == "forward":
            self.inwave_np = incident_field
        if direction == "backward":
            self.outwave_np = incident_field

    def _update_outgoing_fields(self, direction: str = "forward") -> None:
        """Update the fields from the external nodes and put them into the network
        inwave/outwaves"""
        for i, node in enumerate(self.external_nodes):
            if direction == "forward":
                self.outwave[str(node.index)] = node.outwave["-1"]
                self.outwave_np[i] = node.outwave["-1"]
            if direction == "backward":
                self.inwave[str(node.index)] = node.inwave["-1"]
                self.inwave_np[i] = node.inwave["-1"]

    def get_network_matrix(
        self, k0: float | complex, method: str = "formula"
    ) -> np.ndarray:
        """Get the 'infinite' order network matrix

        The method parameter can be either
            "eigenvalue" - calculate the matrix from an eigendecomposition
            "formula" - calculate the matrix from the direct formula"""
        match method:
            case "eigenvalue":
                step_matrix = self._get_network_step_matrix(k0)
                lam, v = np.linalg.eig(step_matrix)
                modified_lam = np.where(
                    np.isclose(lam, 1.0 + 0.0 * 1j), lam, 0.0
                )
                network_matrix = v @ np.diag(modified_lam) @ np.linalg.inv(v)
            case "formula":
                S_ee = self.get_S_ee(k0)
                S_ie = self.get_S_ie(k0)
                identity = np.identity(
                    self.external_vector_length, dtype=np.complex128
                )

                # Zero matrices used to build up the block matrix
                z_ee = np.zeros(
                    (self.external_vector_length, self.external_vector_length),
                    dtype=np.complex128,
                )
                z_ei = np.zeros(
                    (self.external_vector_length, self.internal_vector_length),
                    dtype=np.complex128,
                )
                z_ie = np.zeros(
                    (
                        2 * self.internal_vector_length,
                        self.external_vector_length,
                    ),
                    dtype=np.complex128,
                )
                z_ii = np.zeros(
                    (
                        2 * self.internal_vector_length,
                        self.internal_vector_length,
                    ),
                    dtype=np.complex128,
                )
                network_matrix = np.block(
                    [
                        [z_ee, S_ee, z_ei, z_ei],
                        [z_ee, identity, z_ei, z_ei],
                        [z_ie, S_ie, z_ii, z_ii],
                    ]
                )

        return network_matrix

    def _get_network_step_matrix(self, k0: float | complex) -> np.ndarray:
        """The network step matrix satisfies

        (O_e)       (0 0         |P_ei       0)(O_e)
        (I_e)       (0 1         |0          0)(I_e)
        (---)   =   (-------------------------)(---)
        (O_i)       (0 S_ii*P_ie | S_ii*P_ii 0)(O_i)
        (I_i)_n+1   (0 P_ie      | P_ii      0)(I_i)_n
        """
        # Get the internal S
        internal_S = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for node in self.internal_nodes:
            node_index = node.index
            node_S = node.get_S(k0)
            new_slice = self.internal_scattering_slices[str(node_index)]
            internal_S[new_slice, new_slice] = node_S

        # Get internal P
        internal_P = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S = link.get_S(k0)
            phase_factor = link_S[0, 1]

            # Wave that is going into node_one
            row = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = self.internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor
            # Wave propagating the other way
            internal_P[col, row] = phase_factor

        # Get external P
        external_P = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.external_links:
            node_one_index, node_two_index = link.node_indices
            link_S = link.get_S(k0)
            phase_factor = link_S[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            external_P[row, col] = phase_factor

        # Build up network matrix
        # First define all the zero matrices to keep things simpler
        z_ee = np.zeros(
            (self.external_vector_length, self.external_vector_length),
            dtype=np.complex128,
        )
        z_long = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        z_tall = z_long.T
        z_ii = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        identity = np.identity(
            self.external_vector_length, dtype=np.complex128
        )

        network_step_matrix = np.block(
            [
                [z_ee, z_ee, external_P, z_long],
                [z_ee, identity, z_long, z_long],
                [
                    z_tall,
                    internal_S @ external_P.T,
                    internal_S @ internal_P,
                    z_ii,
                ],
                [z_tall, external_P.T, internal_P, z_ii],
            ]
        )
        return network_step_matrix

    # -------------------------------------------------------------------------
    # Calculate various scattering matrices
    # -------------------------------------------------------------------------

    def get_S_ee(self, k0: float | complex) -> np.ndarray:
        """Get the external scattering matrix from the inverse formula

        O_e = S_ee @ I_e"""
        P_ei = self.get_P_ei(k0)
        P_ie = self.get_P_ie(k0)
        S_ii = self.get_S_ii(k0)
        P_ii = self.get_P_ii(k0)

        # Bracketed part to be inverted
        bracket = np.identity(len(S_ii), dtype=np.complex128) - S_ii @ P_ii
        inv = np.linalg.inv(bracket)

        S_ee = P_ei @ inv @ S_ii @ P_ie
        return S_ee

    def get_S_ee_inv(self, k0: float | complex) -> np.ndarray:
        """Get the external inverse scattering matrix from the inverse
        formula

        I_e = S^-1_ee @ O_e"""
        P_ei_inv = self.get_P_ei_inv(k0)
        P_ie_inv = P_ei_inv.T
        S_ii_inv = self.get_S_ii_inv(k0)
        P_ii_inv = self.get_P_ii_inv(k0)

        # Bracketed part to be inverted
        bracket = (
            np.identity(len(S_ii_inv), dtype=np.complex128)
            - P_ii_inv @ S_ii_inv
        )
        inv = np.linalg.inv(bracket)

        S_ee_inv = P_ei_inv @ S_ii_inv @ inv @ P_ie_inv
        return S_ee_inv

    def get_S_ie(self, k0: float | complex) -> np.ndarray:
        """Get the external scattering matrix from the inverse formula

        (O_i, I_i) = S_ie @ I_e"""
        P_ie = self.get_P_ie(k0)
        S_ii = self.get_S_ii(k0)
        P_ii = self.get_P_ii(k0)

        # Bracketed part to be inverted
        bracket_top = np.identity(len(S_ii), dtype=np.complex128) - S_ii @ P_ii
        bracket_bottom = (
            np.identity(len(S_ii), dtype=np.complex128) - P_ii @ S_ii
        )
        inv_top = np.linalg.inv(bracket_top)
        inv_bottom = np.linalg.inv(bracket_bottom)

        top = inv_top @ S_ii @ P_ie
        bottom = inv_bottom @ P_ie
        S_ie = np.block([[top], [bottom]])
        return S_ie

    def get_S_ie_inv(self, k0: float | complex) -> np.ndarray:
        """Get the external scattering matrix from the inverse formula

        (O_i, I_i) = S_ie @ O_e"""
        P_ie_inv = self.get_P_ie_inv(k0)
        S_ii_inv = self.get_S_ii_inv(k0)
        P_ii_inv = self.get_P_ii_inv(k0)

        # Bracketed part to be inverted
        bracket_top = (
            np.identity(len(S_ii_inv), dtype=np.complex128)
            - P_ii_inv @ S_ii_inv
        )
        bracket_bottom = (
            np.identity(len(S_ii_inv), dtype=np.complex128)
            - S_ii_inv @ P_ii_inv
        )
        inv_top = np.linalg.inv(bracket_top)
        inv_bottom = np.linalg.inv(bracket_bottom)

        top = inv_top @ P_ie_inv
        bottom = inv_bottom @ S_ii_inv @ P_ie_inv
        S_ie_inv = np.block([[top], [bottom]])
        return S_ie_inv

    def get_S_ii(self, k0: float | complex) -> np.ndarray:
        """Return the S_ii matrix formed from node scattering matrices"""
        internal_S = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for node in self.internal_nodes:
            node_index = node.index
            node_S = node.get_S(k0)
            new_slice = self.internal_scattering_slices[str(node_index)]
            internal_S[new_slice, new_slice] = node_S
        return internal_S

    def get_S_ii_inv(self, k0: float | complex) -> np.ndarray:
        """Return the inverse of the S_ii matrix formed from node scattering
        matrices"""
        return np.linalg.inv(self.get_S_ii(k0))

    def get_P_ii(self, k0: float | complex) -> np.ndarray:
        """Return P matrix calculated from internal network links"""
        # Get internal P
        internal_P = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S = link.get_S(k0)
            phase_factor = link_S[0, 1]

            # Wave that is going into node_one
            row = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = self.internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor
            internal_P[col, row] = phase_factor
        return internal_P

    def get_P_ii_inv(self, k0: float | complex) -> np.ndarray:
        """Return P matrix calculated from internal network links"""
        # Get internal P
        internal_P = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_inv = link.get_S_inv(k0)
            phase_factor = link_S_inv[0, 1]

            # Wave that is going into node_one
            row = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = self.internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor
            # Wave propagating the other way
            internal_P[col, row] = phase_factor
        return internal_P

    def get_P_ei(self, k0: float | complex) -> np.ndarray:
        """Get the matrix that deals with propagation in external links"""
        external_P = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.external_links:
            node_one_index, node_two_index = link.node_indices
            link_S = link.get_S(k0)
            phase_factor = link_S[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            external_P[row, col] = phase_factor
        return external_P

    def get_P_ei_inv(self, k0: float | complex) -> np.ndarray:
        """Get the matrix that deals with propagation in external links"""
        external_P = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.external_links:
            node_one_index, node_two_index = link.node_indices
            link_S_inv = link.get_S_inv(k0)
            phase_factor = link_S_inv[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            external_P[row, col] = phase_factor
        return external_P

    def get_P_ie(self, k0: float | complex) -> np.ndarray:
        return self.get_P_ei(k0).T

    def get_P_ie_inv(self, k0: float | complex) -> np.ndarray:
        return np.linalg.inv(self.get_P_ie(k0))

    # -------------------------------------------------------------------------
    # Other miscellaneous scattering matrix related functions
    # These ones are mostly used in pole searching
    # -------------------------------------------------------------------------

    def get_inv_factor(self, k0: float | complex) -> complex:
        """Calculate I - S_ii P_ii

        This will have zero determinant at a pole"""
        S_ii = self.get_S_ii(k0)
        P_ii = self.get_P_ii(k0)
        inv_factor = np.identity(len(S_ii), dtype=np.complex128) - S_ii @ P_ii
        return inv_factor

    def get_inv_factor_det(self, k0: float | complex) -> complex:
        """Calculate det(I - S_ii P_ii)

        This will be zero at a pole"""
        return np.linalg.det(self.get_inv_factor(k0))

    def get_S_ee_inv_det(self, k0: float | complex) -> complex:
        """Calculate det(I - S_ii P_ii)

        This will be zero at a pole"""
        return np.linalg.det(self.get_S_ee_inv(k0))

    # -------------------------------------------------------------------------
    # Methods for getting derivatives and Wigner-Smith operators
    # -------------------------------------------------------------------------

    def get_wigner_smith(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Calculate directly the wavenumber Wigner-Smith operator
        Q = -i * S^-1 * dS/dv where v = variable.

        variable has a few options:
            "k0": wavenumber (standard Wigner-Smith operator)
            "Dn": refractive index perturbation

        kwargs should contain any additional arguments required for calculating
        the derivative. This is particularly relevant for perturbation
        derivatives."""
        S_ee_inv = self.get_S_ee_inv(k0)
        dS_ee = self.get_dS_ee(k0, variable, **kwargs)
        ws = -1j * S_ee_inv @ dS_ee
        return ws

    def get_wigner_smith_volume(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Calculate the Wigner-Smith operator using volume integrals
        Q = -i * S^-1 * dS/dv where v = variable.

        variable has a few options:
            "k0": wavenumber (standard Wigner-Smith operator)
            "Dn": refractive index perturbation

        kwargs should contain any additional arguments required for calculating
        the derivative. This is particularly relevant for perturbation
        derivatives."""
        U_0_links = self.get_U_0_links(k0)
        U_1_links = self.get_U_1_links(k0, variable, **kwargs)
        U_2_links = self.get_U_2_links(k0, variable, **kwargs)
        U_3_links = self.get_U_3_links(k0, variable, **kwargs)

        S_prod_nodes = self.get_S_prod_nodes(k0)
        ws_nodes = self.get_ws_nodes(k0, variable, **kwargs)

        pre_factor = np.linalg.inv(
            np.identity(len(U_0_links), dtype=np.complex128)
            + U_0_links
            + S_prod_nodes
        )

        post_factor = U_1_links + U_2_links + U_3_links + ws_nodes
        ws_volume = pre_factor @ post_factor
        return ws_volume

    def get_dS_ee(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Get the derivative of S_ee with respect to variable"""

        P_ei = self.get_P_ei(k0)
        P_ie = P_ei.T
        P_ii = self.get_P_ii(k0)
        S_ii = self.get_S_ii(k0)

        dP_ei = self.get_dP_ei(k0, variable)
        dP_ie = dP_ei.T
        dP_ii = self.get_dP_ii(k0, variable, **kwargs)
        dS_ii = self.get_dS_ii(k0, variable)

        inv = np.linalg.inv(np.identity(len(S_ii)) - S_ii @ P_ii)
        dinv = inv @ (dS_ii @ P_ii + S_ii @ dP_ii) @ inv

        term_one = dP_ei @ inv @ S_ii @ P_ie
        term_two = P_ei @ dinv @ S_ii @ P_ie
        term_three = P_ei @ inv @ dS_ii @ P_ie
        term_four = P_ei @ inv @ S_ii @ dP_ie

        dS_ee = term_one + term_two + term_three + term_four
        return dS_ee

    def get_dS_ie(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Get the derivative of S_ie with respect to variable"""

        P_ie = self.get_P_ie(k0)
        P_ii = self.get_P_ii(k0)
        S_ii = self.get_S_ii(k0)

        dP_ie = self.get_dP_ie(k0, variable)
        dP_ii = self.get_dP_ii(k0, variable, **kwargs)
        dS_ii = self.get_dS_ii(k0, variable)

        # Top matrix (O part)
        inv = np.linalg.inv(
            np.identity(len(S_ii), np.complex128) - S_ii @ P_ii
        )
        dinv = inv @ (dS_ii @ P_ii + S_ii @ dP_ii) @ inv

        term_one = dinv @ S_ii @ P_ie
        term_two = inv @ dS_ii @ P_ie
        term_three = inv @ S_ii @ dP_ie
        top = term_one + term_two + term_three

        # Bottom matrix (I part)
        inv = np.linalg.inv(
            np.identity(len(P_ii), dtype=np.complex128) - P_ii @ S_ii
        )
        dinv = inv @ (dP_ii @ S_ii + P_ii @ dS_ii) @ inv

        term_one = dinv @ P_ie
        term_two = inv @ dP_ie
        bottom = term_one + term_two

        dS_ie = np.block([[top], [bottom]])
        return dS_ie

    def get_dS_ii(
        self,
        k0: float | complex,
        variable: str = "k0",
    ) -> np.ndarray:
        """Return the derivative of the S_ii matrix formed from node
        scattering matrices"""
        dS_ii = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for node in self.internal_nodes:
            node_index = node.index
            node_dS = node.get_dS(k0, variable)
            new_slice = self.internal_scattering_slices[str(node_index)]
            dS_ii[new_slice, new_slice] = node_dS
        return dS_ii

    def get_dP_ii(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Return derivative of P_ii with respect to the given variable"""
        # Unpacking of kwargs for certain types of derivatives
        match variable:
            case "Dn":
                perturbed_link_index = kwargs.get("perturbed_link_index", None)
                if perturbed_link_index is None:
                    raise ValueError(
                        "perturbed_link_index needs to be passed "
                        "to the derivative methods. It wasn't."
                    )
            case _:
                pass

        dP_ii = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )

        for link in self.internal_links:
            # Perturbation variables only require localized calculations on
            # certain links
            match variable:
                case "Dn":
                    if link.index != perturbed_link_index:
                        continue
                case _:
                    pass

            # Unpack link parameters
            node_one_index, node_two_index = link.node_indices
            link_dS = link.get_dS(k0, variable)
            phase_factor = link_dS[0, 1]

            # Wave that is going into node_one
            row = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = self.internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]

            dP_ii[row, col] = phase_factor
            dP_ii[col, row] = phase_factor

        return dP_ii

    def get_dP_ei(
        self,
        k0: float | complex,
        variable: str = "k0",
    ) -> np.ndarray:
        """Return derivative of P_ei with respect to variable

        Only non-zero if variable is k0"""
        dP_ei = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )

        # Unpacking of kwargs for certain types of derivatives
        match variable:
            case "Dn":
                return dP_ei
            case _:
                pass

        for link in self.external_links:
            node_one_index, node_two_index = link.node_indices
            link_dS = link.get_dS(k0, variable)
            phase_factor = link_dS[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            dP_ei[row, col] = phase_factor
        return dP_ei

    def get_dP_ie(
        self,
        k0: float | complex,
        variable: str = "k0",
    ) -> np.ndarray:
        """Return derivative of P_ie with respect to k0"""
        return self.get_dP_ei(k0, variable).T

    # -------------------------------------------------------------------------
    # Volume integral methods
    # Note to developers: U functions are all quite similar. Might be possible
    # to combine their calculations into one function to reduce redundancy
    # -------------------------------------------------------------------------

    def get_S_prod_nodes(self, k0: float | complex) -> np.ndarray:
        """Work out contributions to S^dag S from the node integrals"""

        num_externals = self.num_external_nodes
        U_0 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        # First work out all of the fields throughout the network
        S_ie = self.get_S_ie(k0)

        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Internal fields
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):

                # Get the field distribution associated with q
                # illumination
                q_o = O_int_vectors[q]
                q_i = I_int_vectors[q]
                p_o = O_int_vectors[p]
                p_i = I_int_vectors[p]
                partial_sum = 0.0 + 0.0j

                for node in self.internal_nodes:
                    sorted_connected_nodes = node.sorted_connected_nodes
                    node_index = node.index
                    keys = [
                        f"{node_index},{connected_index}"
                        for connected_index in sorted_connected_nodes
                    ]

                    # Get field component vector
                    I_p = np.zeros(len(keys), dtype=np.complex128)
                    I_q = np.zeros(len(keys), dtype=np.complex128)
                    for i, key in enumerate(keys):
                        I_p[i] = p_i[self.internal_scattering_map[key]]
                        I_q[i] = q_i[self.internal_scattering_map[key]]

                    S = node.get_S(k0)
                    matrix = np.conj(S.T) @ S - np.identity(
                        len(S), dtype=np.complex128
                    )
                    partial_sum += np.dot(np.conj(I_q), matrix @ I_p)

                U_0[q, p] = partial_sum
        return U_0

    def get_U_0_links(self, k0: float | complex) -> np.ndarray:
        """Calculate the U_0 matrix (see theory notes)"""

        num_externals = self.num_external_nodes
        U_0 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        # First work out all of the fields throughout the network
        S_ie = self.get_S_ie(k0)

        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Internal fields
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):
                # Get the field distribution associated with q
                # illumination
                q_o = O_int_vectors[q]
                q_i = I_int_vectors[q]
                p_o = O_int_vectors[p]
                p_i = I_int_vectors[p]
                partial_sum = 0.0 + 0.0j

                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = self.internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    partial_sum += O_mp * np.conj(O_mq) * (
                        np.exp(-2 * (n + Dn) * np.imag(k0) * length) - 1.0
                    ) + I_mp * np.conj(I_mq) * (
                        1.0 - np.exp(2 * (n + Dn) * np.imag(k0) * length)
                    )

                U_0[q, p] = partial_sum

        return U_0

    def get_U_0_unconjugated(self, k0: float | complex) -> np.ndarray:
        """Calculate the U_0 matrix (see theory notes)"""

        num_externals = self.num_external_nodes
        U_0 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        # First work out all of the fields throughout the network
        S_ie = self.get_S_ie(k0)

        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Internal fields
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):
                # Get the field distribution associated with q
                # illumination
                q_o = O_int_vectors[q]
                q_i = I_int_vectors[q]
                p_o = O_int_vectors[p]
                p_i = I_int_vectors[p]

                partial_sum = 0.0 + 0.0j

                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]

                    key = f"{node_one_index},{node_two_index}"
                    index = self.internal_scattering_map[key]
                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    partial_sum -= O_mp * O_mq * (
                        np.exp(2.0j * (n + Dn) * k0 * length) - 1.0
                    ) + I_mp * I_mq * (
                        1.0 - np.exp(-2.0j * (n + Dn) * k0 * length)
                    )

                    # Node contributions
                    # z = 0
                    near_term = (
                        O_mp * O_mq - O_mp * I_mq + I_mp * O_mq - I_mp * I_mq
                    )

                    # z = L
                    far_I_mp = O_mp * np.exp(1j * (n + Dn) * k0 * length)
                    far_O_mp = I_mp * np.exp(-1j * (n + Dn) * k0 * length)
                    far_I_mq = O_mq * np.exp(1j * (n + Dn) * k0 * length)
                    far_O_mq = I_mq * np.exp(-1j * (n + Dn) * k0 * length)

                    if link.nature == "external":
                        far_term = 0.0
                    else:
                        far_term = (
                            far_O_mp * far_O_mq
                            - far_O_mp * far_I_mq
                            + far_I_mp * far_O_mq
                            - far_I_mp * far_I_mq
                        )
                    print(-near_term - far_term)
                    partial_sum += -near_term - far_term

                # Node contributions
                # for node in self.internal_nodes:
                #     sorted_connected_nodes = node.sorted_connected_nodes
                #     node_index = node.index
                #     keys = [
                #         f"{node_index},{connected_index}"
                #         for connected_index in sorted_connected_nodes
                #     ]

                #     # Get field component vector
                #     I_p = np.zeros(len(keys), dtype=np.complex128)
                #     I_q = np.zeros(len(keys), dtype=np.complex128)
                #     for i, key in enumerate(keys):
                #         I_p[i] = p_i[self.internal_scattering_map[key]]
                #         I_q[i] = q_i[self.internal_scattering_map[key]]

                #     S = node.get_S(k0)
                #     matrix = np.identity(len(S), dtype=np.complex128) - S.T @ S
                #     partial_sum += np.dot(I_p, matrix @ I_q)

                U_0[q, p] = partial_sum

        return U_0

    def get_ws_nodes(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Work out contributions to S^dag S from the node integrals"""

        num_externals = self.num_external_nodes
        U_3 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)
        dS_ie = self.get_dS_ie(k0, variable, **kwargs)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []
        dO_int_vectors = []
        dI_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            doutgoing_vector = dS_ie @ incident_field

            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            new_dO = doutgoing_vector[: int(len(outgoing_vector) / 2)]
            new_dI = doutgoing_vector[int(len(outgoing_vector) / 2) :]

            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)
            dO_int_vectors.append(new_dO)
            dI_int_vectors.append(new_dI)

        internal_scattering_map = self.internal_scattering_map

        for q in range(num_externals):
            for p in range(num_externals):

                q_o = O_int_vectors[q]
                q_i = I_int_vectors[q]
                dq_o = dO_int_vectors[q]
                dq_i = dI_int_vectors[q]

                p_o = O_int_vectors[p]
                p_i = I_int_vectors[p]
                dp_o = dO_int_vectors[p]
                dp_i = dI_int_vectors[p]

                partial_sum = 0.0 + 0.0j

                for node in self.internal_nodes:
                    sorted_connected_nodes = node.sorted_connected_nodes
                    node_index = node.index
                    keys = [
                        f"{node_index},{connected_index}"
                        for connected_index in sorted_connected_nodes
                    ]

                    # Get field component vectors
                    I_p = np.zeros(len(keys), dtype=np.complex128)
                    I_q = np.zeros(len(keys), dtype=np.complex128)
                    dI_p = np.zeros(len(keys), dtype=np.complex128)
                    dI_q = np.zeros(len(keys), dtype=np.complex128)

                    for i, key in enumerate(keys):
                        I_p[i] = p_i[internal_scattering_map[key]]
                        dI_p[i] = dp_i[internal_scattering_map[key]]
                        I_q[i] = q_i[internal_scattering_map[key]]
                        dI_q[i] = dq_i[internal_scattering_map[key]]

                    S = node.get_S(k0)
                    dS = node.get_dS(k0, variable)

                    # First term
                    matrix = -1j * np.conj(S.T) @ dS
                    partial_sum += np.dot(np.conj(I_q), matrix @ I_p)

                    # Second term
                    matrix = 1j * (
                        np.identity(len(S), dtype=np.complex128)
                        - np.conj(S.T) @ S
                    )
                    partial_sum += np.dot(np.conj(I_q), matrix @ dI_p)

                U_3[q, p] = partial_sum
        return U_3

    def get_U_1_links(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Calculate the U_1 matrix relevant to the k0 Wigner-Smith operator
        (see theory notes)"""
        # Unpacking of kwargs for certain types of derivatives
        match variable:
            case "Dn":
                perturbed_link_index = kwargs.get("perturbed_link_index", None)
                if perturbed_link_index is None:
                    raise ValueError(
                        "perturbed_link_index needs to be passed "
                        "to the derivative methods. It wasn't."
                    )
            case _:
                pass

        num_externals = self.num_external_nodes
        U_1 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    dn = link.dn(k0)

                    Dn = link.Dn

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = self.internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    # Factor related to refractive index derivative
                    match variable:
                        case "k0":
                            factor = length * (n + Dn + k0 * (dn))
                        case "Dn":
                            if link.index != perturbed_link_index:
                                continue
                            factor = length * k0

                    partial_sum += factor * (
                        O_mp
                        * np.conj(O_mq)
                        * (np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                        + I_mp
                        * np.conj(I_mq)
                        * (np.exp(2 * (n + Dn) * np.imag(k0) * length))
                    )

                U_1[q, p] = partial_sum

        return U_1

    def get_U_2_links(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Calculate the U_2 matrix (see theory notes)"""
        # Unpacking of kwargs for certain types of derivatives
        match variable:
            case "Dn":
                perturbed_link_index = kwargs.get("perturbed_link_index", None)
                if perturbed_link_index is None:
                    raise ValueError(
                        "perturbed_link_index needs to be passed "
                        "to the derivative methods. It wasn't."
                    )
            case _:
                pass

        num_externals = self.num_external_nodes
        U_2 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j

                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    dn = link.dn(k0)
                    Dn = link.Dn

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = self.internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    # Factor related to refractive index derivative
                    match variable:
                        case "k0":
                            factor = (
                                0.5
                                * (dn)
                                / (n + Dn)
                                * np.imag(k0)
                                / np.real(k0)
                            )
                        case "Dn":
                            if link.index != perturbed_link_index:
                                continue
                            factor = 0.5 / (n + Dn) * np.imag(k0) / np.real(k0)

                    partial_sum += factor * (
                        O_mp
                        * np.conj(I_mq)
                        * (np.exp(2j * (n + Dn) * np.real(k0) * length) - 1.0)
                        + I_mp
                        * np.conj(O_mq)
                        * (1.0 - np.exp(-2j * (n + Dn) * np.real(k0) * length))
                    )

                U_2[q, p] = partial_sum

        return U_2

    def get_U_3_links(
        self,
        k0: float | complex,
        variable: str = "k0",
        **kwargs: dict[str, Any],
    ) -> np.ndarray:
        """Calculate the U_3 matrix associated with the wavenumber
        (see theory notes)"""
        # Unpacking of kwargs for certain types of derivatives
        match variable:
            case "Dn":
                perturbed_link_index = kwargs.get("perturbed_link_index", None)
                if perturbed_link_index is None:
                    raise ValueError(
                        "perturbed_link_index needs to be passed "
                        "to the derivative methods. It wasn't."
                    )
            case _:
                pass

        num_externals = self.num_external_nodes
        U_3 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)
        dS_ie = self.get_dS_ie(k0, variable, **kwargs)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []
        dO_int_vectors = []
        dI_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            doutgoing_vector = dS_ie @ incident_field

            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            new_dO = doutgoing_vector[: int(len(outgoing_vector) / 2)]
            new_dI = doutgoing_vector[int(len(outgoing_vector) / 2) :]

            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)
            dO_int_vectors.append(new_dO)
            dI_int_vectors.append(new_dI)

        internal_scattering_map = self.internal_scattering_map

        for q in range(num_externals):
            for p in range(num_externals):

                partial_sum = 0.0 + 0.0j
                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    dq_o = dO_int_vectors[q]
                    dq_i = dI_int_vectors[q]

                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]
                    dp_o = dO_int_vectors[p]
                    dp_i = dI_int_vectors[p]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    dI_mp = dp_i[index]
                    dI_mq = dq_i[index]
                    dO_mp = dp_o[index]
                    dO_mq = dq_o[index]

                    partial_sum += 1j * (
                        dO_mp
                        * np.conj(O_mq)
                        * (1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                        + dI_mp
                        * np.conj(I_mq)
                        * (np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0)
                    )

                U_3[q, p] = partial_sum

        return U_3

    # -------------------------------------------------------------------------
    # Plotting methods
    # -------------------------------------------------------------------------

    def draw(
        self,
        ax=None,
        show_indices: bool = False,
        show_external_indices: bool = False,
        equal_aspect: bool = False,
        highlight_nodes: list[int] | None = None,
        highlight_links: list[int] | None = None,
        highlight_perturbed: bool = True,
        hide_axes: bool = False,
        draw_boundary: float | None = None,
        title: str | None = None,
        save_dir: str | None = None,
    ) -> None:
        """Draw network"""

        if ax is None:
            ax = plt.gca()  # use current axis if none is given
        if equal_aspect:
            ax.set_aspect("equal")

        # Boundary
        if draw_boundary is not None:
            t = np.linspace(-draw_boundary, draw_boundary, 10**6)
            y = np.sqrt(draw_boundary**2 - t**2)
            linewidth = 1
            ax.plot(t, y, linestyle="--", color="black")
            ax.plot(t, -y, linestyle="--", color="black")

        # Title
        if title is not None:
            ax.set_title(title)

        # Plot links
        for link in self.links:
            node_1_index, node_2_index = link.node_indices
            node_1_pos = self.get_node(node_1_index).position
            node_2_pos = self.get_node(node_2_index).position
            color = (
                "red"
                if highlight_links is not None
                and link.index in highlight_links
                else None
            )
            link.draw(ax, node_1_pos, node_2_pos, show_indices, color=color)

        # Highlight links
        # if highlight_links is not None:
        #     for link_index in highlight_links:
        #         link = self.get_link(link_index)
        #         node_1_index, node_2_index = link.node_indices
        #         node_1_pos = self.get_node(node_1_index).position
        #         node_2_pos = self.get_node(node_2_index).position
        #         link.draw(ax, node_1_pos, node_2_pos, color="red")

        # Plot nodes
        for node in self.nodes:
            if node.is_perturbed:
                continue
            color = (
                "red"
                if highlight_nodes is not None
                and node.index in highlight_nodes
                else None
            )
            node.draw(ax, show_indices, show_external_indices, color=color)

        # Highlight nodes
        # if highlight_nodes is not None:
        #     for node_index in highlight_nodes:
        #         node = self.get_node(node_index)
        #         node.draw(ax, color="red")

        # Custom highlighting for perturbations
        if highlight_perturbed:
            for node in self.nodes:
                if not node.is_perturbed:
                    continue

                match node.perturbation_data.get("perturbation_type"):
                    case "pseudonode":
                        r = node.perturbation_data.get("r")
                        node.draw(
                            ax,
                            show_indices,
                            show_external_indices,
                            color="red",
                            markersize=1 + (6 - 1) * r,
                        )
                    case _:
                        pass

        if hide_axes:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
            )
            ax.set_xticklabels([])
            ax.set_yticklabels([])

        if save_dir is not None:
            plt.savefig(save_dir, format="svg", bbox_inches="tight")

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

        # external nodes
        for node in self.external_nodes:
            node.draw(ax, color="white")

        # Highlight nodes
        for node_index in highlight_nodes:
            node = self.get_node(node_index)
            node.draw(ax, color="red")

    def plot_internal(
        self,
        k0: complex,
        highlight_nodes: list[int] | None = None,
        title: str = "Field distribution",
        max_val: float | None = None,
        save_dir: str | None = None,
        lw: float = 1.0,
    ) -> None:
        """Show internal field distribution in the network"""
        if highlight_nodes is None:
            highlight_nodes = []

        fig, ax = plt.subplots()
        # ax.set_aspect("equal")
        ax.set_title(title)
        ax.set_aspect("equal")
        # Get link intensities to define the colormap
        # Plot links
        maxes = []
        for link in self.links:
            node_index, _ = link.node_indices
            inwave = link.inwave[str(node_index)]
            outwave = link.outwave[str(node_index)]
            z = np.linspace(0, link.length, 10**4)
            field = inwave * np.exp(
                1j * k0 * (link.n(k0) + link.Dn) * z
            ) + outwave * np.exp(-1j * k0 * (link.n(k0) + link.Dn) * z)
            intensity = np.abs(field) ** 2
            max_val = np.max(intensity)
            maxes.append(max_val)

        norm = mcolors.Normalize(vmin=np.min(maxes), vmax=np.max(maxes))
        cmap = plt.cm.coolwarm
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "MyCmapName", ["b", "r"]
        )

        # Plot links
        for i, link in enumerate(self.links):
            node_1_index, node_2_index = link.node_indices
            node_1_pos = self.get_node(node_1_index).position
            node_2_pos = self.get_node(node_2_index).position
            color = cmap(norm(maxes[i]))
            link.draw(ax, node_1_pos, node_2_pos, color=color, lw=lw)

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
        # ax.set_facecolor(bg_color)

        # external nodes
        for node in self.external_nodes:
            node.draw(ax, color="white")

        # Highlight nodes
        for node_index in highlight_nodes:
            node = self.get_node(node_index)
            node.draw(ax, color="red")

        if save_dir is not None:
            plt.savefig(save_dir, format="svg", bbox_inches="tight")
