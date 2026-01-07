"""Main network class module."""

from typing import Any, Callable

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
from complex_network.components.link import Link
from complex_network.components.node import Node
from complex_network.scattering_matrices import link_matrix, node_matrix
from typing import Tuple, List

from multiprocessing import Pool, cpu_count

from complex_network.networks.network_spec import NetworkSpec



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
        node_dict: dict[int, Node],
        link_dict: dict[int, Link],
        data: dict[str, Any] | None = None,
        spec: "NetworkSpec | None" = None,
    ) -> None:
        """Nodes and links are primarily stored in dictionaries whose keys
        are the indices. Don't make a network directly from here; use
        network_factory.py."""
        self._reset_values(data)
        self.node_dict = node_dict
        self.link_dict = link_dict
        self._network_spec = spec  # Store the NetworkSpec used to create this network
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
            self.inwave[node.index] = 0 + 0j
            self.outwave[node.index] = 0 + 0j

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
        # Precompute indices for internal link fields
        self.internal_link_indices_A_to_B = np.array([
            self.internal_scattering_map[link.node_indices[0],link.node_indices[1]]
            for link in self.internal_links
        ])
        self.internal_link_indices_B_to_A = np.array([
            self.internal_scattering_map[link.node_indices[1],link.node_indices[0]]
            for link in self.internal_links
        ])

    def _get_network_matrix_maps(
        self,
    ) -> tuple[dict[tuple[int,int], int], dict[int, slice], dict[int, int]]:
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
                internal_scattering_map[node_index,new_index] = i
                i += 1
            end = i
            internal_scattering_slices[node_index] = slice(start, end)

        i = 0
        for node in self.external_nodes:
            external_scattering_map[node.index] = i
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

    @property
    def spec(self) -> "NetworkSpec | None":
        """Return the NetworkSpec object that was used to create this network.
        
        Returns
        -------
        NetworkSpec | None
            The specification object describing how this network was constructed,
            or None if no spec was provided during network creation.
        """
        return self._network_spec

    # -------------------------------------------------------------------------
    # Basic utility functions
    # -------------------------------------------------------------------------

    def get_node(self, index: int | str) -> Node:
        """Returns the node with the specified index"""
        node = self.node_dict.get(index, None)
        if node is None:
            raise ValueError(f"Node {index} does not exist.")
        return node

    def get_link(self, index: int | str) -> Link:
        """Returns the link with the specified index"""
        link = self.link_dict.get(index, None)
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
    
    def get_connecting_nodes(self, node_index: int | str) -> list[Node]:
        """Returns a list of nodes that are connected to the node with the specified index"""
        nodes = []
        for link in self.link_dict.values():
            if node_index in link.node_indices:
                neighbor_index = (
                    link.node_indices[0]
                    if link.node_indices[1] == node_index
                    else link.node_indices[1]
                )

                node = self.node_dict.get(neighbor_index, None)
                nodes.append(node)

        return nodes

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
                old_to_new_nodes[key]: value
                for key, value in node.inwave.items()
            }
            node.outwave = {
                old_to_new_nodes[key]: value
                for key, value in node.outwave.items()
            }
            new_node_dict[node.index] = node

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
                old_to_new_nodes[key]: value
                for key, value in link.inwave.items()
            }
            link.outwave = {
                old_to_new_nodes[key]: value
                for key, value in link.outwave.items()
            }
            new_link_dict[link.index] = link

        self.node_dict = new_node_dict
        self.link_dict = new_link_dict


    #  _________________Altering the geometry of the network____________________________
 
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
            del node.inwave[other_node.index]
            node.inwave[new_node.index] = 0 + 0j
            del node.outwave[other_node.index]
            node.outwave[new_node.index] = 0 + 0j

        # Set up indices and other properties of the new node
        new_node.sorted_connected_nodes = sorted(
            [node_one.index, node_two.index]
        )
        new_node.sorted_connected_links = sorted(
            [new_link_one.index, new_link_two.index]
        )
        new_node.inwave = {
            node_one.index: 0 + 0j,
            node_two.index: 0 + 0j,
        }
        new_node.outwave = {
            node_one.index: 0 + 0j,
            node_two.index: 0 + 0j,
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
                node.index: 0 + 0j,
                new_node.index: 0 + 0j,
            }
            link.outwave = {
                node.index: 0 + 0j,
                new_node.index: 0 + 0j,
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
        self.node_dict[new_node.index] = new_node
        self.link_dict[new_link_one.index] = new_link_one
        self.link_dict[new_link_two.index] = new_link_two

        # Delete the old link from the dict
        del self.link_dict[link_index]

        # Update matrix maps and so on
        self.reset_dict_indices()
        self._reset_fields()
        self._set_matrix_calc_utils()

    def add_segment_to_link(
        self,
        link_index: int,
        fractional_positions: tuple[float, float],
        node_S_matrix_type: str = "fresnel",
    ) -> int:
        """Add a segment to a link.
        The segment looks like this

        O       |---------|    O       |---------|     O       |--------|    O
        node_one link_one insert_node1 middle_link insert_node2 link_two node_two

        
        parameters
        ----------
        link_index: int
            The index of the link to which the segment is added.
        fractional_positions: tuple[float, float]
            The fractional positions of the new nodes along the link in units of ratio of link length.
        Node_S_matrix_type: str
            The type of scattering matrix to be used for the new nodes. refractive index of new link
            is the same as the old link.
        
        Returns
        -------
        int
            The index of the new link added to the network. Also updates the network"""

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
        insert_node1_position = node_one.position + s1 * (
            node_two.position - node_one.position
        )
        insert_node2_position = node_one.position + s2 * (
            node_two.position - node_one.position
        )

        # Add first node and find its index
        self.add_node_to_link(link_index, s1)
        insert_node1_index = self.get_node_by_position(insert_node1_position).index

        # Find the link that connects the new node and node_two
        # Note: node_two_index may have changed!
        node_two_index = self.get_node_by_position(node_two_position).index
        second_link_index = self.get_link_by_node_indices(
            (insert_node1_index, node_two_index)
        ).index

        # Add second node and find its index
        # Ratio is its fractional position along the new link
        ratio = (s2 - s1) / (1 - s1)
        self.add_node_to_link(second_link_index, ratio)

        insert_node1_index = self.get_node_by_position(insert_node1_position).index
        insert_node2_index = self.get_node_by_position(
            insert_node2_position
        ).index
        middle_link = self.get_link_by_node_indices(
            (insert_node1_index, insert_node2_index)
        )

        self.update_segment_matrices(middle_link)
        return middle_link.index

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

    def update_node_scattering_matrix(
        self, node_index: int, node_S_mat_type: str, node_S_mat_params: dict
    ) -> None:
        """Update the S matrix of a node"""
        node = self.get_node(node_index)

        if node.node_type == "external":
            raise ValueError(
                "External nodes do not have modifiable scattering matrices."
            )
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

    def add_node(self, 
                 node_position: Tuple[float, float],
                 node_connections: List[int],
                 node_type: str = "internal",
                 new_get_S: Callable | None = None,
                 new_get_S_inv: Callable | None = None,
                 new_get_dS: Callable | None = None
                 )->None:
        
        """ This method adds a node to the position (x,y) and connects them to the nodes listed in
             node_connections
            parameters:
             node_position: Tuple[float, float]  position of the newly added node in the form (x,y)
             node_connections: List[int]  list of indices the newly added node will be connected to
             node_type: str  type of the node (internal or external)
             new_get_S: Callable | None  optional function to compute the S matrix
             new_get_S_inv: Callable | None  optional function to compute the inverse S matrix
             new_get_dS: Callable | None  optional function to compute the derivative S matrix
             """
        
        # Validate inputs
        if not node_connections:
            raise ValueError("node_connections cannot be empty")
        
        for node_index in node_connections:
            if node_index not in self.node_dict:
                raise ValueError(f"Node with index {node_index} does not exist to connect the new node to it")
        
        # Give default values if none are passed
        if new_get_S is None:
            from complex_network.scattering_matrices import node_matrix
            new_get_S = node_matrix.get_constant_node_S_closure(
                "neumann", len(node_connections), {}
            )
        if new_get_S_inv is None:
            new_get_S_inv = node_matrix.get_inverse_matrix_closure(new_get_S)
        if new_get_dS is None:
            new_get_dS = node_matrix.get_zero_matrix_closure(len(node_connections))
        
        # Create new node - assign it the highest available index to avoid conflicts
        new_node_index = max(self.node_dict.keys()) + 1 if self.node_dict else 0
        new_node = Node(
            new_node_index,
            node_type,
            node_position,
            data={
                "get_S": new_get_S,
                "get_S_inv": new_get_S_inv,
                "get_dS": new_get_dS,
                "S_mat_type": "neumann",
                "S_mat_params": {},
            },
        )
        
        # Set up node properties
        new_node.sorted_connected_nodes = sorted(node_connections)
        new_node.num_connect = len(node_connections)
        new_node.inwave = {node_idx: 0 + 0j for node_idx in node_connections}
        new_node.outwave = {node_idx: 0 + 0j for node_idx in node_connections}
        new_node.inwave_np = np.zeros(len(node_connections), dtype=np.complex128)
        new_node.outwave_np = np.zeros(len(node_connections), dtype=np.complex128)
        
        # Add node to network
        self.node_dict[new_node_index] = new_node
        
        # Get material properties from existing internal links (they should all have the same material)
        # This is the proper way to inherit material properties in the network
        material = None
        link_n = None
        link_dn = None
        link_Dn = 0.0
        
        if self.internal_links:
            # Use the material properties from the first internal link as reference
            reference_link = self.internal_links[0]
            material = getattr(reference_link, 'material', None)
            link_n = getattr(reference_link, 'n', lambda k0: 1.0)
            link_dn = getattr(reference_link, 'dn', lambda k0: 0.0)
            link_Dn = getattr(reference_link, 'Dn', 0.0)
        else:
            # Fallback to default values if no internal links exist yet
            link_n = lambda k0: 1.0
            link_dn = lambda k0: 0.0
            link_Dn = 0.0
        
        # Create links to connected nodes
        new_links = []
        for connected_node_index in node_connections:
            connected_node = self.get_node(connected_node_index)
            
            # Create new link - assign it the highest available index to avoid conflicts
            new_link_index = max(self.link_dict.keys()) + 1 + len(new_links) if self.link_dict else len(new_links)
            
            # Determine link type: external if either node is external, internal otherwise
            link_type = "external" if (node_type == "external" or connected_node.node_type == "external") else "internal"
            
            # Ensure node indices are always in sorted order
            sorted_node_indices = (min(new_node_index, connected_node_index), max(new_node_index, connected_node_index))
            
            new_link = Link(
                new_link_index, 
                link_type, 
                sorted_node_indices
            )
            
            # Calculate link length
            new_link.length = np.linalg.norm(
                new_node.position - connected_node.position
            )
            new_link.sorted_connected_nodes = sorted([new_node_index, connected_node_index])
            new_link.inwave = {
                new_node_index: 0 + 0j,
                connected_node_index: 0 + 0j,
            }
            new_link.outwave = {
                new_node_index: 0 + 0j,
                connected_node_index: 0 + 0j,
            }
            new_link.inwave_np = np.array([0 + 0j, 0 + 0j])
            new_link.outwave_np = np.array([0 + 0j, 0 + 0j])
            
            # Set material properties using the same approach as network_factory._initialise_links
            if material is not None:
                new_link.material = material
            new_link.n = link_n
            new_link.dn = link_dn
            new_link.Dn = link_Dn
            
            # Set up link scattering matrices
            new_link.get_S = link_matrix.get_propagation_matrix_closure(new_link)
            new_link.get_S_inv = link_matrix.get_propagation_matrix_inverse_closure(new_link)
            new_link.get_dS = link_matrix.get_propagation_matrix_derivative_closure(new_link)
            
            new_links.append(new_link)
            
            # Update connected node properties
            connected_node.sorted_connected_nodes.append(new_node_index)
            connected_node.sorted_connected_nodes.sort()
            connected_node.sorted_connected_links.append(new_link_index)
            connected_node.sorted_connected_links.sort()
            connected_node.num_connect += 1
            connected_node.inwave[new_node_index] = 0 + 0j
            connected_node.outwave[new_node_index] = 0 + 0j
            
            # Update numpy arrays for connected node
            connected_node.inwave_np = np.zeros(connected_node.num_connect, dtype=np.complex128)
            connected_node.outwave_np = np.zeros(connected_node.num_connect, dtype=np.complex128)
            
            # Update connected node's scattering matrices to handle new dimensionality
            # Generate new scattering matrices for the increased number of connections
            from complex_network.scattering_matrices import node_matrix
            connected_node.get_S = node_matrix.get_constant_node_S_closure(
                connected_node.S_mat_type, connected_node.num_connect, connected_node.S_mat_params
            )
            connected_node.get_S_inv = node_matrix.get_inverse_matrix_closure(connected_node.get_S)
            connected_node.get_dS = node_matrix.get_zero_matrix_closure(connected_node.num_connect)
        
        # Set up new node's connected links list
        new_node.sorted_connected_links = sorted([link.index for link in new_links])
        
        # Add links to network
        for link in new_links:
            self.link_dict[link.index] = link
        
        # Update network matrices and indices
        self.reset_dict_indices()
        self._reset_fields()
        self._set_matrix_calc_utils()
        
        # Fix any nodes that may have inconsistent scattering matrix settings
        # and regenerate all scattering matrices to ensure consistency
        # This must be done AFTER all index resets to override any permutation matrices
        for node in self.nodes:
            # Ensure we have proper attributes
            if not hasattr(node, 'S_mat_type') or node.S_mat_type == "custom":
                if not hasattr(node, 'S_mat_params') or "S_mat" not in node.S_mat_params:
                    node.S_mat_type = "neumann"
            if not hasattr(node, 'S_mat_params'):
                node.S_mat_params = {}
            
            # Completely replace the scattering matrix functions to override any permutation matrices
            node.get_S = node_matrix.get_constant_node_S_closure(
                node.S_mat_type, node.num_connect, node.S_mat_params
            )
            node.get_S_inv = node_matrix.get_inverse_matrix_closure(node.get_S)
            node.get_dS = node_matrix.get_zero_matrix_closure(node.num_connect)
        
        return None

    # You can only add links between existing internal nodes
    # links added will be internal links 

    def add_link(self,
                link_connections: List[Tuple[int, int]]
                ) -> None:
        
        """ Adds the links described in the link_connections list to the network
            parameters:
             link_connections: List[Tuple[int, int]]  list of tuples in the form (node1_index, node2_index)
             representing the links to be added
            """
        
        # Validate inputs
        if not link_connections:
            raise ValueError("link_connections cannot be empty")
        
        for node1_index, node2_index in link_connections:
            if node1_index not in self.node_dict:
                raise ValueError(f"Node with index {node1_index} does not exist")
            if node2_index not in self.node_dict:
                raise ValueError(f"Node with index {node2_index} does not exist")
            if node1_index == node2_index:
                raise ValueError("Cannot create link between a node and itself")
            
            # Check if link already exists
            for existing_link in self.links:
                if (existing_link.node_indices == (node1_index, node2_index) or 
                    existing_link.node_indices == (node2_index, node1_index)):
                    raise ValueError(f"Link between nodes {node1_index} and {node2_index} already exists")
        
        new_links = []
        
        for i, (node1_index, node2_index) in enumerate(link_connections):
            node1 = self.get_node(node1_index)
            node2 = self.get_node(node2_index)
            
            # Ensure node indices are always in sorted order
            sorted_node_indices = (min(node1_index, node2_index), max(node1_index, node2_index))
            
            # Create new link
            new_link_index = self.num_links + len(new_links)
            new_link = Link(
                new_link_index,
                "internal",  # Default to internal links
                sorted_node_indices
            )
            
            # Calculate link length
            new_link.length = np.linalg.norm(node1.position - node2.position)
            new_link.sorted_connected_nodes = sorted([node1_index, node2_index])
            new_link.inwave = {
                node1_index: 0 + 0j,
                node2_index: 0 + 0j,
            }
            new_link.outwave = {
                node1_index: 0 + 0j,
                node2_index: 0 + 0j,
            }
            new_link.inwave_np = np.array([0 + 0j, 0 + 0j])
            new_link.outwave_np = np.array([0 + 0j, 0 + 0j])
            
            # Copy material properties from existing links if they exist
            if self.links:
                # Get material properties from the first existing link
                existing_link = self.links[0]
                new_link.material = existing_link.material
                new_link.n = existing_link.n
                new_link.dn = existing_link.dn
                if hasattr(existing_link, 'Dn'):
                    new_link.Dn = existing_link.Dn
            
            # Set up link scattering matrices
            new_link.get_S = link_matrix.get_propagation_matrix_closure(new_link)
            new_link.get_S_inv = link_matrix.get_propagation_matrix_inverse_closure(new_link)
            new_link.get_dS = link_matrix.get_propagation_matrix_derivative_closure(new_link)
            
            new_links.append(new_link)
            
            # Update both connected nodes
            for node in [node1, node2]:
                other_node_index = node2_index if node.index == node1_index else node1_index
                
                # Add to connected nodes and links lists
                if other_node_index not in node.sorted_connected_nodes:
                    node.sorted_connected_nodes.append(other_node_index)
                    node.sorted_connected_nodes.sort()
                    node.num_connect += 1
                    
                node.sorted_connected_links.append(new_link_index)
                node.sorted_connected_links.sort()
                
                # Update wave dictionaries
                if other_node_index not in node.inwave:
                    node.inwave[other_node_index] = 0 + 0j
                if other_node_index not in node.outwave:
                    node.outwave[other_node_index] = 0 + 0j
                
                # Update numpy arrays
                node.inwave_np = np.zeros(node.num_connect, dtype=np.complex128)
                node.outwave_np = np.zeros(node.num_connect, dtype=np.complex128)
        
        # Add all new links to network
        for link in new_links:
            self.link_dict[link.index] = link
        
        # Update network matrices and indices
        self.reset_dict_indices()
        self._reset_fields()
        self._set_matrix_calc_utils()
        
        # Fix any nodes that may have inconsistent scattering matrix settings
        # and regenerate all scattering matrices to ensure consistency
        # This must be done AFTER all index resets to override any permutation matrices
        from complex_network.scattering_matrices import node_matrix
        for node in self.nodes:
            # Ensure we have proper attributes
            if not hasattr(node, 'S_mat_type') or node.S_mat_type == "custom":
                if not hasattr(node, 'S_mat_params') or "S_mat" not in node.S_mat_params:
                    node.S_mat_type = "neumann"
            if not hasattr(node, 'S_mat_params'):
                node.S_mat_params = {}
            
            # Completely replace the scattering matrix functions to override any permutation matrices
            node.get_S = node_matrix.get_constant_node_S_closure(
                node.S_mat_type, node.num_connect, node.S_mat_params
            )
            node.get_S_inv = node_matrix.get_inverse_matrix_closure(node.get_S)
            node.get_dS = node_matrix.get_zero_matrix_closure(node.num_connect)
        
        return None
    
    # maybe add a method to remove nodes or links in the future
    # Maybe also add a method to change the connections between existing nodes

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
            node.inwave[node.sorted_connected_nodes[1]] = value
            node.inwave_np[1] = value

            connected_link.outwave[node_index] = value
            connected_link.outwave_np[1] = value

            # Set incoming external values
            value = incoming_external[count]
            node_index = node.index
            connected_link_index = node.sorted_connected_links[0]
            connected_link = self.get_link(connected_link_index)

            node.inwave[-1] = value
            node.inwave_np[0] = value
            node.outwave[node.sorted_connected_nodes[1]] = value
            node.outwave_np[1] = value

            connected_link.inwave[node_index] = value
            connected_link.inwave_np[1] = value

            count += 1
        # Set internal node values
        count = 0
        for node in self.internal_nodes:
            for i, connected_index in enumerate(node.sorted_connected_nodes):
                incoming_value = incoming_internal[count]
                node.inwave[connected_index] = incoming_value
                node.inwave_np[i] = incoming_value

                outgoing_value = outgoing_internal[count]
                node.outwave[connected_index] = outgoing_value
                node.outwave_np[i] = outgoing_value

                count += 1

        # Set internal link values
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            node_one = self.get_node(node_one_index)
            node_two = self.get_node(node_two_index)

            # Set link fields
            link.inwave[node_one_index] = node_one.outwave[node_two_index]

            link.inwave_np[0] = node_one.outwave[node_two_index]

            link.inwave[node_two_index] = node_two.outwave[node_one_index]

            link.inwave_np[1] = node_two.outwave[node_one_index]

            # Outwaves
            link.outwave[node_one_index] = node_one.inwave[node_two_index]

            link.outwave_np[0] = node_one.inwave[node_two_index]

            link.outwave[node_two_index] = node_two.inwave[node_one_index]
            
            link.outwave_np[1] = node_two.inwave[node_one_index]

        # Remaining external links values
        for link in self.external_links:
            external_index, node_index = link.node_indices
            node = self.get_node(node_index)

            # Set link fields
            link.inwave[node_index] = node.outwave[external_index]
            link.inwave_np[1] = node.outwave[external_index]

            link.outwave[node_index] = node.inwave[external_index]
            link.outwave_np[1] = node.inwave[external_index]

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
                self.inwave[external_node.index] = incident_field[i]
                external_node.inwave["-1"] = incident_field[i]
                external_node.inwave_np[0] = incident_field[i]
            elif direction == "backward":
                self.outwave[external_node.index] = incident_field[i]
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
                self.outwave[node.index] = node.outwave["-1"]
                self.outwave_np[i] = node.outwave["-1"]
            if direction == "backward":
                self.inwave[node.index] = node.inwave["-1"]
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
            new_slice = self.internal_scattering_slices[node_index]
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
            row = self.internal_scattering_map[(node_one_index,node_two_index) ]
            col = self.internal_scattering_map[
                (node_two_index,node_one_index)
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
            row = self.external_scattering_map[node_two_index]
            col = self.internal_scattering_map[
                (node_one_index,node_two_index)
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
        bracket = np.eye(len(S_ii), dtype=np.complex128) - S_ii @ P_ii
        # S_ee = P_ei @ inv(bracket) @ S_ii @ P_ie

        # Instead of computing the inverse, solve the linear system bracket@x = S_ii@P_ie such that x = bracket^-1@S_ii@P_ie 
        # for faster implementation
        solution = np.linalg.solve(bracket, S_ii@P_ie)
        S_ee = P_ei @ solution

        return S_ee

    def get_S_ee_inv(self, k0: float | complex) -> np.ndarray:
        """Get the external inverse scattering matrix from the inverse
        formula"""
        # S_ee_inv = P_ei_inv @ S_ii_inv @ inv @ P_ie_inv

        # Instead of computing the inverse, call the S_ee function. Since the See matrix is often small
        # it is better we use the func that is already optimized and invert it
        S_ee = self.get_S_ee(k0)
        S_ee_inv = np.linalg.inv(S_ee)
        

        return S_ee_inv

    def get_S_ie(self, k0: float | complex) -> np.ndarray:
        """Get the external scattering matrix from the inverse formula

        (O_i, I_i) = S_ie @ I_e"""
        P_ie = self.get_P_ie(k0)
        S_ii = self.get_S_ii(k0)
        P_ii = self.get_P_ii(k0)

        S_ii_shape = S_ii.shape[0]

        # Bracketed part to be inverted
        bracket_top = np.eye(S_ii_shape, dtype=np.complex128) - S_ii @ P_ii
        bracket_bottom = np.eye(S_ii_shape, dtype=np.complex128) - P_ii @ S_ii

        top = np.linalg.solve(bracket_top, S_ii @ P_ie)
        bottom = np.linalg.solve(bracket_bottom, P_ie)

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
            new_slice = self.internal_scattering_slices[node_index]
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
                (node_one_index,node_two_index)
            ]
            col = self.internal_scattering_map[
                (node_two_index,node_one_index)
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
                (node_one_index,node_two_index)
            ]
            col = self.internal_scattering_map[
                (node_two_index,node_one_index)
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
            row = self.external_scattering_map[node_two_index]
            col = self.internal_scattering_map[
                (node_one_index,node_two_index)
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
            row = self.external_scattering_map[node_two_index]
            col = self.internal_scattering_map[
                (node_one_index,node_two_index)
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
    
    def get_RT_matrix(self,k0: float | complex) -> np.ndarray:
        """Calculate the reflection and transmission matrix of the scattering matrix which is valid for slab geometries
            Convert the scattering matrix to the slab scattering matrix.
            The slab scattering matrix is defined as the scattering matrix of a slab of material
            but the terms are organized as [[r,t']
                                            [t,r']]
            First column: Response to an incoming wave from the left r (reflection from the left) t (transmission from left to right).
            Second column: Response to an incoming wave from the right r (reflection from the right) t (transmission from right to left).

            The left nodes have +ve coordinates and the right nodes have -ve coordinates."""

        
        external_scattering_map = self.external_scattering_map
        port_to_node = {v: k for k, v in external_scattering_map.items()}
        
        left_ports = []
        right_ports = []

        # Sort the ports into left and right
        for port in port_to_node:
            node = self.get_node(port_to_node[port])
            if node.position[0] > 0:
                left_ports.append(port)
            else:
                right_ports.append(port)

        left = np.array(left_ports)
        right = np.array(right_ports)

        S = self.get_S_ee(k0)

        # Extract submatrices using ix_ to handle index arrays correctly
        r = S[np.ix_(left, left)] if left.size else np.empty((0, 0))
        t_prime = S[np.ix_(left, right)] if left.size and right.size else np.empty((left.size, right.size))
        t = S[np.ix_(right, left)] if right.size and left.size else np.empty((right.size, left.size))
        r_prime = S[np.ix_(right, right)] if right.size else np.empty((0, 0))
        
        # Construct the block matrix using numpy's block function
        block_matrix = np.block([[r, t_prime], [t, r_prime]])
        
        return block_matrix
    
    def get_reflection_matrix(self, k0: float | complex) -> np.ndarray:
        """Calculate the reflection matrix of the scattering matrix which is valid for slab geometries
            Convert the scattering matrix to the slab scattering matrix.
            The slab scattering matrix is defined as the scattering matrix of a slab of material
            but the terms are organized as [[r,t']
                                            [t,r']]
            First column: Response to an incoming wave from the left r (reflection from the left) t (transmission from left to right).
            Second column: Response to an incoming wave from the right r (reflection from the right) t (transmission from right to left).
            The left nodes have +ve coordinates and the right nodes have -ve coordinates."""
    
        external_scattering_map = self.external_scattering_map

        port_to_node = {v: k for k, v in external_scattering_map.items()}
        
        left_ports = []
        right_ports = []

        # Sort the ports into left and right
        for port in port_to_node:
            node = self.get_node(port_to_node[port])
            if node.position[0] > 0:
                left_ports.append(port)
            else:
                right_ports.append(port)

        left = np.array(left_ports)

        S = self.get_S_ee(k0)

        # Extract submatrices using ix_ to handle index arrays correctly
        r = S[np.ix_(left, left)] if left.size else np.empty((0, 0))

        return r
    
    def get_transmission_matrix(self, k0: float | complex) -> np.ndarray:
        """Calculate the transmission matrix of the scattering matrix which is valid for slab geometries
            Convert the scattering matrix to the slab scattering matrix.
            The slab scattering matrix is defined as the scattering matrix of a slab of material
            but the terms are organized as [[r,t']
                                            [t,r']]
            First column: Response to an incoming wave from the left r (reflection from the left) t (transmission from left to right).
            Second column: Response to an incoming wave from the right r (reflection from the right) t (transmission from right to left).
            The left nodes have +ve coordinates and the right nodes have -ve coordinates."""
        external_scattering_map = self.external_scattering_map
        port_to_node = {v: k for k, v in external_scattering_map.items()}
        
        left_ports = []
        right_ports = []

        # Sort the ports into left and right
        for port in port_to_node:
            node = self.get_node(port_to_node[port])
            if node.position[0] > 0:
                left_ports.append(port)
            else:
                right_ports.append(port)

        left = np.array(left_ports)
        right = np.array(right_ports)

        S = self.get_S_ee(k0)

        t = S[np.ix_(right, left)] if right.size and left.size else np.empty((right.size, left.size))

        return t


    def get_internal_link_fields(self, k0: float | complex, I_e: np.ndarray) -> np.ndarray:
        """
        Compute the fields inside internal links given wavenumber k0 and external input I_e.
        
        Args:
            k0 (float | complex): Wavenumber.
            I_e (np.ndarray): Incoming external field vector, shape (num_external_nodes,).
        
        Returns:
            np.ndarray: Array of shape (num_internal_links, 2) where each row contains
                        [field from node A to B, field from node B to A] for an internal link.
        """
        S_ie = self.get_S_ie(k0)
        internal_vector = S_ie @ I_e
        O_i = internal_vector[:self.internal_vector_length]
        fields_A_to_B = O_i[self.internal_link_indices_A_to_B]
        fields_B_to_A = O_i[self.internal_link_indices_B_to_A]
        internal_fields = np.column_stack((fields_A_to_B, fields_B_to_A))
        return internal_fields

    def get_all_link_energy_densities(self, k0: float | complex, I_e: np.ndarray) -> np.ndarray:
        """
        Compute the average energy density inside all internal links using the provided formula.

        Args:
            k0 (float | complex): Wavenumber.
            I_e (np.ndarray): Incoming external field vector, shape (num_external_nodes,).

        Returns:
            np.ndarray: Array of shape (num_internal_links,) containing the average energy density for each internal link.
        """
        # Get precomputed internal fields for all links
        internal_fields = self.get_internal_link_fields(k0, I_e)  # Shape: (num_internal_links, 2)
        inwave = internal_fields[:, 0]  # Forward fields (psi_A_to_B), shape: (num_internal_links,)
        outwave = internal_fields[:, 1]  # Backward fields (psi_B_to_A), shape: (num_internal_links,)

        # Get link lengths
        lengths = np.array([link.length for link in self.internal_links])  # Shape: (num_internal_links,)

        # Extract amplitudes and phases
        r1 = np.abs(inwave,dtype=np.float64)  # Shape: (num_internal_links,)
        r2 = np.abs(outwave,dtype=np.float64)  # Shape: (num_internal_links,)
        theta1 = np.angle(inwave)  # Shape: (num_internal_links,)
        theta2 = np.angle(outwave)  # Shape: (num_internal_links,)

        # Compute amplitude term
        amplitude_term = r1**2 + r2**2  # Shape: (num_internal_links,)

        # Compute interference term
        interference_term = (
            2 * r1 * r2 * np.cos(k0 * lengths + theta1 - theta2) * np.sin(k0 * lengths) / k0
        )  # Shape: (num_internal_links,)

        # Compute energy density
        energy_density = amplitude_term + interference_term / lengths  # Shape: (num_internal_links,)

        return energy_density

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
            new_slice = self.internal_scattering_slices[node_index]
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
                (node_one_index, node_two_index)
            ]
            col = self.internal_scattering_map[
                (node_two_index, node_one_index)
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
            row = self.external_scattering_map[node_two_index]
            col = self.internal_scattering_map[
                (node_one_index,node_two_index)
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
            new_O = outgoing_vector[: len(outgoing_vector) // 2]
            new_I = outgoing_vector[len(outgoing_vector) // 2 :]
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
            new_O = outgoing_vector[: len(outgoing_vector) // 2]
            new_I = outgoing_vector[len(outgoing_vector) // 2 :]
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
            new_O = outgoing_vector[: len(outgoing_vector) // 2]
            new_I = outgoing_vector[len(outgoing_vector) // 2 :]
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

            new_O = outgoing_vector[: len(outgoing_vector) // 2]
            new_I = outgoing_vector[len(outgoing_vector) // 2 :]
            new_dO = doutgoing_vector[: len(outgoing_vector) // 2]
            new_dI = doutgoing_vector[len(outgoing_vector) // 2 :]

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
            new_O = outgoing_vector[: len(outgoing_vector) // 2]
            new_I = outgoing_vector[len(outgoing_vector) // 2 :]
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
            new_O = outgoing_vector[: len(outgoing_vector) // 2]
            new_I = outgoing_vector[len(outgoing_vector) // 2 :]
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

            new_O = outgoing_vector[: len(outgoing_vector) // 2]
            new_I = outgoing_vector[len(outgoing_vector) // 2 :]
            new_dO = doutgoing_vector[: len(outgoing_vector) // 2]
            new_dI = doutgoing_vector[len(outgoing_vector) // 2 :]

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

    # ______________________plotting methods_______________________________

    def draw(
        self,
        ax=None,
        show_indices: bool = False,
        show_external_indices: bool = False,
        show_internal_indices: bool = False,
        equal_aspect: bool = False,
        highlight_nodes: list[int] | None = None,
        highlight_links: list[int] | None = None,
        highlight_perturbed_nodes: bool = True,
        highlight_perturbed_links: bool = True,
        hide_axes: bool = False,
        draw_boundary: float | tuple[float, float] | None = None,
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
            if isinstance(draw_boundary, float):
                t = np.linspace(-draw_boundary, draw_boundary, 10**6)
                y = np.sqrt(draw_boundary**2 - t**2)
                linewidth = 1
                ax.plot(t, y, linestyle="--", color="black")
                ax.plot(t, -y, linestyle="--", color="black")
            elif isinstance(draw_boundary, tuple) and len(draw_boundary) == 2:
                x1 = draw_boundary[0] / 2
                x2 = -draw_boundary[0] / 2
                y1 = draw_boundary[1] / 2
                y2 = -draw_boundary[1] / 2
                linewidth = 1
                ax.plot([x1, x1], [y1, y2], linestyle="--", color="black")
                ax.plot([x2, x2], [y1, y2], linestyle="--", color="black")
                ax.plot([x1, x2], [y1, y1], linestyle="--", color="black")
                ax.plot([x1, x2], [y2, y2], linestyle="--", color="black")

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
            node.draw(ax, show_indices, show_external_indices, show_internal_indices, color=color)

        # Highlight nodes
        # if highlight_nodes is not None:
        #     for node_index in highlight_nodes:
        #         node = self.get_node(node_index)
        #         node.draw(ax, color="red")

        # Custom highlighting for perturbations
        if highlight_perturbed_nodes:
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
                            show_internal_indices,
                            color="red",
                            markersize=1 + (6 - 1) * r,
                        )
                    case _:
                        pass
            if highlight_perturbed_links:
                for link in self.links:
                    if not link.is_perturbed:
                        continue
                    node_1_index, node_2_index = link.node_indices
                    node_1_pos = self.get_node(node_1_index).position
                    node_2_pos = self.get_node(node_2_index).position
                    color = "red"
                    link.draw(ax, node_1_pos, node_2_pos, color=color)

                    # draw the nodes at the ends of the links(because the perturbed red line comes above the nodes and looks bad)
                    node_1 = self.get_node(node_1_index)
                    node_2 = self.get_node(node_2_index)
                    node_1.draw(ax, show_index = False, show_external_index = False, show_internal_index = False, color=None)
                    node_2.draw(ax, show_index = False, show_external_index = False, show_internal_index = False, color=None)

        # Set scientific notation for axes
        if not hide_axes:
            from matplotlib.ticker import ScalarFormatter
            formatter = ScalarFormatter(useMathText=True)
            formatter.set_scientific(True)
            formatter.set_powerlimits((-1, 1))  # Use scientific notation for values outside 0.1 to 10
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

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
            inwave = link.inwave[node_index]
            outwave = link.outwave[node_index]
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

    # ____________________Spatial Network Property __________________________
    # Write a function that given the link_index or node_index or fractional ratio returns the position in space
    def spatial_position_within_link(self,
                              link_index: int | None = None,
                              node_tuple: Tuple[int, int] | None = None,
                              fractional_ratio: float | None = None,
                              ) -> np.ndarray:
        """Given a link index or the node tuple of the link
         and fractional ratio, return the spatial position in the network"""
        if link_index is None and node_tuple is None:
            raise ValueError("Both link_index and node_tuple cannot be None. Provide at least one.")
        if fractional_ratio is None:
            raise ValueError("fractional_ratio cannot be None. Provide a value between 0 and 1.")
        if not (0.0 <= fractional_ratio <= 1.0):
            raise ValueError("fractional_ratio must be between 0 and 1.")
        if link_index is None:
            link = self.get_link_by_node_indices(node_tuple)
        elif node_tuple is None:
            link = self.get_link(link_index)
        else:
            # Check consistency
            link = self.get_link(link_index)
            if set(link.sorted_connected_nodes) != set(sorted(node_tuple)):
                raise ValueError("Provided link_index and node_tuple do not correspond to the same link.")

        node_1_index, node_2_index = link.sorted_connected_nodes
        node_1_pos_x, node_1_pos_y = self.get_node(node_1_index).position[0], self.get_node(node_1_index).position[1]
        node_2_pos_x, node_2_pos_y = self.get_node(node_2_index).position[0], self.get_node(node_2_index).position[1]
        delta_x = node_2_pos_x - node_1_pos_x
        delta_y = node_2_pos_y - node_1_pos_y

        position_x = node_1_pos_x + fractional_ratio * delta_x
        position_y = node_1_pos_y + fractional_ratio * delta_y

        return np.array([position_x, position_y])

    # -------------------------------------------------------------------------
    #  Network properties and analysis methods
    # -------------------------------------------------------------------------
    @property
    def laplacian_matrix(self, ):
        """
        Returns Laplacian matrix for network
        https://en.wikipedia.org/wiki/Laplacian_matrix

        """
        A = self.adjacency_matrix
        D = self.degree_matrix
        return D - A

    @property
    def degree_matrix(self, ):
        """
        Returns degree matrix for network
        https://en.wikipedia.org/wiki/Degree_matrix

        Matrix is ordered according to increasing node index

        """
        deg = np.zeros((self.num_nodes, self.num_nodes))

        # sort nodes
        sorted_nodes = sorted(self.node_dict.keys())

        # construct adjacency matrix
        for index, n_id in enumerate(sorted_nodes):
            node = self.get_node(n_id)
            deg[index, index] = node.degree

        return deg

    @property
    def adjacency_matrix(self, ):
        """
        Returns adjacency matrix for network
        https://en.wikipedia.org/wiki/Adjacency_matrix

        Matrix is ordered according to increasing node index

        """
        adj = np.zeros((self.num_nodes, self.num_nodes))

        # sort nodes
        sorted_nodes = sorted([key for key in self.node_dict.keys()])

        # construct adjacency matrix
        for index, node_index in enumerate(sorted_nodes):
            connected_indices = [node.index for node in self.get_connecting_nodes(node_index)]
            for connected_index in connected_indices:
                adj[index, sorted_nodes.index(connected_index)] = 1

        return adj

    @property
    def fiedler(self, ):
        """
        Returns Fiedler value or algebraic connectivity for network
        https://en.wikipedia.org/wiki/Algebraic_connectivity
        """

        L = self.laplacian_matrix
        eigensystem = np.linalg.eig(L)
        eigenvalues = eigensystem[0]
        eigenvectors = eigensystem[1]
        sorted_eigenvalues = sorted(eigenvalues)

        f = sorted_eigenvalues[1]
        fv = eigenvectors[list(eigenvalues).index(f)]
        return f, fv

    """"Write a version if possible that accounts memory usage optimization too
        the current implementation will fill up memory for large networks. Some 
        cut off should be implemented to avoid this."""
    def breadth_first_search_simple_paths(
            self,
            start_node: int,
            end_node:   int,
            max_path_length: int | float = np.inf,
            *,
            use_mult_proc: bool = False,
            min_tasks_per_core: int = 4,
            hard_parallel_depth: int | None = None):
        """
        Enumerate *all* simple paths between `start_node` and `end_node`.

        parameters
        
        start_node: int
            Index of the node to start from.
        end_node: int
            Index of the node to end at.
        max_path_length: int | float
            Maximum length of the paths to be found. If set to `np.inf`, no limit    
        use_mult_proc: bool
            If `True`, use multiprocessing to speed up the search.
        min_tasks_per_core: int
            Minimum number of tasks per core to use when multiprocessing.
        hard_parallel_depth: int | None
            If set, the search will stop at this depth to avoid excessive parallelization.
        """
        # trivial case
        if start_node == end_node:
            return [np.asarray([start_node], int)]

        # build adjacency
        adjacency = {n.index: set() for n in self.nodes}
        for link in self.link_dict.values():
            a, b = link.node_indices
            adjacency[a].add(b);  adjacency[b].add(a)

        # single-threaded BFS
        def _bfs_cpu():
            dq, out = deque([(start_node, [start_node])]), []
            while dq:
                node, path = dq.popleft()
                if node == end_node:
                    out.append(path);  continue
                if len(path) > max_path_length:
                    continue
                for nbr in adjacency[node]:
                    if nbr not in path:
                        dq.append((nbr, path + [nbr]))
            return [np.asarray(p, int) for p in out]

        if not use_mult_proc:
            return _bfs_cpu()

        # seed geenrator for parallel BFS
        def _make_frontier():
            frontier = [(nbr, [start_node, nbr]) for nbr in adjacency[start_node]]
            complete = []        # finished paths encountered so far
            depth    = 1

            target = cpu_count() * min_tasks_per_core
            while frontier:
                # stop if we already have “enough” parallel work
                if hard_parallel_depth is not None and depth >= hard_parallel_depth:
                    break
                if len(frontier) >= target:
                    break

                next_frontier = []
                for node, path in frontier:
                    if node == end_node:
                        complete.append(path);          # save finished path
                        continue
                    if len(path) >= max_path_length:
                        continue
                    for nbr in adjacency[node]:
                        if nbr not in path:
                            next_frontier.append((nbr, path + [nbr]))
                if not next_frontier:
                    break
                frontier = next_frontier
                depth   += 1
            return frontier, complete

        seeds, finished_in_seeder = _make_frontier()
        if not seeds:   
            return [np.asarray(p, int) for p in finished_in_seeder]

        # Randomly shuffle the seeds to balance the load
        np.random.shuffle(seeds)
        n_workers  = cpu_count()
        chunk_goal = n_workers * min_tasks_per_core
        chunksize  = max(1, len(seeds) // chunk_goal)
        chunksize  = min(chunksize, 128)

        with Pool(n_workers,
                initializer=_init_pool,
                initargs=(adjacency, end_node, max_path_length)) as pool:
            results = pool.imap_unordered(_bfs_worker, seeds, chunksize)
            worker_paths = [p for grp in results for p in grp]

        return [np.asarray(p, int) for p in (finished_in_seeder + worker_paths)]


    def get_lengths_along_path(self, path_indices):
        lengths = []
        for jj, index1 in enumerate(path_indices[:-1]):
            index2 = path_indices[jj+1]

            link = self.get_link_by_node_indices((index1, index2))
            lengths.append(link.length)

        return np.array(lengths)

    def get_path_length(self, path_indices):
        """Calculate the total length of a given path in the network."""
        lengths = self.get_lengths_along_path(path_indices)
        return np.sum(lengths)

    def get_optical_path_length(self, path_indices: List[int], k0: float | complex = 1e7):
        """Calculate the optical path length along a given path in the network."""

        # The default k value is set to 1e7 must be changed when accounting for dispersion
        optical_path_length = []
        for jj, index1 in enumerate(path_indices[:-1]):
            index2 = path_indices[jj + 1]

            link = self.get_link_by_node_indices((index1, index2))
            n = link.n(k0)
            Dn = link.Dn
            length = link.length
            optical_path_length.append(length * (n + Dn))

        total_opl = np.sum(optical_path_length)
        return total_opl


# Helper functions outside the Network class that are used for multiprocessing
# Global placeholders that every worker process can reach
_adj = _end_node_glob = _max_len_glob = None
def _init_pool(adj, end_node, max_len):
    global _adj, _end_node_glob, _max_len_glob
    _adj, _end_node_glob, _max_len_glob = adj, end_node, max_len

def _bfs_worker(seed):
    node, path = seed
    done, dq = [], deque([(node, path)])
    while dq:
        v, cur = dq.popleft()
        if v == _end_node_glob:
            done.append(cur);  continue
        if len(cur) > _max_len_glob:
            continue
        for nxt in _adj[v]:
            if nxt not in cur:
                dq.append((nxt, cur + [nxt]))
    return done