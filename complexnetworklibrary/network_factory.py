"""Factory module for building networks"""

# setup code logging
import logging
import math
import random
import warnings
from typing import Any

import numpy as np
import scipy
from scipy.spatial import ConvexHull, Voronoi
from complexnetworklibrary.spec import NetworkSpec
from complexnetworklibrary.components.node import Node
from complexnetworklibrary.components.link import Link
from complexnetworklibrary.network import Network
import logconfig


# from line_profiler_pycharm import profile


logconfig.setup_logging()
logger = logging.getLogger(__name__)


def generate_network(spec: NetworkSpec) -> Network:
    """Main method for building a network"""

    # Get nodes and links
    # Links will have
    match spec.network_type:
        case "delaunay":
            nodes, links = _generate_delaunay_nodes_links(spec)
        case _:
            raise ValueError(f"network_type '{spec.network_type}' is invalid.")

    _initialise_links(nodes, links, spec)
    _initialise_nodes(nodes, links, spec)
    return Network(nodes, links)


def _initialise_nodes(
    nodes: dict[str, Node], links: dict[str, Link], spec: NetworkSpec
) -> None:
    """Set initial values for links in the network"""
    # First, tell the nodes which links and nodes are connected to them
    for link in links.values():
        node_index_one, node_index_two = link.node_indices

        # Both links are connected to the node
        nodes[str(node_index_one)].sorted_connected_links.append(link.index)
        nodes[str(node_index_two)].sorted_connected_links.append(link.index)

        # Both nodes are also connected to each other
        nodes[str(node_index_one)].sorted_connected_nodes.append(
            node_index_two
        )
        nodes[str(node_index_two)].sorted_connected_nodes.append(
            node_index_one
        )

    for node in nodes.values():
        # Add "ghost" exit channel for exit nodes
        if node.node_type == "exit":
            node.sorted_connected_nodes.append(-1)

        # Sort lists and get the length
        node.sorted_connected_links = sorted(node.sorted_connected_links)
        node.sorted_connected_nodes = sorted(node.sorted_connected_nodes)
        num_connect = len(node.sorted_connected_nodes)
        node.num_connect = num_connect
        size = num_connect

        # Set up in and out waves

        for second_node in node.sorted_connected_nodes:
            node.inwave[str(second_node)] = 0 + 0j
            node.outwave[str(second_node)] = 0 + 0j

        node.inwave_np = np.zeros(size, dtype=np.complex128)
        node.outwave_np = np.zeros(size, dtype=np.complex128)

        # Set up scattering matrices
        if node.node_type == "exit":
            # This matrix just transfers power onwards
            node.S_mat = np.array(
                [[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128
            )
            node.iS_mat = np.array(
                [[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128
            )
            continue

        node.S_mat_params = spec.node_S_mat_params
        node.S_mat = get_S_mat(spec.node_S_mat_params, size)
        node.iS_mat = np.linalg.inv(node.S_mat)


def _initialise_links(
    nodes: dict[str, Node], links: dict[str, Link], spec: NetworkSpec
) -> None:
    """Set initial values for links in the network"""
    for link in links.values():
        # Get nodes
        node_index_one, node_index_two = link.node_indices
        node_one = nodes[str(node_index_one)]
        node_two = nodes[str(node_index_two)]

        # Determine the link length
        length = np.linalg.norm(node_two.position - node_one.position)
        link.length = length

        link.sorted_connected_nodes = sorted(link.node_indices)

        link.inwave = {
            str(node_index_one): 0 + 0j,
            str(node_index_two): 0 + 0j,
        }
        link.outwave = {
            str(node_index_one): 0 + 0j,
            str(node_index_two): 0 + 0j,
        }
        link.update_S_matrices()


def _generate_delaunay_nodes_links(spec: NetworkSpec) -> tuple[dict, dict]:
    """
    Generates a Delaunay type network formed from delaunay triangulation

    Parameters
    ----------
    spec : Dictionary specifying properties of network:
        Keys:
            'internal_nodes': number of internal nodes,
            'exit_nodes': number of external nodes,
            'shape': 'circular' or 'slab'
            'network_size':
                for 'circular': radius of network
                for 'slab': tuple defining (length,width) of rectangular
                             network
            'wavenumber': k,
            'refractive_index': n,
            'exit_size':
                for 'circular': radius of exit nodes from network center
                for 'slab': exit nodes placed at +/-exit_size/2 randomly
                             within width

            'left_exit_fraction': in range [0,1]. Fraction of exit nodes on
                                 lefthand side
                of a slab network. Not needed for circular
    """

    num_internal_nodes = spec.num_internal_nodes
    num_exit_nodes = spec.num_exit_nodes

    network_shape = spec.network_shape
    network_size = spec.network_size
    exit_size = spec.exit_size

    nodes = {}
    links = {}

    if network_shape == "circular":
        # Check type of network_size
        if not isinstance(network_size, float):
            raise ValueError(
                "network_size must be a float for a circular delaunay network"
            )
        # Check type of network_size
        if not isinstance(exit_size, float):
            raise ValueError(
                "exit_size must be a float for a circular delaunay network"
            )
        # Radius of exit nodes must be bigger than radius of internal nodes
        if exit_size <= network_size:
            raise ValueError("exit_size must be larger than network_size")

        # Generate random points
        # Points on the outer circles
        theta_exit = 2 * np.pi * np.random.random(num_exit_nodes)
        points_exit = (
            exit_size * np.array([np.cos(theta_exit), np.sin(theta_exit)]).T
        )
        points_edge = (
            network_size * np.array([np.cos(theta_exit), np.sin(theta_exit)]).T
        )

        # Points in the interior
        theta_int = (
            2 * np.pi * np.random.random(num_internal_nodes - num_exit_nodes)
        )
        r_int = network_size * np.sqrt(
            np.random.random(num_internal_nodes - num_exit_nodes)
        )
        points_int = np.array(
            [r_int * np.cos(theta_int), r_int * np.sin(theta_int)]
        ).T

        # All non-exit points
        points_internal = np.vstack((points_edge, points_int))
        for i, point in enumerate(points_internal):
            nodes[str(i)] = Node(i, "internal", point)

        # Triangulate nodes
        delaunay = scipy.spatial.Delaunay(points_internal)

        # Loop over triangles adding relevant links
        link_index = 0
        created_links = set()
        for i, simplex in enumerate(delaunay.simplices):
            for index in range(0, 3):
                cur_node = simplex[index]
                next_node = simplex[(index + 1) % 3]
                node_pair = tuple(sorted((cur_node, next_node)))

                # Add new node and link to list
                if node_pair not in created_links:
                    links[str(link_index)] = Link(
                        link_index, "internal", node_pair
                    )
                    link_index += 1
                    created_links.add(node_pair)

        # Finally, add exit nodes and link them to the edge nodes
        node_start = len(nodes)
        link_start = len(links)
        for i, point_exit in enumerate(points_exit):
            node_index = node_start + i
            link_index = link_start + i
            # Note that node i is the i'th edge mode, which lines up with the
            # i'th exit ndoe
            node_pair = tuple(sorted((node_index, i)))
            nodes[str(node_index)] = Node(node_index, "exit", point_exit)
            links[str(link_index)] = Link(link_index, "exit", node_pair)

    return nodes, links


def _generate_voronoi_network(network_spec: NetworkSpec):
    pass


def _generate_buffon_network(network_spec: NetworkSpec):
    pass


def _generate_linear_network(network_spec: NetworkSpec):
    pass


def _generate_archimedian_network(network_spec: NetworkSpec):
    pass


# -----------------------------------------------------------------------------
# Scattering matrices
# -----------------------------------------------------------------------------
def get_S_mat(S_mat_params: dict[str, Any], size: int) -> np.ndarray:
    """Generate a random node scattering matrix of a given size.

    S_mat_params must contain at least "S_mat_type". Options are
        'identity':
            identity matrix - complete reflection at each input
        'permute_identity' :
            permuted identity matrix - rerouting to next edge
        'uniform':
            each element takes a value in [0,1)
        'isotropic_unitary':
            unitary isotropic SM, implemented through DFT matrix of correct
            dimension
        'COE' :
            drawn from circular orthogonal ensemble
        'CUE' :
            drawn from circular unitary ensemble
        'unitary_cyclic':
            unitary cyclic SM constructed through specifying phases of
            eigenvalues using 'delta'
        'to_the_lowest_index':
            reroutes all energy to connected node of lowest index
        'custom' :
            Set a custom scattering matrix. Requires kwarg 'S_mat' to be set
    """
    S_mat_type = S_mat_params.get("S_mat_type")
    valid_S_mat_types = [
        "identity",
        "uniform_random",
        "isotropic_unitary",
        "CUE",
        "COE",
        "permute_identity",
        "custom",
        "unitary_cyclic",
    ]

    match S_mat_type:
        case "identity":
            S_mat = np.identity(size, dtype=np.complex128)

        case "gaussian_random":
            S_mat = np.random.random((size, size))

        case "isotropic_unitary":
            S_mat = scipy.linalg.dft(size) / np.sqrt(size)

        case "CUE":
            gamma = S_mat_params.get("subunitary_factor", 1.0)
            S_mat = scipy.stats.unitary_group.rvs(size) * gamma

        case "COE":
            gamma = S_mat_params.get("subunitary_factor", 1.0)
            S_mat = scipy.stats.unitary_group.rvs(size) * gamma

            S_mat = S_mat @ S_mat.T
        case "permute_identity":
            mat = np.identity(size, dtype=np.complex_)
            inds = [(i - 1) % size for i in range(size)]
            S_mat = mat[:, inds]

        case "custom":
            S_mat = S_mat_params.get("S_mat", np.array(0))
            if S_mat.shape != (size, size):
                raise ValueError(
                    "Supplied scattering matrix is of incorrect"
                    f"Given: {S_mat.shape}"
                    f"Expected: {(size, size)}"
                )

        case "unitary_cyclic":
            delta = S_mat_params.get("delta")

            if delta is not None:
                ll = np.exp(1j * delta[0:size])
            else:
                ll = np.exp(1j * 2 * np.pi * np.random.rand(size))

            s = 1 / size * scipy.linalg.dft(size) @ ll

            S_mat = np.zeros((size, size), dtype=np.complex128)
            for jj in range(0, size):
                S_mat[jj, :] = np.concatenate(
                    (s[(size - jj) : size], s[0 : size - jj])
                )

        case _:
            raise ValueError(
                f"Specified scattering matrix type is invalid. Please choose"
                f" one from {valid_S_mat_types}"
            )

    # Introduce incoherent scattering loss
    scat_loss = S_mat_params.get("scat_loss", 0.0)
    if not np.isclose(scat_loss, 0.0):
        S11 = (np.sqrt(1 - scat_loss**2)) * S_mat
        S12 = np.zeros(shape=(size, size), dtype=np.complex128)
        S21 = np.zeros(shape=(size, size), dtype=np.complex128)
        S22 = scat_loss * np.identity(size, dtype=np.complex128)
        S_mat_top_row = np.concatenate((S11, S12), axis=1)
        S_mat_bot_row = np.concatenate((S21, S22), axis=1)
        S_mat = np.concatenate((S_mat_top_row, S_mat_bot_row), axis=0)

    return S_mat


# -----------------------------------------------------------------------------
# Utility functions for generating networks
# -----------------------------------------------------------------------------


def _remove_duplicates(
    node_list, link_list
) -> tuple[list[Node], list[Link], list[int]]:
    """
    Removes duplicate connections between the same nodes and
    duplicate node ids from the corresponding collections
    """
    new_nodes: list[Node] = []
    new_links: list[Link] = []
    ids: list[int] = []
    pairs: list[tuple[int, int]] = []

    for link in link_list:
        if ((link.node1, link.node2) not in pairs) and (
            (link.node2, link.node1) not in pairs
        ):
            new_links.append(link)
            pairs.append((link.node1, link.node2))

    for node in node_list:
        if node.number not in ids:
            new_nodes.append(node)
            ids.append(node.number)

    return new_nodes, new_links, ids


def _count_nodes(node_list: list[Node]) -> tuple[int, int, int]:
    """
    Counts and returns the node count class attributes
    Also resets node index tracking lists

    Returns tuple (i,e,t)
    ----------------------
    i : int
        Number of internal network nodes.
    e : int
        Number of external network nodes.
    t : int
        Total number of network nodes = i + e.

    """
    # count nodes
    num_internal_nodes = 0
    num_exit_nodes = 0
    num_total_nodes = len(node_list)
    node_indices = []
    nodenumber_indices = {}

    for index, node in enumerate(node_list):
        if node.node_type == "internal":
            num_internal_nodes += 1
        else:
            num_exit_nodes += 1

        node_indices.append(node.number)
        nodenumber_indices[node.number] = index

    return num_internal_nodes, num_exit_nodes, num_total_nodes
