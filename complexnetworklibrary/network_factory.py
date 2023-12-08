"""Factory module for building networks"""

# setup code logging
import logging
import math
import random
import warnings
from typing import Union

import numpy as np
import scipy
from scipy.spatial import ConvexHull, Voronoi
from complexnetworklibrary.spec import NetworkSpec
from complexnetworklibrary.node import Node
from complexnetworklibrary.link import Link
from complexnetworklibrary.network import Network
import logconfig


# from line_profiler_pycharm import profile


logconfig.setup_logging()
logger = logging.getLogger(__name__)


def generate_network(network_spec: NetworkSpec | None = None) -> Network:
    """Main method for building a network from its spec object"""

    if network_spec is None:
        network_spec = NetworkSpec()

    # Get nodes and links
    match network_spec.network_type:
        case "delaunay":
            return _generate_delaunay_network(network_spec)
        case "voronoi":
            return _generate_voronoi_network(network_spec)
        case "buffon":
            return _generate_buffon_network(network_spec)
        case "linear":
            return _generate_linear_network(network_spec)
        case "archimedian":
            return _generate_archimedian_network(network_spec)
        case "empty":
            return None
        case _:
            raise ValueError("Unknown network type")

    # Set node scattering matrices

    #


def _generate_delaunay_network(network_spec: NetworkSpec):
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
                for 'slab': tuple defining (length,width) of rectangular network
            'wavenumber': k,
            'refractive_index': n,
            'exit_size':
                for 'circular': radius of exit nodes from network center
                for 'slab': exit nodes placed at +/-exit_size/2 randomly within width

            'left_exit_fraction': in range [0,1]. Fraction of exit nodes on lefthand side
                of a slab network. Not needed for circular
    """
    # Unpack network sepc
    node_spec = network_spec.node_spec

    num_internal_nodes = network_spec.num_internal_nodes
    num_exit_nodes = network_spec.num_exit_nodes

    network_shape = network_spec.network_shape
    network_size = network_spec.network_size
    exit_size = network_spec.exit_size

    exit_node_dict = {}
    internal_node_dict = {}
    node_dict = {}

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

        # Generate exit node positions
        theta_out = 2 * np.pi * np.random.random(num_exit_nodes)
        r_out = np.array([exit_size] * num_exit_nodes)
        points_out = np.array(
            [r_out * np.cos(theta_out), r_out * np.sin(theta_out)]
        ).T
        for point in points_out:
            new_exit_node_index = len(exit_node_dict)
            exit_node_dict[str(new_exit_node_index)] = Node(
                new_exit_node_index, point, "exit"
            )

    #     r_internal_edge = np.array([network_size] * num_exit_nodes)

    #     # generate random internal points
    #     theta_int = (
    #         2 * np.pi * np.random.random(num_internal_nodes - num_exit_nodes)
    #     )

    #     # square root gives a more uniform distribution of points
    #     r_int = network_size * np.sqrt(
    #         np.random.random(num_internal_nodes - num_exit_nodes)
    #     )

    #     theta = np.concatenate((t_out, t_int, t_out))
    #     r = np.concatenate((rio, r_int, r_out))
    #     points = np.array([r * np.cos(t), r * np.sin(t)]).T

    # elif network_shape == "slab":
    #     # Check type of network_size
    #     if not isinstance(network_size, tuple) or len(network_size) != 2:
    #         raise ValueError(
    #             "network_size must be a tuple of two floats for a "
    #             "slab delaunay network"
    #         )
    #     # Check type of exit_size
    #     if not isinstance(exit_size, float):
    #         raise ValueError(
    #             "exit_size must be a float for a circular delaunay " "network"
    #         )
    #     # Radius of exit nodes must be bigger than radius of internal nodes
    #     if exit_size <= network_size:
    #         raise ValueError("exit_size must be larger than network_size")

    #     network_length, network_width = network_spec.network_size
    #     lhs_frac = network_spec.lhs_frac
    #     lhs_exits = int(np.floor(num_exit_nodes * lhs_frac))
    #     rhs_exits = num_exit_nodes - lhs_exits

    #     if exit_size <= network_length:
    #         raise ValueError(
    #             "exit_size must be larger than network_size[0] (length)"
    #         )
    #     if (lhs_frac < 0) or (lhs_frac > 1):
    #         raise ValueError("left_exit_fraction must be between 0 and 1")

    #     # generate exit node positions
    #     xoutL = -np.array([exit_size / 2] * lhs_exits)
    #     xoutR = np.array([exit_size / 2] * rhs_exits)
    #     youtL = network_width * (np.random.random(lhs_exits) - 0.5)
    #     youtR = network_width * (np.random.random(rhs_exits) - 0.5)

    #     # generate random internal points
    #     xintL = -np.array([network_length / 2] * lhs_exits)
    #     xintR = np.array([network_length / 2] * rhs_exits)
    #     yintL = youtL
    #     yintR = youtR

    #     xint = network_length * (
    #         np.random.random(num_internal_nodes - num_exit_nodes) - 0.5
    #     )
    #     yint = network_width * (
    #         np.random.random(num_internal_nodes - num_exit_nodes) - 0.5
    #     )

    #     x = np.concatenate((xintL, xintR, xint, xoutL, xoutR))
    #     y = np.concatenate((yintL, yintR, yint, youtL, youtR))
    #     points = np.array([x, y]).T

    # # Do delaunay meshing
    # tri = scipy.spatial.Delaunay(points)
    # node_list = []
    # node_indices = []
    # link_list = []

    # # Loop over triangles adding relevant links and nodes
    # for cc, simplex in enumerate(tri.simplices):
    #     for index in range(0, 3):
    #         cur_node = simplex[index]
    #         next_node = simplex[(index + 1) % 3]
    #         x1 = tri.points[cur_node][0]
    #         y1 = tri.points[cur_node][1]
    #         x2 = tri.points[next_node][0]
    #         y2 = tri.points[next_node][1]
    #         distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    #         typestr = "internal" if cur_node < num_internal_nodes else "exit"

    #         # Add new node and link to list
    #         node_list.append(
    #             Node(cur_node, (x1, y1), typestr, network_spec.node_spec)
    #         )
    #         node_indices.append(cur_node)
    #         link_list.append(
    #             Link(cur_node, next_node, typestr, network_spec.link_spec)
    #         )

    # # remove duplicates nodes and links
    # node_list, link_list, node_indices = _remove_duplicates(
    #     node_list, link_list
    # )
    # self.count_nodes()
    # self.connect_nodes()

    # self.trim_extra_exit_node_connections()
    # self.count_nodes()
    # self.connect_nodes()


def _generate_voronoi_network(network_spec: NetworkSpec):
    pass


def _generate_buffon_network(network_spec: NetworkSpec):
    pass


def _generate_linear_network(network_spec: NetworkSpec):
    pass


def _generate_archimedian_network(network_spec: NetworkSpec):
    pass


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
