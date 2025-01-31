"""Factory module for building networks.

This should be used to generate networks, rather than the network class
itself."""

import numpy as np
import scipy
from scipy.spatial import ConvexHull

from complex_network.components.link import Link
from complex_network.components.node import Node
from complex_network.networks.network import Network
from complex_network.networks.network_spec import NetworkSpec, VALID_NETWORK_TYPES
from complex_network.scattering_matrices import link_matrix, node_matrix

def generate_network(spec: NetworkSpec) -> Network:
    """Main method for building a network."""
    
    match spec.network_type:
        case "delaunay":
            nodes, links = _generate_delaunay_nodes_links(spec)
        case "voronoi":
            nodes, links = _generate_voronoi_nodes_links(spec)
            nodes, links = _relabel_nodes_links(nodes, links)
        case "buffon":
            nodes, links = _generate_buffon_network(spec)
            nodes, links = _relabel_nodes_links(nodes, links)
        case "linear":
            nodes, links = _generate_linear_network(spec)
        case "archimedean":
            nodes, links = _generate_archimedean_network(spec)
        case _:
            raise ValueError(
                f"network_type '{spec.network_type}' is invalid."
                f"Please choose one from {VALID_NETWORK_TYPES}."
            )

    # By this point we have nodes that all have an index, but don't know what
    # links or other nodes they are connected to. The links are also all
    # numbered and know what two nodes they connect. These two methods finish
    # off the numbering.
    _initialise_links(nodes, links, spec)
    _initialise_nodes(nodes, links, spec)
    return Network(nodes, links)


def _initialise_nodes(
    nodes: dict[str, Node], links: dict[str, Link], spec: NetworkSpec
) -> None:
    """Set initial values for nodes in the network."""
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
        # Add "ghost" external channel for external nodes
        if node.node_type == "external":
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
        if node.node_type == "external":
            # This matrix just transfers power onwards
            # There is no physics here
            node.get_S = lambda k0: np.array(
                [[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128
            )
            node.get_S_inv = lambda k0: np.array(
                [[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128
            )
            node.get_dS = lambda k0: np.array(
                [[0.0, 0.0], [0.0, 0.0]], dtype=np.complex128
            )
        elif node.node_type == "internal":
            node.S_mat_params = spec.node_S_mat_params
            node.get_S = node_matrix.get_constant_node_S_closure(
                spec.node_S_mat_type, size, spec.node_S_mat_params
            )
            node.get_S_inv = node_matrix.get_inverse_matrix_closure(node.get_S)
            node.get_dS = node_matrix.get_zero_matrix_closure(size)


def _initialise_links(
    nodes: dict[str, Node], links: dict[str, Link], spec: NetworkSpec
) -> None:
    """Set initial values for links in the network."""
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

        # Set link material properties
        link.material = spec.material
        link.n = spec.material.n
        link.dn = spec.material.dn

        link.get_S = link_matrix.get_propagation_matrix_closure(link)
        link.get_S_inv = link_matrix.get_propagation_matrix_inverse_closure(
            link
        )
        link.get_dS = link_matrix.get_propagation_matrix_derivative_closure(
            link
        )


def _generate_delaunay_nodes_links(spec: NetworkSpec) -> tuple[dict, dict]:
    """Generates a Delaunay type network formed from delaunay triangulation

    Parameters
    ----------
    spec : Dictionary specifying properties of network:
        Keys:
            'internal_nodes': number of internal nodes,
            'external_nodes': number of external nodes,
            'shape': 'circular' or 'slab'
            'network_size':
                for 'circular': radius of network
                for 'slab': tuple defining (length,width) of rectangular
                             network
            'wavenumber': k,
            'refractive_index': n,
            'external_size':
                for 'circular': radius of external nodes from network center
                for 'slab': external nodes placed at +/-external_size/2
                            randomly within width

            'left_external_fraction': in range [0,1]. Fraction of external
                                      nodes on the lefthand side of a slab
                                      network. Not needed for circular."""
    num_internal_nodes = spec.num_internal_nodes
    num_external_nodes = spec.num_external_nodes

    network_shape = spec.network_shape
    network_size = spec.network_size
    external_size = spec.external_size
    external_offset = spec.external_offset

    nodes = {}
    links = {}

    if network_shape == "circular":
        # Check type of network_size
        if not isinstance(network_size, float):
            raise ValueError(
                "network_size must be a float for a circular delaunay network"
            )
        # Check type of network_size
        if not isinstance(external_size, float):
            raise ValueError(
                "external_size must be a float for a circular delaunay network"
            )
        # Radius of external nodes must be bigger than radius of internal nodes
        if external_size <= network_size:
            raise ValueError("external_size must be larger than network_size")

        # Generate random points
        # Points on the outer circles
        theta_external = 2 * np.pi * np.random.random(num_external_nodes)
        points_external = (
            external_size
            * np.array([np.cos(theta_external), np.sin(theta_external)]).T
        )
        points_edge = (
            network_size
            * np.array([np.cos(theta_external), np.sin(theta_external)]).T
        )

        # Points in the interior
        theta_int = (
            2
            * np.pi
            * np.random.random(num_internal_nodes - num_external_nodes)
        )
        r_int = network_size * np.sqrt(
            np.random.random(num_internal_nodes - num_external_nodes)
        )
        points_int = np.array(
            [r_int * np.cos(theta_int), r_int * np.sin(theta_int)]
        ).T

    elif network_shape == "slab":
        # Check type of network_size
        if not isinstance(network_size, tuple) or len(network_size) != 2:
            raise ValueError(
                "network_size must be a tuple of two floats for a slab "
                "delaunay network"
            )
        # Check type of network_size
        if not isinstance(external_offset, float):
            raise ValueError(
                "external_offset must be a float for a circular delaunay network"
            )

        # Unpack variables specific to the slab shaped networks
        network_length, network_height = network_size

        if isinstance(num_external_nodes, int):
            num_left_external_nodes, num_right_external_nodes = (
                num_external_nodes,
                num_external_nodes,
            )
        else:
            num_left_external_nodes, num_right_external_nodes = (
                num_external_nodes
            )

        # Error check if total number of internal nodes is too small to make
        # the network
        if (
            num_internal_nodes
            < num_left_external_nodes + num_right_external_nodes
        ):
            raise ValueError(
                f"Number of internal nodes {num_internal_nodes} "
                f"must be at least the total number of external "
                f"nodes {num_left_external_nodes}+"
                f"{num_right_external_nodes}="
                f"{num_left_external_nodes+num_right_external_nodes}"
            )

        # Generate edge and external points
        left_edge_points = np.column_stack(
            (
                np.zeros(num_left_external_nodes),
                np.random.uniform(0, network_height, num_left_external_nodes),
            )
        )
        left_external_points = np.copy(left_edge_points)
        left_external_points[:, 0] = -external_offset

        right_edge_points = np.column_stack(
            (
                np.full(num_right_external_nodes, network_length),
                np.random.uniform(0, network_height, num_right_external_nodes),
            )
        )
        right_external_points = np.copy(right_edge_points)
        right_external_points[:, 0] = network_length + external_offset

        points_edge = np.vstack((left_edge_points, right_edge_points))
        points_external = np.vstack(
            (left_external_points, right_external_points)
        )

        # Points in the interior
        num_internal = (
            num_internal_nodes
            - num_left_external_nodes
            - num_right_external_nodes
        )

        points_int_x = np.random.uniform(0, network_length, num_internal)
        points_int_y = np.random.uniform(0, network_height, num_internal)
        points_int = np.column_stack((points_int_x, points_int_y))

    # All non-external points
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

    # Finally, add external nodes and link them to the edge nodes
    node_start = len(nodes)
    link_start = len(links)
    for i, point_external in enumerate(points_external):
        node_index = node_start + i
        link_index = link_start + i
        # Note that node i is the i'th edge mode, which lines up with the
        # i'th external ndoe
        node_pair = tuple(sorted((node_index, i)))
        nodes[str(node_index)] = Node(node_index, "external", point_external)
        links[str(link_index)] = Link(link_index, "external", node_pair)

    return nodes, links


def _generate_voronoi_nodes_links(spec: NetworkSpec,
    ) -> tuple[dict, dict]:
    network_shape = spec.network_shape
    
    if network_shape == "circular":
        nodes, links = _generate_voronoi_nodes_links_circular(spec)
    elif network_shape == "slab":
        nodes, links = _generate_voronoi_nodes_links_slab(spec)

    return nodes, links


def _generate_voronoi_nodes_links_circular(spec: NetworkSpec,
) -> tuple[dict, dict]:
    seed_nodes = spec.num_seed_nodes
    num_external_nodes = spec.num_external_nodes
    network_shape = spec.network_shape
    network_size = spec.network_size
    exit_size = spec.external_size

    nodes = {}
    links = {}

    # Check type of network_size
    if not isinstance(network_size, float):
        raise ValueError(
            "network_size must be a float for a circular Voronoi network"
        )
    # Check type of network_size
    if not isinstance(exit_size, float):
        raise ValueError(
            "external_size must be a float for a circular Voronoi network"
        )
    # Radius of external nodes must be bigger than radius of internal nodes
    if exit_size <= network_size:
        raise ValueError("external_size must be larger than network_size")

    # generate random internal points
    t = 2 * np.pi * np.random.random(seed_nodes)
    r = network_size * np.sqrt(
        np.random.random(seed_nodes))  # square root gives a more uniform distribution of points
    points = np.array([r * np.cos(t), r * np.sin(t)]).T

    # do Voronoi meshing
    vor = scipy.spatial.Voronoi(points)
    vor_vertices = vor.vertices
    vor_ridges = vor.ridge_vertices
    vor_ridge_points = vor.ridge_points

    # add nodes
    _internal_nodes = 0
    _exit_nodes = 0
    _link_count = 0
    vertices_outside = []
    for number, vertex in enumerate(vor_vertices):
        # only add points lying within specified network size
        if np.linalg.norm([vertex[0], vertex[1]]) < exit_size:
            nodes[str(number)] = Node(number, "internal", vertex)
            _internal_nodes += 1
        else:
            # find vertices outside exit_size
            vertices_outside.append(number)

    # remove any ridges that lie wholly outside exit_size
    ridge_inds_to_delete = []
    for number, ridge in enumerate(vor_ridges):
        sortridge = np.sort(ridge)  # will mean -1 is always first if it exists, otherwise doesn't matter
        if sortridge[0] == -1:
            if np.linalg.norm(vor_vertices[sortridge[1]]) > exit_size:
                ridge_inds_to_delete = np.append(ridge_inds_to_delete, number)
        elif (np.linalg.norm(vor_vertices[sortridge[0]]) > exit_size) and \
                (np.linalg.norm(vor_vertices[sortridge[1]]) > exit_size):
            ridge_inds_to_delete = np.append(ridge_inds_to_delete, number)

    vor_ridge_points = [vor_ridge_points[num] for num, ridge in enumerate(vor_ridges) if
                        num not in ridge_inds_to_delete]
    vor_ridges = [ridge for num, ridge in enumerate(vor_ridges) if num not in ridge_inds_to_delete]

    # loop over ridges and mark ridges with one vertex outsize network as being infinite
    for number, ridge in enumerate(vor_ridges):
        if ridge[0] in vertices_outside:
            vor_ridges[number][0] = -1
        if ridge[1] in vertices_outside:
            vor_ridges[number][1] = -1

    for number, ridge in enumerate(vor_ridges):
        if -1 in ridge:  # infinite extending exit ridges
            ridge.remove(-1)
            id0 = ridge[0]
            vertex = vor_vertices[id0]
            if np.linalg.norm([vertex[0], vertex[1]]) < exit_size:  # lies within network size
                _exit_nodes += 1
                id1 = len(vor_vertices) + _exit_nodes
                # calculate position of exit node
                perpids = vor_ridge_points[number]
                pos = nodes[str(id0)].position
                pid0_pos = points[perpids[0]]
                pid1_pos = points[perpids[1]]
                mid = 0.5 * (np.array(pid0_pos) + np.array(pid1_pos))
                midx = mid[0]
                midy = mid[1]
                grad = (pos[1] - midy) / (pos[0] - midx)

                sqrtfac = np.sqrt((1 + grad ** 2) * exit_size ** 2 - (-grad * mid[0] + mid[1]) ** 2)
                denom = (1 + grad ** 2)

                # one solution of y - y1 = m (x - x1) and x^2 + y^2 = r^2
                x1 = (grad ** 2 * midx - grad * midy + sqrtfac) / denom
                x2 = (grad ** 2 * midx - grad * midy - sqrtfac) / denom

                y1 = (grad * sqrtfac - grad * midx + midy) / denom
                y2 = (-grad * sqrtfac - grad * midx + midy) / denom

                d1 = np.linalg.norm([x1-pos[0], y1-pos[1]])
                d2 = np.linalg.norm([x2-pos[0], y2-pos[1]])
                if d1 < d2:
                    x = x1
                    y = y1
                else:
                    x = x2
                    y = y2

                nodes[str(id1)] = Node(id1, "external", (x, y),)

                links[str(_link_count)] = Link(_link_count, "external", (id0, id1))
                _link_count = _link_count + 1
        elif any([r in vertices_outside for r in ridge]):  # one of vertices is outside
            pass
        else:  # finite ridge in network
            id0 = ridge[0]
            id1 = ridge[1]
            links[str(_link_count)] = Link(_link_count, "internal", (id0, id1))
            _link_count = _link_count + 1
            
    # # Generate random points in the interior
    # theta_int = 2 * np.pi * np.random.random(num_seed_nodes)
    # r_int = network_size * np.sqrt(np.random.random(num_seed_nodes))
    # points_int = np.array(
    #     [r_int * np.cos(theta_int), r_int * np.sin(theta_int)]
    # ).T

    # vor = scipy.spatial.Voronoi(points_int)
    # vor_vertices = vor.vertices
    # vor_ridge_vertices = vor.ridge_vertices
    # vor_ridge_points = vor.ridge_points

    # # Set up nodes, excluding those lying beyond the extent of the network
    # removed_node_indices = []
    # for i, vertex in enumerate(vor_vertices):
    #     r = scipy.linalg.norm(vertex)
    #     if r > network_size:
    #         removed_node_indices.append(i)
    #     else:
    #         nodes[str(i)] = Node(i, "internal", vertex)
    # remaining_node_indices = [int(i) for i in list(nodes.keys())]

    # # Set up links
    # edge_node_indices = []
    # for i, ridge_vertices in enumerate(vor_ridge_vertices):
    #     # Presence of -1 indicates a ridge that extends to infinity
    #     # These are dealt with later
    #     if -1 in ridge_vertices:
    #         first, second = ridge_vertices
    #         edge_node_index = first if second == -1 else second
    #         edge_node_indices.append(str(edge_node_index))
    #         continue

    #     # Make sure that link isn't to a node that has been discarded.
    #     # Note, however, that if a node was connected to a now removed
    #     # node, that node will become an edge node that will ultimately
    #     # connect to an external node
    #     first, second = ridge_vertices
    #     if (
    #         first in removed_node_indices
    #         and second in removed_node_indices
    #     ):
    #         continue
    #     elif first in removed_node_indices:
    #         edge_node_indices.append(str(second))
    #     elif second in removed_node_indices:
    #         edge_node_indices.append(str(first))
    #     else:
    #         links[str(i)] = Link(i, "internal", tuple(ridge_vertices))

    # # Prune linear chains at the edge of the network
    # nodes, links, new_edge_node_indices = deep_prune_edge_chains(
    #     nodes, links
    # )
    # edge_node_indices += new_edge_node_indices

    # # Generate external nodes at the edge_node_indices return after pruning
    # # get centroid first of all nodes

    # xs = [node.position[0] for _, node in nodes.items()]
    # ys = [node.position[1] for _, node in nodes.items()]
    # cx = sum(xs) / len(xs)
    # cy = sum(ys) / len(ys)

    # remaining_node_indices = list(nodes.keys())
    # remaining_link_indices = list(links.keys())
    # i = int(remaining_node_indices[-1]) + 1
    # j = int(remaining_link_indices[-1]) + 1

    # # If the number of external nodes is less than the number of
    # # edge nodes found so far, pick a random subset of them
    # np.random.shuffle(edge_node_indices)
    # num_so_far = 0
    # for edge_node_index in edge_node_indices:
    #     if edge_node_index not in remaining_node_indices:
    #         continue
    #     new_edge_node = nodes[edge_node_index]
    #     x, y = new_edge_node.position
    #     theta = np.arctan2(y - cy, x - cx)
    #     new_position = external_size * np.array(
    #         [np.cos(theta), np.sin(theta)]
    #     )
    #     nodes[str(i)] = Node(i, "external", new_position)
    #     links[str(j)] = Link(j, "external", (int(edge_node_index), i))
    #     i += 1
    #     j += 1
    #     num_so_far += 1
    #     if num_so_far >= num_external_nodes:
    #         break
    return nodes, links


def _generate_voronoi_nodes_links_slab(spec: NetworkSpec,
    ) -> tuple[dict, dict]:
    seed_nodes = spec.num_seed_nodes
    exit_nodes = spec.num_external_nodes
    network_size = spec.network_size
    # Unpack variables specific to the slab shaped networks
    network_length, network_width = network_size

    exit_size = network_length + spec.external_offset

    nodes = {}
    links = {}
    
    # Check type of network_size
    if not isinstance(network_size, tuple) or len(network_size) != 2:
        raise ValueError(
            "network_size must be a tuple of two floats for a slab "
            "Voronoi network"
        )
    # Check type of network_size
    if not isinstance(exit_size, float):
        raise ValueError(
            "external_size must be a float for a circular delaunay network"
        )
    
    # Check if irrelevant external size parameter has been used
    if spec.external_size is not None:
        raise ValueError(
            "External size is not valid for slab networks. Use external offset only instead."
            )


    if isinstance(exit_nodes, int):
        num_left_external_nodes, num_right_external_nodes = (
            exit_nodes,
            exit_nodes,
        )
    else:
        num_left_external_nodes, num_right_external_nodes = (
            exit_nodes
        )

    lhs_exits = num_left_external_nodes
    rhs_exits = num_right_external_nodes

    correct_exits = False
    # switch to ensure correct number of exit nodes get generated
    while not correct_exits:
        # generate exit seed node positions
        xoutL = -np.array([exit_size / 2] * (lhs_exits - 1))
        xoutR = np.array([exit_size / 2] * (rhs_exits - 1))
        youtL = network_width * (np.random.random(lhs_exits - 1) - 0.5)
        youtR = network_width * (np.random.random(rhs_exits - 1) - 0.5)
        xoutinf = exit_size * np.array([-1 / 2, -1 / 2, 1 / 2, 1 / 2])
        youtinf = network_width * np.array([-1 / 2, 1 / 2, -1 / 2, 1 / 2])

        # generate random internal points
        xs = network_length * (np.random.random(seed_nodes) - 0.5)
        ys = network_width * (np.random.random(seed_nodes) - 0.5)
        x = np.concatenate((xs, xoutL, xoutR, xoutinf))
        y = np.concatenate((ys, youtL, youtR, youtinf))
        points = np.array([x, y]).T
        if exit_size <= network_length:
            raise ValueError('exit_size must be larger than network_size[0]')

        # do Voronoi meshing
        vor = scipy.spatial.Voronoi(points)
        vor_vertices = vor.vertices
        vor_ridges = vor.ridge_vertices

        from scipy.spatial import voronoi_plot_2d
        import matplotlib.pyplot as plt
        voronoi_plot_2d(vor)
        plt.show()

        # add nodes
        _internal_nodes = 0
        _exit_nodes = 0
        _link_count = 0
        for number, vertex in enumerate(vor_vertices):
            nodes[(str(number))] = Node(number, "internal", vertex)
            _internal_nodes += 1

        for number, ridge in enumerate(vor_ridges):
            if -1 in ridge:  # infinite extending exit ridges
                # check to see if it is desired output nodes
                ridge.remove(-1)
                id0 = ridge[0]
                
                vertex = vor_vertices[id0]
                if np.abs(vertex[1]) < network_width / 2:  # lies within network size
                    _exit_nodes += 1
                    id1 = len(vor_vertices) + _exit_nodes + 1

                    # calculate position of exit node
                    x = np.sign(vertex[0]) * exit_size / 2
                    y = vertex[1]
                    nodes[(str(id1))] = Node(id1, "external", (x,y))
                    links[str(_link_count)] = Link(_link_count, "external", (id0, id1))
                    _link_count = _link_count + 1
                pass
            else:  # finite ridge in network
                id0 = ridge[0]
                id1 = ridge[1]
                links[str(_link_count)] = Link(_link_count, "internal", (id0, id1))
                _link_count = _link_count + 1

        # self.count_nodes()

        # now trim everything outside vertical width of network
        # look for intersections of ridges with upper/lower part of boundary rectangle
        intersectionsU = {}
        intersectionsL = {}
        edge_node_ids_upper = []
        edge_node_ids_lower = []
        xb, yb = None, None

        current_keys = [key for key in links.keys()]
        for link_key in current_keys:
            connection1 = links[link_key]
            A = nodes[str(connection1.node_indices[0])].position
            B = nodes[str(connection1.node_indices[1])].position
            Ax, Ay = A
            xb, yb = (network_length / 2, network_width / 2)

            # upper boundary
            C = (-xb, yb)
            D = (xb, yb)

            # lower boundary
            E = (-xb, -yb)
            F = (xb, -yb)

            lineridge = [A, B]
            lineupper = [C, D]
            linelower = [E, F]
            int_ptU = _intersection(lineupper, lineridge)
            int_ptL = _intersection(linelower, lineridge)

            if (int_ptU is not None) and \
                    (int_ptL is not None):  # intersect with upper and lower boundary
                # upper node

                intersect_node_idU = max([int(key) for key in nodes.keys()]) + 1 # generate unique id
                edge_node_ids_upper.append(intersect_node_idU)
                nodes[(str(intersect_node_idU))] = Node(intersect_node_idU, "internal", int_ptU)
                _internal_nodes += 1

                # lower node
                intersect_node_idL = max([int(key) for key in nodes.keys()]) + 1
                edge_node_ids_lower.append(intersect_node_idL)
                nodes[(str(intersect_node_idL))] = Node(intersect_node_idL, "internal", int_ptL)
                _internal_nodes += 1

                # connection within network
                links[str(_link_count)] = Link(_link_count, "internal", (intersect_node_idU, intersect_node_idL))
                _link_count = _link_count + 1

                intersectionsU[intersect_node_idU] = {'ridge': link_key,
                                                        'position': int_ptU,
                                                        'node1': intersect_node_idU,
                                                        'node2': intersect_node_idL,
                                                        }

                intersectionsL[intersect_node_idL] = {'ridge': link_key,
                                                        'position': int_ptL,
                                                        'node1': intersect_node_idL,
                                                        'node2': intersect_node_idU,
                                                        }
            elif int_ptU is not None:  # intersect with upper boundary
                # get id for node within bounding rectangle
                if (abs(Ax) <= xb) and (abs(Ay) <= yb):
                    initnode = connection1.node_indices[0]
                else:
                    initnode = connection1.node_indices[1]

                intersect_node_id = max([int(key) for key in nodes.keys()]) + 1
                edge_node_ids_upper.append(intersect_node_id)
                nodes[(str(intersect_node_id))] = Node(intersect_node_id, "internal", int_ptU)
                _internal_nodes += 1

                links[str(_link_count)] = Link(_link_count, "internal", (intersect_node_id, initnode))
                _link_count = _link_count + 1

                intersectionsU[intersect_node_id] = {'ridge': link_key,
                                                        'position': int_ptU,
                                                        'node1': intersect_node_id,
                                                        'node2': initnode,
                                                        }
            elif int_ptL is not None:  # intersect with lower boundary
                # get id for node within bounding rectangle
                if (abs(Ax) <= xb) and (abs(Ay) <= yb):
                    initnode = connection1.node_indices[0]
                else:
                    initnode = connection1.node_indices[1]

                intersect_node_id = max([int(key) for key in nodes.keys()]) + 1
                edge_node_ids_lower.append(intersect_node_id)
                nodes[(str(intersect_node_id))] = Node(intersect_node_id, "internal", int_ptL)
                _internal_nodes += 1

                links[str(_link_count)] = Link(_link_count, "internal", (intersect_node_id, initnode))
                _link_count = _link_count + 1

                intersectionsL[intersect_node_id] = {'ridge': link_key,
                                                        'position': int_ptL,
                                                        'node1': intersect_node_id,
                                                        'node2': initnode,
                                                        }

        # self.count_nodes()

        # remove all exterior nodes (will automatically remove associated connections)
        nodes_to_remove = []
        links_to_remove = []
        for node in nodes.values():
            Ax, Ay = node.position
            # if (abs(Ax) > xb) or (abs(Ay) > yb):
            if  (abs(Ay) > yb):
                nodes_to_remove.append(node.index)

                # find associated connected links
                for key, link in links.items():
                    if node.index in link.node_indices:
                        links_to_remove.append(key)

        for nid in nodes_to_remove:
            nodes.pop(str(nid))

        links_to_remove_unique = list(set(links_to_remove))
        for lid in links_to_remove_unique:
            links.pop(lid)

        # get ids of nodes on upper boundary
        uppernode_ids = [interx['node1'] for interx in intersectionsU.values()]
        lowernode_ids = [interx['node1'] for interx in intersectionsL.values()]
        uppernode_xpos = np.array([intersectionsU[nid]['position'][0] for nid in uppernode_ids])
        lowernode_xpos = np.array([intersectionsL[nid]['position'][0] for nid in lowernode_ids])
        sort_indexu = np.argsort(uppernode_xpos)
        sort_indexl = np.argsort(lowernode_xpos)
        sorted_ids_upper = [uppernode_ids[ii] for ii in sort_indexu]
        sorted_ids_lower = [lowernode_ids[ii] for ii in sort_indexl]

        # connect boundary nodes
        for jj in range(0, len(sorted_ids_upper) - 1):
            id1 = sorted_ids_upper[jj]
            id2 = sorted_ids_upper[jj + 1]
            links[str(_link_count)] = Link(_link_count, "internal", (id1, id2))
            _link_count = _link_count + 1

        for jj in range(0, len(sorted_ids_lower) - 1):
            id1 = sorted_ids_lower[jj]
            id2 = sorted_ids_lower[jj + 1]
            links[str(_link_count)] = Link(_link_count, "internal", (id1, id2))
            _link_count = _link_count + 1


        # check number of exit nodes
        exit_nodesids = [node.index for node in nodes.values() if node.node_type == 'external'] 
        nodes_l = sum([1 if nodes[str(nodeid)].position[0] < 0 else 0 for nodeid in exit_nodesids])
        nodes_r = sum([1 if nodes[str(nodeid)].position[0] > 0 else 0 for nodeid in exit_nodesids])
        nodes_t = len(exit_nodesids)

        print([nodes_l, nodes_r, nodes_t, num_left_external_nodes, num_right_external_nodes, exit_nodes])

        # if (nodes_l == lhs_exits) and (nodes_r == rhs_exits) :
        correct_exits = True
        # else: # unsuitable network so we reinitialise and try again
        #     UserWarning("Incorrect number of exit nodes generated - retrying network generation")
        #     nodes = {}
        #     links = {}

        #     _internal_nodes = 0
        #     _exit_nodes = 0

    # points_int_x = np.random.uniform(0, network_length, num_seed_nodes)
    # points_int_y = np.random.uniform(0, network_height, num_seed_nodes)
    # points_int = np.column_stack((points_int_x, points_int_y))

    # vor = scipy.spatial.Voronoi(points_int)
    # vor_vertices = vor.vertices
    # vor_ridge_vertices = vor.ridge_vertices
    # vor_ridge_points = vor.ridge_points

    # # Set up nodes, excluding those lying beyond the extent of the network
    # removed_node_indices = []
    # for i, vertex in enumerate(vor_vertices):
    #     out_left = vertex[0] < 0.0
    #     out_right = vertex[0] > network_length
    #     out_up = vertex[1] > network_height
    #     out_down = vertex[1] < 0.0
    #     out = out_left or out_right or out_up or out_down
    #     if out:
    #         removed_node_indices.append(i)
    #     else:
    #         nodes[str(i)] = Node(i, "internal", vertex)
    # remaining_node_indices = [int(i) for i in list(nodes.keys())]

    # # Set up links
    # edge_node_indices = []
    # for i, ridge_vertices in enumerate(vor_ridge_vertices):
    #     # Presence of -1 indicates a ridge that extends to infinity
    #     if -1 in ridge_vertices:
    #         continue

    #     # Make sure that link isn't to a node that has been discarded.
    #     # Note, however, that if a node was connected to a now removed
    #     # node, that node will become an edge node that will ultimately
    #     # connect to an external node
    #     first, second = ridge_vertices
    #     if (
    #         first not in removed_node_indices
    #         and second not in removed_node_indices
    #     ):
    #         links[str(i)] = Link(i, "internal", tuple(ridge_vertices))

    return nodes, links


@staticmethod
def _intersection(line1, line2):
    """
    Find the intersection of two line segments defined by their endpoints.

    Parameters
    ----------
        line1 [(x1,y1),(x2,y2)]: A list containing two (x, y) coordinate tuples representing
            the endpoints of the first line segment.
        line2 [(x3,y3),(x4,y4)]: A list containing two (x, y) coordinate tuples representing
            the endpoints of the second line segment.

    Returns
    -------
        tuple: A tuple containing the (x, y) coordinates of the intersection point,
            or None if the lines do not intersect.
    """
    # Unpack the coordinates of the line segments
    p1, p2 = line1
    p3, p4 = line2

    # Convert to numpy arrays
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)

    # Calculate the denominator of the line intersection formula
    den = np.linalg.det(np.array([p2 - p1, p4 - p3]))

    # Check if the denominator is 0 (i.e. lines are parallel)
    if den == 0:
        return None
    else:
        # Calculate the numerator of the line intersection formula
        num1 = np.linalg.det(np.array([p3 - p1, p4 - p3]))
        num2 = np.linalg.det(np.array([p3 - p1, p2 - p1]))
        # Calculate the intersection point parameter (t)
        t1 = num1 / den
        t2 = num2 / den

        # Check if the intersection point is within both line segments
        if 0 <= t1 <= 1 and 0 <= t2 <= 1:
            # Calculate the intersection point and return as a tuple
            return tuple(p1 + t1 * (p2 - p1))
        else:
            return None


def _generate_buffon_network(network_spec: NetworkSpec):
    """
    Generates a Buffon type network formed from intersecting line segments

    Parameters
    ----------
    spec : Dictionary specifying properties of network:
        Keys:
            'lines': 30,
            'shape': 'circular' or 'slab'
            'network_size':
                for 'circular': radius of network
                for 'slab': tuple defining (length,width) of rectangular network
            'wavenumber': k,
            'refractive_index': n,
            'fully_connected': True,

    """
    total_lines = spec["lines"]
    external_link_number = 2 * total_lines
    external_link_number = (
        external_link_number + external_link_number % 2
    )  # round up to nearest multilpe of 2

    if spec["shape"] == "circular":
        network_size = spec["network_size"]
    elif spec["shape"] == "slab":
        network_length = spec["network_size"][0]
        network_width = spec["network_size"][1]
    else:
        raise ValueError(
            '"shape" in network spec should be either "circular" or "slab"'
        )

    self.external_nodes = 0  # external_link_number
    iternum = 0
    fibres = []

    while self.external_nodes != external_link_number:
        iternum += 1

        # determine missing number of external nodes
        missing_nodes = external_link_number - self.external_nodes
        number_of_lines = int(missing_nodes / 2)
        available_node_ids = [
            i for i in range(0, total_lines) if i not in self.node_indices
        ]
        # generate random pairs of points

        # fibres = self.links
        intersections = {}
        for nn in range(0, number_of_lines):
            if spec["shape"] == "circular":
                t = 2 * math.pi * np.random.random(2)
                xn = network_size * np.cos(t)
                yn = network_size * np.sin(t)
            elif spec["shape"] == "slab":
                xn = np.array([-network_length / 2, network_length / 2])
                yn = network_width * (np.random.random(2) - 0.5)
            points = np.array([xn, yn]).T

            nodeid = available_node_ids[nn]

            self.add_node(nodeid, (points[0, 0], points[0, 1]), "external")
            self.add_node(
                total_lines + nodeid, (points[1, 0], points[1, 1]), "external"
            )

            distance = self.calculate_distance(
                (points[0, 0], points[0, 1]), (points[1, 0], points[1, 1])
            )

            fibres.append(
                LINK(nodeid, total_lines + nodeid, distance, self.k, self.n)
            )

        # construct array of all intersections and track which fibres these points correspond to
        for ii in range(0, len(fibres)):
            connection1 = fibres[ii]
            A = self.get_node(connection1.node1).position
            B = self.get_node(connection1.node2).position
            for jj in range(ii + 1, len(fibres)):
                connection2 = fibres[jj]
                C = self.get_node(connection2.node1).position
                D = self.get_node(connection2.node2).position

                line1 = [A, B]
                line2 = [C, D]
                int_pt = self.intersection(line1, line2)
                if int_pt is not None:  # lines intersect
                    intersect_node_id = len(self.nodes)
                    self.add_node(intersect_node_id, int_pt)
                    intersections[intersect_node_id] = {
                        "line1": ii,
                        "line2": jj,
                        "position": int_pt,
                    }

        # construct connections
        for ii in range(0, len(fibres)):
            endpos = self.get_node(fibres[ii].node1).position
            # find nodes which lie along this fibre
            nodes = [
                inter
                for inter in intersections
                if (
                    (intersections[inter]["line1"] == ii)
                    or (intersections[inter]["line2"] == ii)
                )
            ]
            # order them in ascending distance from one end
            distances = [
                self.calculate_distance(endpos, intersections[jj]["position"])
                for jj in nodes
            ]
            orderednodes = [x for _, x in sorted(zip(distances, nodes))]

            orderednodes.insert(0, fibres[ii].node1)
            orderednodes.append(fibres[ii].node2)
            # form connections
            for jj in range(0, len(orderednodes) - 1):
                distance = self.calculate_distance(
                    self.get_node(orderednodes[jj]).position,
                    self.get_node(orderednodes[jj + 1]).position,
                )
                self.add_connection(
                    orderednodes[jj],
                    orderednodes[jj + 1],
                    distance,
                    self.k,
                    self.n,
                )

        # loop through the connections and reset those that are connected to external nodes
        for link in self.links:
            if (self.get_node(link.node1).node_type == "external") or (
                self.get_node(link.node2).node_type == "external"
            ):
                link.link_type = "external"
                link.reset_link(link.distance, link.k, link.n)

        self.connect_nodes()
        self.count_nodes()

        # check to see if network is fully connected network request and if generated matrix is thus.
        if spec["fully_connected"] is True:
            (nc, components) = self.connected_component_nodes()
            if nc == 1:
                return
            # find connected component with most components
            # print("Trimming {} components...".format(nc))
            comp_size = [len(comp) for comp in components]
            largest = np.argmax(comp_size)

            # construct list of nodes to remove
            nodes_to_remove = [
                comp
                for index, comp in enumerate(components)
                if index != largest
            ]

            # also remove node indices corersponding to intersections as these will be regenerated
            intersection_nodes = [
                node.number
                for node in self.nodes
                if node.number >= external_link_number
            ]
            nodes_to_remove.append(intersection_nodes)
            nodes_to_remove_flat = [
                item for sublist in nodes_to_remove for item in sublist
            ]

            # cycle through links and get indices of those connected to unwanted nodes
            fibres_to_remove_flat = []
            for index, link in enumerate(fibres):
                for nid in nodes_to_remove_flat:
                    if link.node1 == nid or link.node2 == nid:
                        if index not in fibres_to_remove_flat:
                            fibres_to_remove_flat.append(index)

            # remove links and nodes
            # NB we maintain fibres list incase we have to do another iteration. This is cleaner and faster
            fibres = [
                link
                for index, link in enumerate(fibres)
                if index not in fibres_to_remove_flat
            ]
            newnodes = [
                node
                for node in self.nodes
                if node.number not in nodes_to_remove_flat
            ]
            ids = [
                idn
                for idn in self.node_indices
                if idn not in nodes_to_remove_flat
            ]

            self.links = []  # these will be regenerated
            self.nodes = newnodes
            self.node_indices = ids
            self.count_nodes()


def _generate_linear_network(network_spec: NetworkSpec):
    """Generates a linear network with all nodes on a straight line

    Parameters
    ----------
    spec : Dictionary specifying properties of network:
        Keys:
            internal_nodes: number of internal nodes of network
            network_size: all internal nodes will be distributed randomly within range [-1/2,1/2]*network_size
            external_size: two external nodes placed at +/-external_size/2"""
    node_number = spec["internal_nodes"]
    network_size = spec["network_size"]
    external_size = spec["external_size"]

    if external_size < network_size:
        raise ValueError("external_size must be larger than network_size.")

    # generate random positions
    x = network_size * (np.random.random(node_number) - 0.5)
    xs = sorted(x)

    # add external nodes
    xs = np.insert(xs, 0, -external_size / 2)
    xs = np.append(xs, external_size / 2)

    for index in range(0, len(xs)):
        if index == 0 or index == len(xs) - 1:
            self.add_node(index, (xs[index], 0), "external")
        else:
            self.add_node(index, (xs[index], 0), "internal")

    for index in range(0, len(xs) - 1):
        if index == 0 or index == len(xs) - 2:
            self.add_connection(
                index,
                index + 1,
                xs[index + 1] - xs[index],
                self.k,
                self.n,
                "external",
            )
        else:
            self.add_connection(
                index, index + 1, xs[index + 1] - xs[index], self.k, self.n
            )

    self.count_nodes()


def _generate_archimedean_network(network_spec: NetworkSpec):
    """
    Generates a network formed from Euclidean uniform/Archimedean/Catalan tilings
        see https://en.wikipedia.org/wiki/List_of_Euclidean_uniform_tilings

    Parameters
    ----------
    spec : Dictionary specifying properties of network:
        Keys:
            internal_nodes: number of internal nodes of network
            network_size: all internal nodes will be distributed randomly within range [-1/2,1/2]*network_size
            external_size: two external nodes placed at +/-external_size/2

            num_layers':3,
            'scale': network_rad,
            'type': 'square',
            'external_nodes': 5} # square,triangular, honeycomb

    Parameters
    ----------
    spec : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    external_link_number = spec["external_nodes"]

    network_size, external_size = self.generate_tiling(spec)

    points = [
        np.array(node.position)
        for node in self.nodes
        if node.node_type == "internal"
    ]
    numbers = [
        node.number for node in self.nodes if node.node_type == "internal"
    ]
    node_number = max(numbers) + 1

    # find network nodes on convex hull
    hullids = ConvexHull(points)
    # add some external links
    for ii in range(0, external_link_number):
        theta = 2 * math.pi * np.random.random(1)
        externalx = external_size * np.cos(theta)[0]
        externaly = external_size * np.sin(theta)[0]
        self.add_node(node_number + ii, (externalx, externaly), "external")

        # find appropriate connection to closest point on convex hull
        min_distance = 2 * external_size

        for number in hullids.vertices:
            node = self.get_node(numbers[number])

            newdistance = np.sqrt(
                (externalx - node.position[0]) ** 2
                + (externaly - node.position[1]) ** 2
            )

            if newdistance < min_distance:
                min_distance = newdistance
                nearest_id = node.number

        self.add_connection(
            node_number + ii,
            nearest_id,
            min_distance,
            self.k,
            self.n,
            "external",
        )


# -----------------------------------------------------------------------------
# Utility functions for generating networks
# -----------------------------------------------------------------------------


def _relabel_nodes_links(
    nodes: dict[str, Node], links: dict[str, Link]
) -> tuple[dict[str, Node], dict[str, Link]]:
    """Given node and link dictionaries where the keys are not consecutive
    integers (because, for example, some nodes and links were deleted along
    the way when they were generated), relabel them so that the keys are
    consecutive integers. Cleans up indexing."""
    # Create key maps
    node_keys = list(nodes.keys())
    node_key_map = {value: str(index) for index, value in enumerate(node_keys)}
    link_keys = list(links.keys())
    link_key_map = {value: str(index) for index, value in enumerate(link_keys)}

    new_nodes = {}
    new_links = {}

    # Relabel nodes
    for old_node_index, node in nodes.items():
        new_node_index = node_key_map[old_node_index]
        new_nodes[new_node_index] = node

    # Relabel links
    for old_link_index, link in links.items():
        # We must change the node indices property of the link
        old_node_index_one, old_node_index_two = link.node_indices
        new_node_index_one = node_key_map[str(old_node_index_one)]
        new_node_index_two = node_key_map[str(old_node_index_two)]
        link.node_indices = (int(new_node_index_one), int(new_node_index_two))

        # Get the new link index
        new_link_index = link_key_map[old_link_index]
        new_links[new_link_index] = link

    return new_nodes, new_links


def deep_prune_edge_chains(
    nodes: dict[str, Node], links: dict[str, Link]
) -> tuple[dict[str, Node], dict[str, Link], list[int | str] | None]:
    """Execute prune_edge_chains repeatedly until no more linear chains remain
    in the network"""

    old_edges = []
    old_nodes = nodes
    old_links = links

    iteration_limit = 100
    for _ in range(iteration_limit):
        new_nodes, new_links, new_edges = prune_edge_chains(
            old_nodes, old_links
        )

        # None indicates that the new and old lists are identical
        # i.e. no pruning occured
        if new_edges is None:
            # new_edges may contain indices of nodes that have already been
            # pruned. Here we filter it to make sure these are removed.
            final_node_indices = list(old_nodes.keys())
            final_edges = [i for i in old_edges if i in final_node_indices]

            return old_nodes, old_links, final_edges

        # Arriving here indicates that some pruning occured
        old_nodes = new_nodes
        old_links = new_links
        old_edges += new_edges

    raise (
        RuntimeError(
            f"Pruning incomplete after {iteration_limit} iterations. "
            "Network likely faulty. Please inspect manually."
        )
    )


def prune_edge_chains(
    nodes: dict[str, Node], links: dict[str, Link]
) -> tuple[dict[str, Node], dict[str, Link], list[str | int] | None]:
    """Removes linear chains of links and nodes that emanate linearly
    from the edge of a network.

    The returned list contains indices of the nodes at the bases of the pruned
    chains. These might be used as edge nodes for the creation of further external
    nodes.
    """

    # Work out how many connections each node has.
    num_connections = {key: 0 for key in nodes.keys()}

    for link in links.values():
        node_one, node_two = link.node_indices
        num_connections[str(node_one)] += 1
        num_connections[str(node_two)] += 1

    # Get indices of nodes that need to be removed
    remove_list = [key for key, value in num_connections.items() if value == 1]

    # external early with fail flag if no nodes need to be pruned
    if len(remove_list) == 0:
        return nodes, links, None

    # Filter nodes and indices by removing ones from the remove list
    # For links, record adjoining nodes as these will become new edge nodes
    # that will receive external nodes.
    new_nodes = {}
    new_links = {}
    new_edge_node_indices = []

    for node_index, node in nodes.items():
        if node_index not in remove_list:
            new_nodes[node_index] = node

    for link_index, link in links.items():
        node_index_one = str(link.node_indices[0])
        node_index_two = str(link.node_indices[1])

        if node_index_one in remove_list:
            new_edge_node_indices.append(node_index_two)
            continue
        if node_index_two in remove_list:
            new_edge_node_indices.append(node_index_one)
            continue
        new_links[link_index] = link

    return new_nodes, new_links, new_edge_node_indices


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
