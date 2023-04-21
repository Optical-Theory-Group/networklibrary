# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:53:54 2022

@author: Matthew Foreman

Network generator class
"""

import numpy as np
import math

import random
import warnings
from scipy.spatial import Delaunay, Voronoi, ConvexHull
from line_profiler_pycharm import profile


from .node import NODE
from .link import LINK

# setup code logging
import logging
import logconfig

logconfig.setup_logging()
logger = logging.getLogger(__name__)

class NetworkGenerator:
    def __init__(self, network_type, network_spec, seed_number=0):
        # seed random number generator
        np.random.seed(seed_number)

        # initialise
        self.nodes = []
        self.links = []
        self.node_indices = []
        self.nodenumber_indices = {}

        self.internal_nodes = 0
        self.exit_nodes = 0
        self.total_nodes = 0
        self.network_spec = network_spec

        self.k = network_spec['wavenumber'] if 'wavenumber' in network_spec.keys() else 1
        self.n = network_spec['refractive_index'] if 'refractive_index' in network_spec.keys() else 1

        if network_type == 'delaunay':
            self.generate_delaunay(network_spec)
        elif network_type == 'voronoi':
            self.generate_voronoi(network_spec)
        elif network_type == 'buffon':
            self.generate_buffon(network_spec)
        elif network_type == 'linear':
            self.generate_linear(network_spec)
        elif network_type == 'archimedean':
            self.generate_archimedean(network_spec)
        else:
            # we can add extra code here for generating different networks at a later point
            raise (ValueError, 'Unknown network type')

        # we initialise the connected node list for each node in network
        self.count_nodes()
        self.connect_nodes()

    def generate_delaunay(self, spec):
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

        self.internal_nodes = spec['internal_nodes']
        self.exit_nodes = spec['exit_nodes']
        self.total_nodes = self.internal_nodes + self.exit_nodes

        points = None
        if spec['shape'] == 'circular':
            network_size = spec['network_size']
            exit_size = spec['exit_size']
            if exit_size <= network_size:
                raise ValueError('exit_size must be larger than network_size')

            # generate exit node positions
            tout = 2 * math.pi * np.random.random(self.exit_nodes)
            rout = np.array([exit_size] * self.exit_nodes)
            rio = np.array([network_size] * self.exit_nodes)

            # generate random internal points
            tint = 2 * math.pi * np.random.random(self.internal_nodes - self.exit_nodes)
            rint = network_size * np.sqrt(np.random.random(
                self.internal_nodes - self.exit_nodes))  # square root gives a more uniform distribution of points

            t = np.concatenate((tout, tint, tout))
            r = np.concatenate((rio, rint, rout))

            points = np.array([r * np.cos(t), r * np.sin(t)]).T
        if spec['shape'] == 'slab':
            network_length = spec['network_size'][0]
            network_width = spec['network_size'][1]
            exit_size = spec['exit_size']
            lhs_frac = spec['left_exit_fraction']
            lhs_exits = int(np.floor(self.exit_nodes * lhs_frac))
            rhs_exits = self.exit_nodes - lhs_exits

            if exit_size <= network_length:
                raise ValueError('exit_size must be larger than network_size[0] (length)')
            if (lhs_frac < 0) or (lhs_frac > 1):
                raise ValueError('left_exit_fraction must be between 0 and 1')

            # generate exit node positions
            xoutL = -np.array([exit_size / 2] * lhs_exits)
            xoutR = np.array([exit_size / 2] * rhs_exits)
            youtL = network_width * (np.random.random(lhs_exits) - 0.5)
            youtR = network_width * (np.random.random(rhs_exits) - 0.5)

            # generate random internal points
            xintL = -np.array([network_length / 2] * lhs_exits)
            xintR = np.array([network_length / 2] * rhs_exits)
            yintL = youtL
            yintR = youtR

            xint = network_length * (np.random.random(self.internal_nodes - self.exit_nodes) - 0.5)
            yint = network_width * (np.random.random(self.internal_nodes - self.exit_nodes) - 0.5)

            x = np.concatenate((xintL, xintR, xint, xoutL, xoutR))
            y = np.concatenate((yintL, yintR, yint, youtL, youtR))
            points = np.array([x, y]).T

        # do delaunay meshing
        tri = Delaunay(points)

        # loop over triangles adding relevant connections and nodes to network class
        for cc, simplex in enumerate(tri.simplices):  # For 2-D, the points are oriented counterclockwise.
            for index in range(0, 3):
                cur_node = simplex[index]
                next_node = simplex[(index + 1) % 3]
                x1 = tri.points[cur_node][0]
                y1 = tri.points[cur_node][1]
                x2 = tri.points[next_node][0]
                y2 = tri.points[next_node][1]
                distance = np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

                typestr = 'internal' if simplex[index] < self.internal_nodes else 'exit'
                self.add_node(simplex[index], (x1, y1), typestr)
                self.add_connection(simplex[index], simplex[(index + 1) % 3], distance, self.k, self.n, typestr)

        # remove duplicates nodes and links
        self.remove_duplicates()
        self.count_nodes()
        self.connect_nodes()

        self.trim_extra_exit_node_connections()
        self.count_nodes()
        self.connect_nodes()

    def generate_voronoi(self, spec):
        """
        Generates a voronoi type network formed from the dual of delaunay triangulation

        Parameters
        ----------
        spec : Dictionary specifying properties of network:
            Keys:
                'seed_nodes': number of seed points to generate Voronoi nodes,
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
                'dual': boolean. Specifiy whether dual network is used. NOT YET IMPLEMENTED

        """

        if spec['shape'] == 'circular':
            self.generate_voronoi_circular(spec)
        elif spec['shape'] == 'slab':
            self.generate_voronoi_slab(spec)

        # remove duplicates nodes and links
        self.remove_duplicates()
        self.count_nodes()
        self.connect_nodes()

        self.trim_extra_exit_node_connections()
        self.count_nodes()
        self.connect_nodes()

    def generate_voronoi_circular(self, spec):
        seed_nodes = spec['seed_nodes']
        network_size = spec['network_size']
        exit_size = spec['exit_size']

        # generate random internal points
        t = 2 * math.pi * np.random.random(seed_nodes)
        r = network_size * np.sqrt(
            np.random.random(seed_nodes))  # square root gives a more uniform distribution of points
        points = np.array([r * np.cos(t), r * np.sin(t)]).T
        if exit_size <= network_size:
            raise ValueError('exit_size must be larger than network_size')

        # do Voronoi meshing
        vor = Voronoi(points)
        vor_vertices = vor.vertices
        vor_ridges = vor.ridge_vertices
        vor_ridge_points = vor.ridge_points

        # add nodes
        self.internal_nodes = 0
        self.exit_nodes = 0
        vertices_outside = []
        for number, vertex in enumerate(vor_vertices):
            # only add points lying within specified network size
            if np.linalg.norm([vertex[0], vertex[1]]) < exit_size:
                self.add_node(number, (vertex[0], vertex[1]), 'internal')
                self.internal_nodes += 1
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
                    self.exit_nodes += 1
                    id1 = len(vor_vertices) + self.exit_nodes

                    # calculate position of exit node
                    perpids = vor_ridge_points[number]
                    pos = self.get_node(id0).position
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

                    d1 = self.calculate_distance((x1, y1), pos)
                    d2 = self.calculate_distance((x2, y2), pos)
                    if d1 < d2:
                        x = x1
                        y = y1
                        distance = d1
                    else:
                        x = x2
                        y = y2
                        distance = d2

                    self.add_node(id1, (x, y), 'exit')
                    self.add_connection(id0, id1, distance, self.k, self.n, 'exit')
            elif any([r in vertices_outside for r in ridge]):  # one of vertices is outside
                pass
            else:  # finite ridge in network
                id0 = ridge[0]
                id1 = ridge[1]
                distance = self.calculate_distance(self.get_node(id0).position, self.get_node(id1).position)
                self.add_connection(id0, id1, distance, self.k, self.n, 'internal')
    @staticmethod
    def plot_lines(line1, line2, intersection=None):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], label='Line 1')
        ax.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], label='Line 2')
        if intersection:
            ax.plot(intersection[0], intersection[1], 'ro', label='Intersection')
        ax.legend()
        plt.show()

    def generate_voronoi_slab(self, spec):
        seed_nodes = spec['seed_nodes']
        exit_nodes = spec['exit_nodes']
        exit_size = spec['exit_size']
        network_length = spec['network_size'][0]
        network_width = spec['network_size'][1]
        lhs_frac = spec['left_exit_fraction']
        lhs_exits = int(np.floor(exit_nodes * lhs_frac))
        rhs_exits = exit_nodes - lhs_exits

        correct_exits = False
        # switch to ensure correct number of exit nodes get generated
        while not correct_exits:
            seed_width = network_width
            # generate exit seed node positions
            xoutL = -np.array([exit_size / 2] * (lhs_exits - 1))
            xoutR = np.array([exit_size / 2] * (rhs_exits - 1))
            youtL = network_width * (np.random.random(lhs_exits - 1) - 0.5)
            youtR = network_width * (np.random.random(rhs_exits - 1) - 0.5)
            xoutinf = exit_size * np.array([-1 / 2, -1 / 2, 1 / 2, 1 / 2])
            youtinf = network_width * np.array([-1 / 2, 1 / 2, -1 / 2, 1 / 2])

            # generate random internal points
            xs = network_length * (np.random.random(seed_nodes) - 0.5)
            ys = seed_width * (np.random.random(seed_nodes) - 0.5)
            x = np.concatenate((xs, xoutL, xoutR, xoutinf))
            y = np.concatenate((ys, youtL, youtR, youtinf))
            points = np.array([x, y]).T
            if exit_size <= network_length:
                raise ValueError('exit_size must be larger than network_size[0]')

            # do Voronoi meshing
            vor = Voronoi(points)

            # from scipy.spatial import voronoi_plot_2d
            # import matplotlib.pyplot as plt
            # voronoi_plot_2d(vor)
            # plt.show()

            vor_vertices = vor.vertices
            vor_ridges = vor.ridge_vertices
            # vor_ridge_points = vor.ridge_points

            # add nodes
            self.internal_nodes = 0
            self.exit_nodes = 0
            for number, vertex in enumerate(vor_vertices):
                self.add_node(number, (vertex[0], vertex[1]), 'internal')
                self.internal_nodes += 1

            for number, ridge in enumerate(vor_ridges):
                if -1 in ridge:  # infinite extending exit ridges
                    # check to see if it is desired output nodes
                    ridge.remove(-1)
                    id0 = ridge[0]
                    vertex = vor_vertices[id0]
                    if np.abs(vertex[1]) < network_width / 2:  # lies within network size
                        self.exit_nodes += 1
                        id1 = len(vor_vertices) + self.exit_nodes + 1

                        # calculate position of exit node
                        pos = self.get_node(id0).position
                        x = np.sign(vertex[0]) * exit_size / 2
                        y = vertex[1]
                        distance = self.calculate_distance((x, y), pos)
                        self.add_node(id1, (x, y), 'exit')
                        self.add_connection(id0, id1, distance, self.k, self.n, 'exit')
                    pass
                else:  # finite ridge in network
                    id0 = ridge[0]
                    id1 = ridge[1]
                    distance = self.calculate_distance(self.get_node(id0).position, self.get_node(id1).position)
                    self.add_connection(id0, id1, distance, self.k, self.n, 'internal')

            self.count_nodes()
            self.connect_nodes()

            # now trim everything outside vertical width of network
            # look for intersections of ridges with upper/lower part of boundary rectangle
            intersectionsU = {}
            intersectionsL = {}
            edge_node_ids_upper = []
            edge_node_ids_lower = []
            xb, yb = None, None
            for ii in range(0, len(self.links)):
                connection1 = self.links[ii]
                A = self.get_node(connection1.node1).position
                B = self.get_node(connection1.node2).position
                Ax, Ay = A
                # Bx, By = B
                xb, yb = (exit_size / 2, network_width / 2)

                # upper boundary
                C = (-xb, yb)
                D = (xb, yb)

                # lower boundary
                E = (-xb, -yb)
                F = (xb, -yb)

                lineridge = [A, B]
                lineupper = [C, D]
                linelower = [E, F]
                int_ptU = self.intersection(lineupper, lineridge)
                int_ptL = self.intersection(linelower, lineridge)


                if (int_ptU is not None) and \
                        (int_ptL is not None):  # intersect with upper and lower boundary
                    # upper node
                    intersect_node_idU = self.gen_unique_node_id()
                    edge_node_ids_upper.append(intersect_node_idU)
                    self.add_node(intersect_node_idU, int_ptU)

                    # lower node
                    intersect_node_idL = self.gen_unique_node_id()
                    edge_node_ids_lower.append(intersect_node_idL)
                    self.add_node(intersect_node_idL, int_ptL)

                    # connection within network
                    self.add_connection(intersect_node_idU, intersect_node_idL,
                                        self.calculate_distance(int_ptU, int_ptL), self.k, self.n, 'internal')

                    intersectionsU[intersect_node_idU] = {'ridge': ii,
                                                          'position': int_ptU,
                                                          'node1': intersect_node_idU,
                                                          'node2': intersect_node_idL,
                                                          }

                    intersectionsL[intersect_node_idL] = {'ridge': ii,
                                                          'position': int_ptL,
                                                          'node1': intersect_node_idL,
                                                          'node2': intersect_node_idU,
                                                          }
                elif int_ptU is not None:  # intersect with upper boundary
                    # get id for node within bounding rectangle
                    if (abs(Ax) <= xb) and (abs(Ay) <= yb):
                        initnode = connection1.node1
                    else:
                        initnode = connection1.node2

                    intersect_node_id = self.gen_unique_node_id()
                    edge_node_ids_upper.append(intersect_node_id)
                    self.add_node(intersect_node_id, int_ptU)
                    self.add_connection(intersect_node_id, initnode, self.calculate_distance(A, int_ptU), self.k,
                                        self.n, 'internal')
                    intersectionsU[intersect_node_id] = {'ridge': ii,
                                                         'position': int_ptU,
                                                         'node1': intersect_node_id,
                                                         'node2': initnode,
                                                         }
                elif int_ptL is not None:  # intersect with lower boundary
                    # get id for node within bounding rectangle
                    if (abs(Ax) <= xb) and (abs(Ay) <= yb):
                        initnode = connection1.node1
                    else:
                        initnode = connection1.node2

                    intersect_node_id = self.gen_unique_node_id()
                    edge_node_ids_lower.append(intersect_node_id)
                    self.add_node(intersect_node_id, int_ptL)
                    self.add_connection(intersect_node_id, initnode, self.calculate_distance(A, int_ptL), self.k,
                                        self.n, 'internal')
                    intersectionsL[intersect_node_id] = {'ridge': ii,
                                                         'position': int_ptL,
                                                         'node1': intersect_node_id,
                                                         'node2': initnode,
                                                         }

            self.count_nodes()
            self.connect_nodes()

            # remove all exterior nodes (will automatically remove associated connections)
            nodes_to_remove = []
            for node in self.nodes:
                Ax, Ay = node.position
                if (abs(Ax) > xb) or (abs(Ay) > yb):
                    nodes_to_remove.append(node.number)

            for nid in nodes_to_remove:
                self.remove_node(nid)

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
                if id2 == 127:
                    print(id2)
                pos1 = self.get_node(id1).position
                pos2 = self.get_node(id2).position
                self.add_connection(id1, id2, self.calculate_distance(pos1, pos2), self.k, self.n, 'internal')

            for jj in range(0, len(sorted_ids_lower) - 1):
                id1 = sorted_ids_lower[jj]
                id2 = sorted_ids_lower[jj + 1]
                pos1 = self.get_node(id1).position
                pos2 = self.get_node(id2).position
                self.add_connection(id1, id2, self.calculate_distance(pos1, pos2), self.k, self.n, 'internal')

            self.count_nodes()
            self.connect_nodes()

            # # check number of exit nodes
            exit_nodesids = self.get_exit_node_ids()
            nodes_l = sum([1 if self.get_node(nodeid).position[0] < 0 else 0 for nodeid in exit_nodesids])
            nodes_r = sum([1 if self.get_node(nodeid).position[0] > 0 else 0 for nodeid in exit_nodesids])
            nodes_t = len(exit_nodesids)

            if (nodes_l == lhs_exits) and (nodes_r == rhs_exits) and (nodes_t == exit_nodes):
                correct_exits = True
            else:
                print('Regnerating unsuitable network')
                print([nodes_l, nodes_r, nodes_t])

                # from scipy.spatial import voronoi_plot_2d
                # import matplotlib.pyplot as plt
                # fig = voronoi_plot_2d(vor)
                # plt.show()

                # re-initialise
                self.nodes = []
                self.links = []
                self.node_indices = []
                self.nodenumber_indices = {}

                self.internal_nodes = 0
                self.exit_nodes = 0
                self.total_nodes = 0

    def generate_buffon(self, spec):
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
        total_lines = spec['lines']
        external_link_number = 2 * total_lines
        external_link_number = external_link_number + external_link_number % 2  # round up to nearest multilpe of 2

        if spec['shape'] == 'circular':
            network_size = spec['network_size']
        elif spec['shape'] == 'slab':
            network_length = spec['network_size'][0]
            network_width = spec['network_size'][1]
        else:
            raise ValueError('"shape" in network spec should be either "circular" or "slab"')

        self.exit_nodes = 0  # external_link_number
        iternum = 0
        fibres = []

        while self.exit_nodes != external_link_number:
            iternum += 1

            # determine missing number of exit nodes
            missing_nodes = external_link_number - self.exit_nodes
            number_of_lines = int(missing_nodes / 2)
            available_node_ids = [i for i in range(0, total_lines) if i not in self.node_indices]
            # generate random pairs of points

            # fibres = self.links
            intersections = {}
            for nn in range(0, number_of_lines):
                if spec['shape'] == 'circular':
                    t = 2 * math.pi * np.random.random(2)
                    xn = network_size * np.cos(t)
                    yn = network_size * np.sin(t)
                elif spec['shape'] == 'slab':
                    xn = np.array([- network_length / 2, network_length / 2])
                    yn = network_width * (np.random.random(2) - 0.5)
                points = np.array([xn, yn]).T

                nodeid = available_node_ids[nn]

                self.add_node(nodeid, (points[0, 0], points[0, 1]), 'exit')
                self.add_node(total_lines + nodeid, (points[1, 0], points[1, 1]), 'exit')

                distance = self.calculate_distance((points[0, 0], points[0, 1]),
                                                   (points[1, 0], points[1, 1]))

                fibres.append(LINK(nodeid, total_lines + nodeid, distance, self.k, self.n))

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
                        intersections[intersect_node_id] = {'line1': ii, 'line2': jj, 'position': int_pt}

            # construct connections
            for ii in range(0, len(fibres)):
                endpos = self.get_node(fibres[ii].node1).position
                # find nodes which lie along this fibre
                nodes = [inter for inter in intersections if
                         ((intersections[inter]['line1'] == ii) or (intersections[inter]['line2'] == ii))]
                # order them in ascending distance from one end
                distances = [self.calculate_distance(endpos, intersections[jj]['position']) for jj in nodes]
                orderednodes = [x for _, x in sorted(zip(distances, nodes))]

                orderednodes.insert(0, fibres[ii].node1)
                orderednodes.append(fibres[ii].node2)
                # form connections
                for jj in range(0, len(orderednodes) - 1):
                    distance = self.calculate_distance(self.get_node(orderednodes[jj]).position,
                                                       self.get_node(orderednodes[jj + 1]).position)
                    self.add_connection(orderednodes[jj], orderednodes[jj + 1], distance, self.k, self.n)

            # loop through the connections and reset those that are connected to exit nodes
            for link in self.links:
                if (self.get_node(link.node1).node_type == 'exit') or (self.get_node(link.node2).node_type == 'exit'):
                    link.link_type = 'exit'
                    link.reset_link(link.distance, link.k, link.n)

            self.connect_nodes()
            self.count_nodes()

            # check to see if network is fully connected network request and if generated matrix is thus.
            if spec['fully_connected'] is True:
                (nc, components) = self.connected_component_nodes()
                if nc == 1:
                    return
                # find connected component with most components
                # print("Trimming {} components...".format(nc))
                comp_size = [len(comp) for comp in components]
                largest = np.argmax(comp_size)

                # construct list of nodes to remove
                nodes_to_remove = [comp for index, comp in enumerate(components) if index != largest]

                # also remove node indices corersponding to intersections as these will be regenerated
                intersection_nodes = [node.number for node in self.nodes if node.number >= external_link_number]
                nodes_to_remove.append(intersection_nodes)
                nodes_to_remove_flat = [item for sublist in nodes_to_remove for item in sublist]

                # cycle through links and get indices of those connected to unwanted nodes
                fibres_to_remove_flat = []
                for index, link in enumerate(fibres):
                    for nid in nodes_to_remove_flat:
                        if link.node1 == nid or link.node2 == nid:
                            if index not in fibres_to_remove_flat:
                                fibres_to_remove_flat.append(index)

                # remove links and nodes
                # NB we maintain fibres list incase we have to do another iteration. This is cleaner and faster
                fibres = [link for index, link in enumerate(fibres) if index not in fibres_to_remove_flat]
                newnodes = [node for node in self.nodes if node.number not in nodes_to_remove_flat]
                ids = [idn for idn in self.node_indices if idn not in nodes_to_remove_flat]

                self.links = []  # these will be regenerated
                self.nodes = newnodes
                self.node_indices = ids
                self.count_nodes()

    def generate_linear(self, spec):
        """
        Generates a linear network with all nodes on a straight line

        Parameters
        ----------
        spec : Dictionary specifying properties of network:
            Keys:
                internal_nodes: number of internal nodes of network
                network_size: all internal nodes will be distributed randomly within range [-1/2,1/2]*network_size
                exit_size: two exit nodes placed at +/-exit_size/2

        """
        node_number = spec['internal_nodes']
        network_size = spec['network_size']
        exit_size = spec['exit_size']

        if exit_size < network_size:
            raise ValueError('exit_size must be larger than network_size.')

        # generate random positions
        x = network_size * (np.random.random(node_number) - 0.5)
        xs = sorted(x)

        # add exit nodes
        xs = np.insert(xs, 0, -exit_size / 2)
        xs = np.append(xs, exit_size / 2)

        for index in range(0, len(xs)):
            if index == 0 or index == len(xs) - 1:
                self.add_node(index, (xs[index], 0), 'exit')
            else:
                self.add_node(index, (xs[index], 0), 'internal')

        for index in range(0, len(xs) - 1):
            if index == 0 or index == len(xs) - 2:
                self.add_connection(index, index + 1, xs[index + 1] - xs[index], self.k, self.n, 'exit')
            else:
                self.add_connection(index, index + 1, xs[index + 1] - xs[index], self.k, self.n)

        self.count_nodes()

    def generate_archimedean(self, spec):
        """
        Generates a network formed from Euclidean uniform/Archimedean/Catalan tilings
            see https://en.wikipedia.org/wiki/List_of_Euclidean_uniform_tilings

        Parameters
        ----------
        spec : Dictionary specifying properties of network:
            Keys:
                internal_nodes: number of internal nodes of network
                network_size: all internal nodes will be distributed randomly within range [-1/2,1/2]*network_size
                exit_size: two exit nodes placed at +/-exit_size/2

                num_layers':3,
                'scale': network_rad,
                'type': 'square',
                'exit_nodes': 5} # square,triangular, honeycomb

        Parameters
        ----------
        spec : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        external_link_number = spec['exit_nodes']

        network_size, exit_size = self.generate_tiling(spec)

        points = [list(node.position) for node in self.nodes if node.node_type == 'internal']
        numbers = [node.number for node in self.nodes if node.node_type == 'internal']
        node_number = max(numbers) + 1

        # find network nodes on convex hull
        hullids = ConvexHull(points)
        # add some external links
        for ii in range(0, external_link_number):
            theta = 2 * math.pi * np.random.random(1)
            exitx = exit_size * np.cos(theta)[0]
            exity = exit_size * np.sin(theta)[0]
            self.add_node(node_number + ii, (exitx, exity), 'exit')

            # find appropriate connection to closest point on convex hull
            min_distance = 2 * exit_size

            for number in hullids.vertices:
                node = self.get_node(numbers[number])

                newdistance = np.sqrt((exitx - node.position[0]) ** 2 + (exity - node.position[1]) ** 2)

                if newdistance < min_distance:
                    min_distance = newdistance
                    nearest_id = node.number

            self.add_connection(node_number + ii, nearest_id, min_distance, self.k, self.n, 'exit')

    ##########################################################################
    # %%  utility functions for generating networks
    ##########################################################################
    def gen_unique_node_id(self):
        maxid = max(self.node_indices)
        return maxid + 1

    def add_node(self, number=0, position=(0, 0), node_type='internal', nodedict=None):
        """
        Adds node to node collection

        Parameters
        ----------
        number : int
            Unique id number for node.
        position : tuple
            tuple defining position of node.
        node_type : str, optional
            'exit' or 'internal' to distinguish different nodes.
            The default is 'internal'.
        nodedict : dict, optional
            supply dictionary to set node parameters from previously stored values
        """
        self.nodes.append(NODE(number, position, node_type, nodedict))
        self.node_indices.append(number)

    def add_connection(self, node1=None, node2=None, distance=None, k=None, n=1.0, link_type='internal', linkdict=None):
        """
        Adds edge to collection of links

        Parameters
        ----------
        node1, node2 : int
            Id numbers of nodes at ends of edge.
        distance : float
            Length of connection
        k : float
            wavenumber for propagation along edge
        n : float, optional
            Complex refractive index of edge. The default is 1.0.
        link_type : str, optional
            Specifies whether edge is an 'internal' (default) or connects to an 'exit' node.
        linkdict : dictionary
            supply dictionary to set link properties from stored/previous values
            Default is None

        """
        self.links.append(LINK(node1, node2, distance, k, n, link_type, linkdict))

    def remove_duplicates(self, ):
        """
         Removes duplicate connections between the same nodes and
         duplicate node ids from the corresponding collections

         """
        newnodes = []
        newlinks = []
        ids = []
        pairs = []

        for connection in self.links:
            if (((connection.node1, connection.node2) not in pairs)
                    and ((connection.node2, connection.node1) not in pairs)):
                newlinks.append(connection)
                pairs.append((connection.node1, connection.node2))

        for node in self.nodes:
            if node.number not in ids:
                newnodes.append(node)
                ids.append(node.number)

        self.links = newlinks
        self.nodes = newnodes
        self.node_indices = ids

    def get_node(self, number):
        """
        Returns the Node object from the network node collection with the
        specified identifying number

        Parameters
        ----------
        number : int
            id of desired node.

        Returns
        -------
        Node()
            Node object.

        """
        if len(self.nodenumber_indices) != len(self.nodes):
            self.nodenumber_indices = {}
            for index, node in enumerate(self.nodes):
                self.nodenumber_indices[node.number] = index

        return self.nodes[self.nodenumber_indices[number]]

    def get_exit_node_ids(self, ):
        """
        Returns a list of the node numbers of the exit nodes

        """
        ids = [node.number for node in self.nodes if node.node_type == 'exit']
        return ids

    def get_link(self, i, j):
        """
        Returns the link connection nodes with ids i,j

        """
        for link in self.links:
            nodes = (link.node1, link.node2)
            if (nodes == (i, j)) or (nodes == (j, i)):
                return link

    def get_link_index(self, i, j):
        """
        Returns the index of the link connection nodes with ids i,j in self.links

        """
        for index, link in enumerate(self.links):
            nodes = (link.node1, link.node2)
            if (nodes == (i, j)) or (nodes == (j, i)):
                return index

    @staticmethod
    def calculate_distance(pos1, pos2):
        """
        Calculates Euclidean distance between two positions defined by pos1,pos2
        """
        distance = np.linalg.norm(np.array(pos1) - np.array(pos2))
        return distance

    def remove_node(self, number):
        """
        Removes designated node

        Parameters
        ----------
        number : int
            Id of node to remove.

        """

        # make sure index arrays are uptodate
        self.count_nodes()

        if number in self.nodenumber_indices.keys():  # node exists
            node = self.get_node(number)

            # find connected nodes
            connected_nodes = node.sorted_connected_nodes

            for cn in connected_nodes:
                # get index of edge connecting two nodes
                link_index = self.get_link_index(cn, number)
                del self.links[link_index]

                # remove reference to node from connected node
                cnode = self.get_node(cn)

                cnode.n_connect -= 1
                cnode.sorted_connected_nodes.remove(number)

                # # reinitialise scattering matrix of node if needed
                # # this should also reset input/output wave vectors
                if hasattr(cnode, 'S_mat') and cnode.S_mat is not None:
                    warnings.warn("Remove node after nodal scattering matrices were initialised. Ensure you "
                                  "reinitialise your nodes correctly.")
                #     cnode.init_Smat(cnode.scat_mat_type, cnode.scat_loss, cnode.kwargs)

            # remove node
            del self.nodes[self.nodenumber_indices[number]]

            # remove any nodes that are left floating i.e. without any connections
            nodes_to_remove = []
            for index, rnode in enumerate(self.nodes):
                if rnode.n_connect == 0:
                    nodes_to_remove.append(index)

            self.nodes = [node for j, node in enumerate(self.nodes) if j not in nodes_to_remove]

            # update network node counts
            self.count_nodes()

    def connect_nodes(self, ):
        """
        Cycles through all nodes in network and determines number and ID of
        connected nodes. Also initialise each node to have the right number
        of incoming and outgoing mode coefficients

        Returns
        -------
        None.

        """
        for node in self.nodes:
            node.n_connect = 0
            connected_nodes = []
            node.inwave = {}
            node.outwave = {}

            # determine which links connect to this node and identify the connected nodes
            for connection in self.links:
                if node.number == connection.node1 or node.number == connection.node2:
                    empty = [connection.node1, connection.node2]
                    node.n_connect += 1

                    for othernode in empty:
                        if othernode != node.number:
                            connected_nodes.append(
                                othernode)  # create a list of the nodes that this node connected with

            sorted_connected_nodes = sorted(connected_nodes)  # sort the nodes im connected to from small to large
            node.sorted_connected_nodes = sorted_connected_nodes
            node.inwave = {n: (0 + 0j) for n in sorted_connected_nodes}
            node.outwave = {n: (0 + 0j) for n in sorted_connected_nodes}

    def count_nodes(self, ):
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
        self.internal_nodes = 0
        self.exit_nodes = 0
        self.total_nodes = len(self.nodes)
        self.node_indices = []
        self.nodenumber_indices = {}

        for index, node in enumerate(self.nodes):
            if node.node_type == 'internal':
                self.internal_nodes += 1
            else:
                self.exit_nodes += 1

            self.node_indices.append(node.number)
            self.nodenumber_indices[node.number] = index

        return self.internal_nodes, self.exit_nodes, self.total_nodes

    def connected_component_nodes(self, ):
        """
        Returns
        -------
        ncomponents : int
            Number of components in network
        components : list of list
            lists of node ids present in each constituent network component

        """
        components = []
        node_ids = [node.number for node in self.nodes]

        while node_ids:
            startnode = node_ids.pop(0)
            component = self.breadth_first_search(startnode)
            components.append(sorted(component))

            # remove visited nodes from list of possible starting nodes
            for c in component:
                try:
                    node_ids.remove(c)
                except ValueError:  # component already removed
                    pass

        return len(components), components

    @staticmethod
    def intersection(line1, line2):
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

    def breadth_first_search(self, initial):
        """
        Does a breadth first search of network and returns node ids within
        the network component containing 'initial' node id

        Arguments:
        ----------
            initial (int): The id of the initial node to start the search from

        Returns:
        ----------
            list: A list of node ids within the network component containing the initial node

        """
        # Initialize visited and queue lists
        visited = []
        queue = [initial]

        # While there are still nodes to visit
        while queue:
            # Get the next node from the front of the queue
            node = queue.pop(0)
            # If the node has not been visited yet
            if node not in visited:
                # Add the node to the visited list
                visited.append(node)
                # Get the sorted list of connected nodes for the current node
                neighbours = self.get_node(node).sorted_connected_nodes
                # For each connected node
                for neighbour in neighbours:
                    # Add it to the end of the queue
                    queue.append(neighbour)

        # Return the list of visited nodes
        return visited

    def trim_network_exits(self, nexit=1):
        """
        Removes specified number of exit nodes from network and trims off
        remaining edges. Exit nodes are picked at random. If node ids are known
        then use remove_node function.

        Parameters
        ----------
        nexit : int
            Number of exit nodes to remove.
            Default = 1

        Returns
        -------
        None.

        """

        if nexit >= self.exit_nodes:
            nexit = self.exit_nodes - 1  # always leave at least one exit node

        for nn in range(0, nexit):
            # pick a random exit node ...
            exit_ids = self.get_exit_node_ids()
            to_remove = random.choice(exit_ids)
            # print('Removing exit node {}'.format(to_remove))
            # ... and remove
            self.remove_node(to_remove)

    def trim_extra_exit_node_connections(self, ):
        """
            Removes all but one connection originating from an exit node. The link
            that is remains is chosen based on the shortest distance to an internal node.
        """

        # loop over exit nodes and remove all but one of the links
        #  - remaining link is chosen based on shortest distance so we first identify
        #    which are the relevant exit-internal node pairs
        exit_ids = self.get_exit_node_ids()
        exit_connect_ids = [0] * len(exit_ids)
        exit_position = [0] * len(exit_ids)
        exit_distances = [0] * len(exit_ids)
        for count, eid in enumerate(exit_ids):
            exitnode = self.get_node(eid)
            exitnodepos = exitnode.position

            # get connected nodes distances
            cids = [cid for cid in exitnode.sorted_connected_nodes if self.get_node(cid).node_type == 'internal']
            distances = [self.calculate_distance(exitnodepos, self.get_node(cid).position) for cid in cids]
            mindistid = np.argmin(distances)

            # store necessary info for later
            exit_position[count] = exitnodepos  # exit node position
            exit_connect_ids[count] = cids[mindistid]  # connected node id
            exit_distances[count] = distances[mindistid]  # link length

        # easiest to remove exit node with all connections and then re-add node and single connection
        for eid in exit_ids:
            self.remove_node(eid)

        for eid, cid, pos, d in zip(exit_ids, exit_connect_ids, exit_position, exit_distances):
            self.add_node(eid, pos, 'exit')
            self.add_connection(eid, cid, d, self.k, self.n, 'exit')

        self.connect_nodes()
        self.count_nodes()

    def generate_tiling(self, params):
        self.internal_nodes = 0
        self.exit_nodes = 0

        septol = 1e-6 * params['scale']
        if params['type'] == 'square':
            layers = params['num_layers']
            scale = params['scale']
            centerx = np.linspace(-(layers - 1), (layers - 1), 2 * (layers - 1) + 1)
            centery = np.linspace(-(layers - 1), (layers - 1), 2 * (layers - 1) + 1)
            CX, CY = np.meshgrid(centerx, centery)

            # vectors of coordinates of all vertices of 'element' of lattice
            centerx = np.ndarray.flatten(CX)
            centery = np.ndarray.flatten(CY)
            nodeid = 1

            for ix in range(0, len(centerx)):  # loop over elements
                # vertex coordinates within elements
                xv = scale * (np.array([-0.5, 0.5, -0.5, 0.5]) + centerx[ix])
                yv = scale * (np.array([-0.5, -0.5, 0.5, 0.5]) + centery[ix])
                #
                #   2 --- 3
                #   |     |
                #   |     |
                #   0 --- 1
                # order of node indices used internally by add_tile_element matches that of
                # coordinates in xv,yv.
                connections = [(0, 1), (1, 3), (3, 2), (2, 0)]
                nodeid = self.add_tile_element(xv, yv, connections, nodeid, septol)

            # remove duplicate connections
            self.remove_duplicates()
            network_size = scale * (layers - 0.5) * np.sqrt(2)
            exit_size = scale * (layers + 0.5) * np.sqrt(2)

            # add some exit nodes to those nodes on exit ring
            edge_nodes = [node for node in self.nodes if
                          (np.abs(np.linalg.norm(node.position) - network_size) <= septol)]
            for node in edge_nodes:
                pass  # future me - seems i might not have finished this function - TO DO // check

            self.total_nodes = self.internal_nodes + self.exit_nodes

        elif params['type'] == 'triangular':
            layers = params['num_layers']
            scale = params['scale']

            centerx = [0]
            centery = [0]

            for r in range(2, layers + 1):
                x = (r - 1) * -1 / 2
                y = (r - 1) * -np.sqrt(3) / 2
                centerx.append(x)
                centery.append(y)

                for jj in range(1, r):
                    x += 1
                    y += 0
                    centerx.append(x)
                    centery.append(y)

            centerx = np.array(centerx)
            centery = np.array(centery) + (layers - 1) * np.sqrt(3) / 3

            # vectors of coordinates of all vertices of 'element' of lattice
            nodeid = 1

            for ix in range(0, len(centerx)):  # loop over elements
                # vertex coordinates within elements
                xv = scale * np.array(
                    [centerx[ix] + np.sqrt(3) / 3 * np.cos(i * 2 * np.pi / 3 + np.pi / 2) for i in range(0, 3)])
                yv = scale * np.array(
                    [centery[ix] + np.sqrt(3) / 3 * np.sin(i * 2 * np.pi / 3 + np.pi / 2) for i in range(0, 3)])
                # xv =[ centerx[ix]]
                # yv =[ centery[ix]]
                #       0
                #      / \
                #    /     \
                #   1 ----- 2

                # order of node indices used internally by add_tile_element matches that of
                # coordinates in xv,yv.
                connections = [(0, 1), (1, 2), (2, 0)]
                # connections = []
                nodeid = self.add_tile_element(xv, yv, connections, nodeid, septol)

            # remove duplicate connections
            self.remove_duplicates()

            network_size = scale * layers * np.sqrt(3) / 3
            exit_size = network_size + scale
            self.total_nodes = self.internal_nodes + self.exit_nodes
        elif params['type'] == 'honeycomb':
            layers = params['num_layers']
            scale = params['scale']
            # adjust scale so that input value corresponds to the length of the side of the hexagon
            scale = scale * np.sqrt(3)

            centerx = [0]
            centery = [0]
            deltas = [(1 / 2, np.sqrt(3) / 2), (-1 / 2, np.sqrt(3) / 2), (-1, 0),
                      (-1 / 2, -np.sqrt(3) / 2), (1 / 2, -np.sqrt(3) / 2), (1, 0)]

            for r in range(2, layers + 1):
                x = (r - 1) / 2
                y = -(r - 1) * np.sqrt(3) / 2
                centerx.append(x)
                centery.append(y)

                for delta in deltas:
                    for jj in range(0, r - 1):
                        x += delta[0]
                        y += delta[1]
                        centerx.append(x)
                        centery.append(y)

            # vectors of coordinates of all vertices of 'element' of lattice
            nodeid = 1

            for ix in range(0, len(centerx)):  # loop over elements
                # vertex coordinates within elements
                xv = scale * np.array(
                    [centerx[ix] + 1 / (2 * np.cos(np.pi / 6)) * np.cos(i * np.pi / 3 + np.pi / 6) for i in
                     range(0, 6)])
                yv = scale * np.array(
                    [centery[ix] + 1 / (2 * np.cos(np.pi / 6)) * np.sin(i * np.pi / 3 + np.pi / 6) for i in
                     range(0, 6)])
                #      /1\
                #    /     \
                #   2       0
                #   |       |
                #   |       |
                #   3       5
                #    \     /
                #      \4/
                # order of node indices used internally by add_tile_element matches that of
                # coordinates in xv,yv.
                connections = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0)]
                nodeid = self.add_tile_element(xv, yv, connections, nodeid, septol)

            # remove duplicate connections
            self.remove_duplicates()

            rs = []
            for node in self.nodes:
                rs.append(self.calculate_distance(node.position, (0, 0)))
            network_size = max(rs)
            exit_size = network_size + scale
            self.total_nodes = self.internal_nodes + self.exit_nodes
        else:
            raise (ValueError, 'Unknown tiling type')
            pass

        return network_size, exit_size

    def add_tile_element(self, xv, yv, connections, nodeid, septol=1e-6):
        # add nodes if they dont already exist
        currentelement_nodes = []
        for (x, y) in zip(xv, yv):  # loop over each potentially new node
            node_exists = False
            for node in self.nodes:  # check if it exists
                if self.calculate_distance(node.position, (x, y)) <= septol:
                    node_exists = True
                    currentelement_nodes.append(node.number)
                    continue

            if node_exists is False:
                self.add_node(nodeid, (x, y), 'internal')
                currentelement_nodes.append(nodeid)
                self.internal_nodes += 1
                nodeid += 1

        # Now add appropriate connections
        for connect in connections:
            node1 = currentelement_nodes[connect[0]]
            node2 = currentelement_nodes[connect[1]]
            distance = self.calculate_distance(self.get_node(node1).position, self.get_node(node2).position)

            # we don't check if connection exists since we remove duplicates later
            self.add_connection(node1, node2, distance, self.k, self.n, link_type='internal')

        return nodeid  # return so we can keep a running id
