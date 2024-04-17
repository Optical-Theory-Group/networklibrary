import os


os.chdir("..")
import matplotlib.pyplot as plt
import numpy as np

from complex_network.networks import network_factory
from complex_network.networks.network_spec import NetworkSpec


np.random.seed(1)

spec = NetworkSpec(
    network_type="delaunay",
    num_internal_nodes=20,
    num_seed_nodes=20,
    num_exit_nodes=10,
    network_shape="circular",
    network_size=1.0,
    exit_offset=1.0,
    exit_size=1.2,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network_factory.generate_network(spec).draw()

spec = NetworkSpec(
    network_type="delaunay",
    num_internal_nodes=50,
    num_exit_nodes=(5,5),
    num_seed_nodes=20,
    network_shape="slab",
    network_size=(3.0, 1.0),
    exit_size=5.0,
    exit_offset=0.1,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network_factory.generate_network(spec).draw()

spec = NetworkSpec(
    network_type="voronoi",
    num_internal_nodes=100,
    num_exit_nodes=5,
    num_seed_nodes=25,
    network_shape="circular",
    network_size=1.0,
    exit_offset=1.0,
    exit_size=1.1,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
