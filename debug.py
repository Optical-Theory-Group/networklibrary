import matplotlib.pyplot as plt
import numpy as np

from complex_network.networks import network_factory
from complex_network.networks.network_spec import NetworkSpec

for i in range(100):
    np.random.seed(i)
    # fig, ax = plt.subplots()
    spec = NetworkSpec(
        network_type="voronoi",
        network_shape="circular",
    )
    network = network_factory.generate_network(spec)
    # network.draw(
    #     ax=ax,
    #     equal_aspect=True,
    #     highlight_links=[],
    #     show_external_indices=False,
    #     show_indices=False,
    #     highlight_nodes=[],
    #     draw_boundary=spec.network_size,
    # )
    S = network.get_S_ee(1)
    print(i)