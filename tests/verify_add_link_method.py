# This code is to verify that the added methods are consistent with the existing methods
# Lets create a triangle network, then make the same network using the adding method

# create a custom network 
"""                     
 The network has this shape   
                         o
                         |
                         |
                         o
                        / \
                       /   \
                      /     \
                     /       \
            o______o__________o________o

"""

import numpy as np
from complex_network.components.node import Node
from complex_network.components.link import Link
from complex_network.networks.network import Network
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_factory import generate_network
import matplotlib.pyplot as plt

new_get_S = None
new_get_S_inv =None
new_get_dS = None
k = 6.28e6


data={
    "get_S": new_get_S,
    "get_S_inv": new_get_S_inv,
    "get_dS": new_get_dS,
    "S_mat_type": "neumann",
    "S_mat_params": {},
}

node_dict = {
    0: Node(0, "internal", (-1,0), data = data),
    1: Node(1, 'internal', (1,0), data = data),
    2: Node(2, 'internal', (0,1), data = data),
    3: Node(3, "external", (-2,0), data = data),
    4: Node(4, 'external', (2,0), data = data),
    5: Node(5, 'external', (0,2), data = data)
}

link_dict = {
    0: Link(0, "internal", (0,1)),
    1: Link(1, "internal", (1,2)),
    2: Link(2, "internal", (0,2)),
    3: Link(3, "external", (0,3)),
    4: Link(4, "external", (1,4)),
    5: Link(5, "external", (2,5))
}

spec = NetworkSpec(
    network_type="custom",
    network_shape=None,
    node_dict=node_dict,
    link_dict=link_dict,
    network_size=2.5
)
network = generate_network(spec)


# Now lets create a seperate network of the same structure but using add methods

""""
We start with the below network and will add the missing link

                        o
                        |
                        |
                        o
                         \
                          \
                           \
                            \
         o_______o___________o_______o


"""

node_dict2 = {
    0: Node(0, "internal", (-1,0), data = data),
    1: Node(1, 'internal', (1,0), data = data),
    2: Node(2, 'internal', (0,1), data = data),
    3: Node(3, "external", (-2,0), data = data),
    4: Node(4, 'external', (2,0), data = data),
    5: Node(5, 'external', (0,2), data = data)
}

link_dict2 = {
    0: Link(0, "internal", (0,1)),
    1: Link(1, "internal", (1,2)),
    2: Link(2, "external", (0,3)),
    3: Link(3, "external", (1,4)),
    4: Link(4, "external", (2,5))
}

spec2 = NetworkSpec(
    network_type="custom",
    network_shape=None,
    node_dict=node_dict2,
    link_dict=link_dict2,
    network_size=2.5
)

network2 = generate_network(spec2)

# Add the link for network 2
network2.add_link([(0,2)])

"Uncomment if you need to visually inspect the network"
network.draw(show_indices=True)
plt.savefig("original_network.png")

plt.close()

network2.draw(show_indices=True)
plt.savefig("network_link_removed.png")

# Test scattering matrix calculation

see1 = network.get_S_ee(k)
see2 = network2.get_S_ee(k)

if np.all(see1 == see2):
    print("The two methods give the same scattering matrix")
else:
    print("Something is wrong, the two methods give different scattering matrices")