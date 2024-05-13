import os
import scipy
import copy

import matplotlib.pyplot as plt
import numpy as np

from complex_network.networks import network_factory
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_perturbator import (
    NetworkPerturbator,
)
from complex_network.networks import pole_calculator
os.chdir("/var/home/niall/Code/Science/networklibrary")
# Generate the random network
np.random.seed(1)
spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    external_offset=0.0,
    num_internal_nodes=15,
    num_external_nodes=5,
    network_size=500e-6,
    external_size=550e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)
network = network_factory.generate_network(spec)
network.draw(show_indices=True)
link_index = 9

# Compare the U functions old vs new
lam0 = 500e-9
k0 = 2 * np.pi / lam0 + 1j * np.random.randn()

U_0_old = network.get_U_0_old(k0)
U_0_new = network.get_U_0(k0)
print("U_0")
print(f"Difference: {np.max(np.abs(U_0_old - U_0_new))}")
print(f"Smallest value: {np.min(np.abs(U_0_old))}")
print(f"Ratio: {np.max(np.abs(U_0_new - U_0_old)) / np.min(np.abs(U_0_old))}")
print("---")

print("U_1")
U_1_old = network.get_U_1_old(k0)
U_1_new = network.get_U_1_k0(k0)
print(f"Difference: {np.max(np.abs(U_1_old - U_1_new))}")
print(f"Smallest value: {np.min(np.abs(U_1_old))}")
print(f"Ratio: {np.max(np.abs(U_1_new - U_1_old)) / np.min(np.abs(U_1_old))}")
print("---")

print("U_2")
U_2_old = network.get_U_2_old(k0)
U_2_new = network.get_U_2_k0(k0)
print(f"Difference: {np.max(np.abs(U_2_old - U_2_new))}")
print(f"Smallest value: {np.min(np.abs(U_2_old))}")
print(f"Ratio: {np.max(np.abs(U_2_new - U_2_old)) / np.min(np.abs(U_2_old))}")
print("---")

print("U_3")
U_3_old = network.get_U_3_old(k0)
U_3_new = network.get_U_3_k0(k0)
print(f"Difference: {np.max(np.abs(U_3_old - U_3_new))}")
print(f"Smallest value: {np.min(np.abs(U_3_old))}")
print(f"Ratio: {np.max(np.abs(U_3_new - U_3_old)) / np.min(np.abs(U_3_old))}")


# Test S_ie
network_matrix_one = network.get_network_matrix(k0)
network_matrix_two = network.get_network_matrix(k0, "eigenvalue")
diff = network_matrix_one - network_matrix_two

dk = 1e-4
S_before = network.get_S_ie(k0)
S_after = network.get_S_ie(k0 + dk)
deriv = (S_after - S_before) / dk

real = network.get_dS_ie_dk0(k0)

diff=  real - deriv