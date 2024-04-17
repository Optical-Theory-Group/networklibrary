import os
import scipy
import copy

import matplotlib.pyplot as plt
import numpy as np

from complex_network.poles import pole_finder
from complex_network.networks import network_factory
from complex_network.networks.network_spec import NetworkSpec
from complex_network.poles import pole_finder

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
network.draw()





# Find a pole (guess has already been precalculated)
guess = 12532230.332102112 - 11.136143180724291j
pole = pole_finder.find_pole(network, guess, n_glass)

# Test that the pole works
pole_real, pole_imag = np.real(pole), np.imag(pole)
pole_wavelength = 2 * np.pi / pole_real
pole_n = n_glass(pole_wavelength)
S_ee = network.get_S_ee(pole_n, pole)
S_ee_inv = network.get_S_ee_inv(pole_n, pole)
print("")
print("Pole of unperturbed network")
print(f"det(S): {np.abs(np.linalg.det(S_ee))}")
print(f"det(S^-1): {np.abs(np.linalg.det(S_ee_inv))}")

# Set up the perturbed network and find the new pole
perturbed_network = copy.deepcopy(network)
perturbed_node_index = 9
perturbed_angle_index = 0
dt = 0.1
factor = np.exp(1j * dt)
perturbed_network.perturb_node_eigenvalue(
    perturbed_node_index, perturbed_angle_index, factor
)
new_pole = pole_finder.find_pole(perturbed_network, pole, n_glass)


# Check if new pole is still a pole (shouldn't be)
pole_real, pole_imag = np.real(new_pole), np.imag(new_pole)
pole_wavelength = 2 * np.pi / pole_real
pole_n = n_glass(pole_wavelength)
S_ee = network.get_S_ee(pole_n, new_pole)
S_ee_inv = network.get_S_ee_inv(pole_n, new_pole)
print("")
print("New pole used on unperturbed network")
print(f"det(S): {np.abs(np.linalg.det(S_ee))}")
print(f"det(S^-1): {np.abs(np.linalg.det(S_ee_inv))}")

# Check if new pole is a pole of perturbed network (should be)
pole_real, pole_imag = np.real(new_pole), np.imag(new_pole)
pole_wavelength = 2 * np.pi / pole_real
pole_n = n_glass(pole_wavelength)
S_ee = perturbed_network.get_S_ee(pole_n, new_pole)
S_ee_inv = perturbed_network.get_S_ee_inv(pole_n, new_pole)
print("")
print("New pole used on perturbed network")
print(f"det(S): {np.abs(np.linalg.det(S_ee))}")
print(f"det(S^-1): {np.abs(np.linalg.det(S_ee_inv))}")
