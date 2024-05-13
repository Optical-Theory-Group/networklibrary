import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from complex_network.networks import network_factory
from complex_network.networks.network_perturbator import (NetworkPerturbator,
                                                          pole_calculator)
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.pole_calculator import (contour_integral,
                                                  contour_integral_segment,
                                                  find_pole, sweep)

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

perturbed_node_index = 9
perturbed_angle_index = 0

# -----------------------------------------------------------------------------
# Real frequency
# -----------------------------------------------------------------------------

k0 = 2 * np.pi / (500e-9)
network.update_links(n, k0)

S_ee = network.get_S_ee(n, k0)
S_ee_inv = network.get_S_ee_inv(n, k0)
dS_ee_dk0 = network.get_dS_ee_dk0()
dS_ee_dt = network.get_dS_ee_dt(perturbed_node_index, perturbed_angle_index)
ws_k0 = network.get_wigner_smith_k0(n, k0)
ws_t = network.get_wigner_smith_t(
    n, k0, perturbed_node_index, perturbed_angle_index
)

is_unitary = np.allclose(S_ee @ np.conj(S_ee.T) - np.identity(len(S_ee)), 0.0)
is_inverse = np.allclose(S_ee_inv - np.linalg.inv(S_ee), 0.0)
is_ws_k0 = np.allclose(ws_k0 - (-1j * S_ee_inv @ dS_ee_dk0), 0.0)
is_hermitian_k0 = np.allclose(ws_k0 - np.conj(ws_k0.T), 0.0)
is_ws_t = np.allclose(ws_t - (-1j * S_ee_inv @ dS_ee_dt), 0.0)
is_hermitian_t = np.allclose(ws_t - np.conj(ws_t.T), 0.0)
print("Real frequency:")
print(f"The scattering matrix is unitary: {is_unitary}")
print(f"The scattering matrix inverse is correct: {is_inverse}")
print(f"The k0 WS is correct: {is_ws_k0}")
print(f"The t WS ic correct: {is_ws_t}")
print(f"The k0 WS is Hermitian: {is_hermitian_k0}")
print(f"The t WS is Hermitian: {is_hermitian_t}")

# -----------------------------------------------------------------------------
# Complex frequency
# -----------------------------------------------------------------------------

k0 = 2 * np.pi / (500e-9) - 100j
network.update_link_S_matrices(n, k0)

S_ee = network.get_S_ee(n, k0)
S_ee_inv = network.get_S_ee_inv(n, k0)
dS_ee_dk0 = network.get_dS_ee_dk0()
dS_ee_dt = network.get_dS_ee_dt(perturbed_node_index, perturbed_angle_index)
ws_k0 = network.get_wigner_smith_k0(n, k0)
ws_t = network.get_wigner_smith_t(
    n, k0, perturbed_node_index, perturbed_angle_index
)

S_dag_S_volume = network.get_S_dag_S_volume(n, k0)
S_dag_S_exit = np.conj(S_ee.T) @ S_ee
diff = S_dag_S_volume - S_dag_S_exit


is_unitary = np.allclose(S_ee @ np.conj(S_ee.T) - np.identity(len(S_ee)), 0.0)
is_inverse = np.allclose(S_ee_inv - np.linalg.inv(S_ee), 0.0)
is_ws_k0 = np.allclose(ws_k0 - (-1j * S_ee_inv @ dS_ee_dk0), 0.0)
is_hermitian_k0 = np.allclose(ws_k0 - np.conj(ws_k0.T), 0.0)
is_ws_t = np.allclose(ws_t - (-1j * S_ee_inv @ dS_ee_dt), 0.0)
is_hermitian_t = np.allclose(ws_t - np.conj(ws_t.T), 0.0)
print("Complex frequency:")
print(f"The scattering matrix is unitary: {is_unitary}")
print(f"The scattering matrix inverse is correct: {is_inverse}")
print(f"The k0 WS is correct: {is_ws_k0}")
print(f"The t WS ic correct: {is_ws_t}")
print(f"The k0 WS is Hermitian: {is_hermitian_k0}")
print(f"The t WS is Hermitian: {is_hermitian_t}")

# -----------------------------------------------------------------------------
# Pole
# -----------------------------------------------------------------------------

pole_calculator = 12532234.746804317 - 18.467637263524797j
k0 = pole_calculator
network.update_link_S_matrices(n, k0)

S_ee = network.get_S_ee(n, k0)
S_ee_inv = network.get_S_ee_inv(n, k0)
dS_ee_dk0 = network.get_dS_ee_dk0()
dS_ee_dt = network.get_dS_ee_dt(perturbed_node_index, perturbed_angle_index)
ws_k0 = network.get_wigner_smith_k0(n, k0)
ws_t = network.get_wigner_smith_t(
    n, k0, perturbed_node_index, perturbed_angle_index
)
SP = network.get_SP(n, k0)

print("Pole:")
print(f"Determinant of S at pole: {np.linalg.det(S_ee)}")
print(f"Determinant of S^-1 at pole: {np.linalg.det(S_ee_inv)}")

is_unitary = np.allclose(S_ee @ np.conj(S_ee.T) - np.identity(len(S_ee)), 0.0)
is_inverse = np.allclose(S_ee_inv - np.linalg.inv(S_ee), 0.0)
is_ws_k0 = np.allclose(ws_k0 - (-1j * S_ee_inv @ dS_ee_dk0), 0.0)
is_hermitian_k0 = np.allclose(ws_k0 - np.conj(ws_k0.T), 0.0)
is_ws_t = np.allclose(ws_t - (-1j * S_ee_inv @ dS_ee_dt), 0.0)
is_hermitian_t = np.allclose(ws_t - np.conj(ws_t.T), 0.0)
print(f"The scattering matrix is unitary: {is_unitary}")
print(f"The scattering matrix inverse is correct: {is_inverse}")
print(f"The k0 WS is correct: {is_ws_k0}")
print(f"The t WS ic correct: {is_ws_t}")
print(f"The k0 WS is Hermitian: {is_hermitian_k0}")
print(f"The t WS is Hermitian: {is_hermitian_t}")

# -----------------------------------------------------------------------------
# Perturbation
# -----------------------------------------------------------------------------

perturbed_network = copy.deepcopy(network)

dt = 0.000001
factor = np.exp(1j * dt)
perturbed_network.perturb_node_eigenvalue(
    perturbed_node_index, perturbed_angle_index, factor
)
new_pole = pole_calculator.find_pole(perturbed_network, pole_calculator)

# Check if old pole is still a pole (shouldn't be)
S_ee = perturbed_network.get_S_ee(n, pole_calculator)
S_ee_inv = perturbed_network.get_S_ee_inv(n, pole_calculator)
dS_ee_dk0 = perturbed_network.get_dS_ee_dk0()
dS_ee_dt = perturbed_network.get_dS_ee_dt(
    perturbed_node_index, perturbed_angle_index
)
ws_k0 = perturbed_network.get_wigner_smith_k0(n, pole_calculator)
ws_t = perturbed_network.get_wigner_smith_t(
    n, pole_calculator, perturbed_node_index, perturbed_angle_index
)
print(f"Determinant of perturbed S at old pole: {np.linalg.det(S_ee)}")
print(f"Determinant of perturbed S^-1 at old pole: {np.linalg.det(S_ee_inv)}")


# Check if new pole is correct (should be)
perturbed_network.update_link_S_matrices(n, new_pole)
S_ee = perturbed_network.get_S_ee(n, new_pole)
S_ee_inv = perturbed_network.get_S_ee_inv(n, new_pole)
dS_ee_dk0 = perturbed_network.get_dS_ee_dk0()
dS_ee_dt = perturbed_network.get_dS_ee_dt(
    perturbed_node_index, perturbed_angle_index
)
ws_k0 = perturbed_network.get_wigner_smith_k0(n, new_pole)
ws_t = perturbed_network.get_wigner_smith_t(
    n, new_pole, perturbed_node_index, perturbed_angle_index
)
print(f"Determinant of perturbed S at new pole: {np.linalg.det(S_ee)}")
print(f"Determinant of perturbed S^-1 at new pole: {np.linalg.det(S_ee_inv)}")

# Check perturbation theory results

dk0 = new_pole - pole_calculator
frac = dk0 / dt

# Method based on det(I - SP) = 0
S_ii = network.get_S_ii()
P_ii = network.get_P_ii()
I = np.identity(len(S_ii))
dS_ii_dt = network.get_dS_ii_dt(perturbed_node_index, perturbed_angle_index)
dP_ii_dk0 = network.get_dP_ii_dk0()
inv = np.linalg.inv(I - S_ii @ P_ii)

top = np.trace(inv @ dS_ii_dt @ P_ii)
bottom = np.trace(inv @ S_ii @ dP_ii_dk0)
frac_one = -top / bottom

# Method based on det(S^-1) = 0
ws_t = network.get_wigner_smith_t(
    n, pole_calculator, perturbed_node_index, perturbed_angle_index
)
ws_k0 = network.get_wigner_smith_k0(n, pole_calculator)

top = np.trace(ws_t)
bottom = np.trace(ws_k0)
frac_two = -top / bottom

print("Calculating dk0/dt:")
print(f"Numerical root finding of new pole: {frac}")
print(f"Theory based on det(I-SP) = 0: {frac_one}")
print(f"Theory based on det(S^-1) = 0: {frac_two}")
