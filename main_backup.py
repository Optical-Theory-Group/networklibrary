import copy
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy
from tqdm import tqdm

from complex_network.materials.dielectric import Dielectric
from complex_network.networks import network_factory
from complex_network.networks.network_perturbator import NetworkPerturbator
from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.pole_finder import (contour_integral,
                                                  contour_integral_segment,
                                                  find_pole, sweep)

np.random.seed(1)

spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    num_internal_nodes=15,
    num_external_nodes=5,
    network_size=500e-6,
    external_size=550e-6,
    external_offset=0.0,
    node_S_mat_type="COE",
    node_S_mat_params={},
    material = Dielectric("glass")
)

network = network_factory.generate_network(spec)
network.draw(show_indices=True)

# Broad sweep to find one of the poles
k_min = 2 * np.pi / (520e-9) - 500j
k_max = 2 * np.pi / (500e-9) - 0j
x, y, data = sweep(k_min, k_max, 1 * 10**2, network)
plt.figure()
plt.imshow(1 / data)
plt.colorbar()
min_index = np.argmin(data)
row, col = np.unravel_index(min_index, data.shape)
pole_guess = x[row, col] + 1j * y[row, col]

# Check how good the guess is
S = network.get_S_ee(n, pole_guess)
S_inv = network.get_S_ee_inv(n, pole_guess)
det = np.abs(np.linalg.det(S))
det_inv = np.abs(np.linalg.det(S_inv))
print(f"Pole guess determinant: {det}")
print(f"Pole guess determinant inerse: {det_inv}")

# Hone in on pole
pole = find_pole(network, pole_guess)
pole = 12532234.746804317 - 18.467637263524797j
S = network.get_S_ee(n, pole)
S_inv = network.get_S_ee_inv(n, pole)
det = np.abs(np.linalg.det(S))
det_inv = np.abs(np.linalg.det(S_inv))
print(f"Pole determinant: {det}")
print(f"Pole determinant inerse: {det_inv}")

network_perturbator = NetworkPerturbator(network)
factor = 1
points_per_cycle = 250
thetas = np.linspace(0, factor * 2 * np.pi, factor * points_per_cycle)
k0s_real = network_perturbator.node_eigenvalue_perturbation_iterative(
    1, 0, thetas, pole, n
)

colors = ["tab:blue", "tab:orange"]

plt.figure()
for i in range(factor):
    plt.plot(
        np.real(k0s_real)[i * points_per_cycle : (i + 1) * points_per_cycle],
        np.imag(k0s_real)[i * points_per_cycle : (i + 1) * points_per_cycle],
        color=colors[i % 2],
    )
# plt.scatter(np.real(k0s_theory), np.imag(k0s_theory), color="tab:orange")
a = k0s_real[0]
b = k0s_real[-1]

# New sweep
k0_min = np.real(b) - 14 + 0j
k0_max = np.real(a) + 14 - 210j
lam_min = 2 * np.pi / (k0_max.real * 1e-9)
lam_max = 2 * np.pi / (k0_min.real * 1e-9)

# new_x, new_y, data = sweep(k0_min, k0_max, 1 * 10**2, network)
fig, ax = plt.subplots()
im = ax.imshow(
    1 / data,
    extent=(lam_min, lam_max, k0_max.imag, k0_min.imag),
    aspect="auto",
)
ax.set_xticks(
    [
        lam_min,
        lam_min + 0.25 * (lam_max - lam_min),
        lam_min + 0.5 * (lam_max - lam_min),
        lam_min + 0.75 * (lam_max - lam_min),
        lam_max,
    ]
)
ax.set_xticklabels(["501.3614", "501.3625", "501.3636", "501.3648", "501.3659"])
ax.set_yticks([0.0, -50, -100, -150, -200])
# x_data = np.array(k0s_real).real
# y_data = np.array(k0s_real).imag
# ax.scatter(np.array(k0s_real).real, np.array(k0s_real).imag)
# Add a colorbar
cbar = fig.colorbar(im, ax=ax)
cbar.set_ticks([0.0, 200, 400, 600, 800, 1000])  
plot_area_position = ax.get_position()
plot_area_width = plot_area_position.width
plot_area_height = plot_area_position.height
fig_width_inches = fig.get_figwidth()
fig_height_inches = fig.get_figheight()
print(plot_area_width, plot_area_height, fig_width_inches, fig_height_inches)
# Print the size of the plot area
fig, ax = plt.subplots(figsize=(6.4*0.6200000000000001, 4.8*0.77))
for i in range(factor):
    plt.plot(
        np.real(k0s_real)[i * points_per_cycle : (i + 1) * points_per_cycle],
        np.imag(k0s_real)[i * points_per_cycle : (i + 1) * points_per_cycle],
        color="black",
        linewidth=10
    )
plot_area_position = ax.get_position()
plot_area_width = plot_area_position.width
plot_area_height = plot_area_position.height
fig_width_inches = fig.get_figwidth()
fig_height_inches = fig.get_figheight()
print(plot_area_width, plot_area_height, fig_width_inches, fig_height_inches)


fig.savefig("hook.svg", format="svg")
# network_copy = copy.deepcopy(network)
# network_copy.perturb_node_eigenvalue(1,0, np.exp(1j*4*np.pi))
# S = network.get_S_ee(n, k0s_real[-1])
# S_inv = network.get_S_ee_inv(n, k0s_real[-1])
# det = np.abs(np.linalg.det(S))
# det_inv = np.abs(np.linalg.det(S_inv))
# print(f"Pole determinant: {det}")
# print(f"Pole determinant inerse: {det_inv}")

# thetas, k0s = network_perturbator.node_eigenvalue_perturbation_iterative(
#     1, 0, thetas, pole, n
# )
# vertices = [
#     12530000.0 - 17.0j,
#     12530000.0 - 20.0j,
#     12533000.0 - 20.0j,
#     12533000.0 - 17.0j,
# ]

# a, b = contour_integral(network, vertices, n)

# u, s, vh = np.linalg.svd(a)
# omega = np.conj(u.T) @ b @ np.conj(vh.T) @ np.diag(1/s)
# l, w = np.linalg.eig(omega)

# # Draw the boundary
# num_vals = 3 * 10**3
# k0_res = np.linspace(k_min.real, k_max.real, num_vals)
# k0_ims = np.linspace(0, 20, num_vals)

# x = []
# y = []
# angles = []

# for k0_re in tqdm(k0_res, leave=False):
#     for k0_im in tqdm(k0_ims, leave=False):
#         k0 = k0_re - 1j * k0_im
#         SP = network.get_SP(n, k0)
#         l, w = np.linalg.eig(SP)

#         if np.max(np.abs(l)) < 1.0:
#             continue

#         max_index = np.argmax(np.abs(l))

#         x.append(k0_re)
#         y.append(-k0_im)
#         angles.append(np.angle(l[max_index]))
#         break

# fig, ax = plt.subplots()
# ax.plot(x, y)
# ax.scatter(pole.real, pole.imag)
# ax.set_ylim(-20, 2.0)
# ax.set_xlim(k0_res[0], k0_res[-1])
