import os
import scipy

import matplotlib.pyplot as plt
import numpy as np

from complex_network import network_factory
from complex_network.spec import NetworkSpec
from complex_network.pole_finder import sweep, find_pole
from tqdm import tqdm

np.random.seed(1)

spec = NetworkSpec(
    network_type="delaunay",
    network_shape="circular",
    num_seed_nodes=0,
    exit_offset=0.0,
    num_internal_nodes=15,
    num_exit_nodes=5,
    network_size=500e-6,
    exit_size=550e-6,
    node_S_mat_type="COE",
    node_S_mat_params={},
)

n = 1.5
network = network_factory.generate_network(spec)
network.draw()

k_min = 2 * np.pi / (520e-9) - 500j
k_max = 2 * np.pi / (500e-9) - 0j
x, y, data = sweep(k_min, k_max, 1 * 10**4, network)
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
print(det)
print(det_inv)

# Hone in on pole
pole = find_pole(network, pole_guess)
S = network.get_S_ee(n, pole)
S_inv = network.get_S_ee_inv(n, pole)
det = np.abs(np.linalg.det(S))
det_inv = np.abs(np.linalg.det(S_inv))
print(det)
print(det_inv)

# Draw the boundary
num_vals = 10**4
k0_res = np.linspace(k_min.real, k_max.real, num_vals)
k0_ims = np.linspace(0, 20, num_vals)

x = []
y = []
angles = []

for k0_re in tqdm(k0_res, leave=False):
    for k0_im in tqdm(k0_ims, leave=False):
        k0 = k0_re - 1j * k0_im
        SP = network.get_SP(n, k0)
        l, w = np.linalg.eig(SP)

        if np.max(np.abs(l)) < 1.0:
            continue

        max_index = np.argmax(np.abs(l))

        x.append(k0_re)
        y.append(-k0_im)
        angles.append(np.angle(l[max_index]))
        break

fig, ax = plt.subplots()
ax.plot(x, y)
ax.scatter(pole.real, pole.imag)
ax.set_ylim(-20, 2.0)
ax.set_xlim(k0_res[0], k0_res[-1])
