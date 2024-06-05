import functools
from typing import Any

import numpy as np
import quadpy
import scipy
from tqdm import tqdm

from complex_network.networks.network import Network


def sweep(
    k0_min: complex, k0_max: complex, num_points: int, get_S_inv: callable
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Look for poles in a rectangular region with diagonal ranging from
    k0_min to k0_max

                    k0_max
          ----------
          |        |
          |        |
          ----------
    k0_min
    """

    k0_reals = np.linspace(k0_min.real, k0_max.real, num_points)
    k0_imags = np.linspace(k0_min.imag, k0_max.imag, num_points)
    k0_r, k0_i = np.meshgrid(k0_reals, k0_imags)

    data = np.zeros((num_points, num_points))

    for i in tqdm(range(len(k0_reals)), leave=False):
        for j in tqdm(range(len(k0_imags)), leave=False):
            k0 = k0_r[i, j] + 1j * k0_i[i, j]
            new_data = np.abs(np.linalg.det(get_S_inv(k0)))
            data[i, j] = new_data

    return k0_r, k0_i, data
