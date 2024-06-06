import functools
from typing import Any

import numpy as np
import quadpy
import scipy
from tqdm import tqdm

from complex_network.networks.network import Network


def find_pole(
    get_S_inv: callable,
    k0: complex,
    method: str = "CG",
    options: dict[str, Any] | None = None,
    bounds: tuple[Any] | None = None,
) -> complex:
    """
    Finds poles of the scattering matrix in the complex k plane using the
    inverse determinant search method.

    Parameters
    ----------
    network: Network
        The network for which the poles are found
    k0 : complex
        First guess
    n_functoon:
        Function that returns the refractive index as a function of wavelength
    method : string, optional
        Search algorithm (see optimize.minimize documentation).
        The default is 'CG'.
    options :
        Search algorithm options (see optimize.minimize documentation).
        The default is None.
    bounds : tuple of bounds, optional
        Bounds on search region (see optimize.minimize documentation).

    Returns
    -------
    pole
        Complex wavenumber defining position of pole.

    """
    func = lambda k: np.abs(np.linalg.det(get_S_inv(k[0] + 1j * k[1])))
    out = scipy.optimize.minimize(
        func,
        np.array([k0.real, k0.imag]),
        method=method,
        options=options,
        bounds=bounds,
    )
    pole = out.x[0] + 1j * out.x[1]
    return pole


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


def get_residue(
    function,
    pole: complex,
    radius: float = 0.1,
    scheme: Any = None,
    degree: int = 10,
) -> complex:
    """Find the pole of a function numerically using a contour integral.

    This is not particularly optimized, but should do the job. Make sure the
    radius is small enough so that only 1 pole is contained within."""

    # We convert the integral to polar coordinates
    # int f(z) dz = int f(pole + R*e^it) * i * R * e^it
    def integrand(t):
        arg = pole + radius * np.exp(1j * t)
        return 1j * radius * function(arg) * np.exp(1j * t)

    if scheme is None:
        scheme = quadpy.c1.gauss_legendre(degree)

    points = scheme.points
    weights = scheme.weights

    polar = (1.0 + points) * np.pi

    # Main loop for integral calculation
    integral = 0.0
    for point, weight in zip(polar, weights):
        value = integrand(point)
        integral += value * weight

    integral *= np.pi
    residue = integral / (2 * np.pi * 1j)

    return residue
