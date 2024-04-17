import numpy as np
import scipy
from complex_network.networks.network import Network
from typing import Any, Callable
import functools
from tqdm import tqdm
import quadpy


def get_adjugate(M: np.ndarray) -> np.ndarray:
    n_row, n_col = np.shape(M)
    adjugate = np.zeros((n_row, n_col), dtype=np.complex128)

    for i in range(n_row):
        for j in range(n_col):
            modified = np.copy(M)
            modified[i, :] = np.zeros(n_col)
            modified[:, j] = np.zeros(n_row)
            modified[i, j] = 1.0
            adjugate[i, j] = np.linalg.det(modified)
    return adjugate.T


def contour_integral(
    network: Network, vertices: np.ndarray, n: float | complex
) -> np.ndarray:
    """Computes the pair of contour integrals over a rectangular
    region defined by the vertices"""

    size = network.num_external_nodes
    integral_one = np.zeros((size, size), dtype=np.complex128)
    integral_two = np.zeros((size, size), dtype=np.complex128)

    for i in range(len(vertices)):
        start = vertices[i]
        end = vertices[(i + 1) % len(vertices)]

        if np.isclose(start.real, end.real):
            direction = "vertical"
        elif np.isclose(start.imag, end.imag):
            direction = "horizontal"
        else:
            raise ValueError(
                "Contour must consist of horizontal and vertical lines"
            )

        new_one, new_two = contour_integral_segment(
            network, start, end, n, direction
        )
        integral_one += new_one
        integral_two += new_two

    return integral_one, integral_two


def contour_integral_segment(
    network: Network,
    start: complex,
    end: complex,
    n: float | complex,
    direction: str,
    num_points: int = 300,
) -> tuple[complex, complex]:
    """Computes the pair of integrals over a line segment in the complex plane,
    either horizontally or vertically.

    Parameters

    network: Network
        The complex network from which the scattering matrix is calculated.
    start: complex
        A complex number that defines the start of the contour over which the
        integral is performed
    end: complex
        A complex number that defines the end of the contour over which the
        integral is performed
    n: complex
        The refractive index
    direction: str
        A string, either "horizontal" or "vertical" that specifies the
        direction of the contour segment over which the integral is calculated
    """

    if direction not in ["horizontal", "vertical"]:
        raise ValueError("Direction must be 'horizontal' or 'vertical'.")
    if direction == "horizontal" and not np.isclose(start.imag, end.imag):
        raise ValueError(
            "Start and end points do not have the same imaginary part for a "
            "horizontal integral."
        )
    if direction == "vertical" and not np.isclose(start.real, end.real):
        raise ValueError(
            "Start and end points do not have the same real part for a "
            "vertical integral."
        )

    fixed_value = start.imag if direction == "horizontal" else start.real
    range_start = start.real if direction == "horizontal" else start.imag
    range_end = end.real if direction == "horizontal" else end.imag

    # Define integrands
    def matrix(variable):
        z = (
            variable + 1j * fixed_value
            if direction == "horizontal"
            else fixed_value + 1j * variable
        )
        return network.get_S_ee(n, z)

    # Compute integrals 'manually'. Doesn't work if you try to use quadpy
    # directly due to the way it tries to vectorise over sample points
    scheme = quadpy.c1.gauss_legendre(num_points)
    normalized_points = scheme.points
    weights = scheme.weights

    diff = (range_end - range_start) / 2.0
    mean = (range_end + range_start) / 2.0

    sample_points = diff * normalized_points + mean

    # Calculate the scattering matrices at the different values along the curve
    size = network.num_external_nodes
    data_one = np.zeros((size, size, len(sample_points)), dtype=np.complex128)
    data_two = np.zeros((size, size, len(sample_points)), dtype=np.complex128)

    for i, z in enumerate(sample_points):
        new_matrix = matrix(z)
        data_one[:, :, i] = new_matrix * weights[i]
        data_two[:, :, i] = new_matrix * weights[i] * z

    integral_one = diff * np.sum(data_one, axis=2)
    integral_two = diff * np.sum(data_two, axis=2)

    return integral_one, integral_two


def tanh_sinh(function, start, end):
    pass


def find_pole(
    network: Network,
    k0: complex,
    n_function: Callable[[float], float],
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
    func = functools.partial(
        inv_factor_det, network=network, n_function=n_function
    )

    out = scipy.optimize.minimize(
        func,
        np.array([k0.real, k0.imag]),
        method=method,
        options=options,
        bounds=bounds,
    )
    pole = out.x[0] + 1j * out.x[1]
    return pole


def inv_factor_det(
    k0: np.ndarray, network: Network, n_function: Callable[[float], float]
) -> float:
    k = k0[0] + 1j * k0[1]
    wavelength = 2 * np.pi / k0[0]
    n = n_function(wavelength)
    det = network.get_inv_factor_det(n, k)
    return np.abs(det)


def inverse_determinant(k0: np.ndarray, network: Network) -> float:
    """
    Helper function for find_poles function that calculates the determinant
    of the inverse S matrix

    Parameters
    ----------
    k0 : complex
        real part of wavenumber.
    network : Network
        instance of Network being used for minimization.

    Returns
    -------
    det
        logarithmic determinant of network scattering matrix at specified
        wavenumber.
    """

    k = k0[0] + 1j * k0[1]
    S_ee_inv = network.get_S_ee_inv(None, k)
    return np.abs(np.linalg.det(S_ee_inv))


def determinant(k0: np.ndarray, network: Network) -> float:
    """
    Helper function for find_poles function that calculates the determinant
    of the inverse S matrix

    Parameters
    ----------
    k0 : complex
        real part of wavenumber.
    network : Network
        instance of Network being used for minimization.

    Returns
    -------
    det
        logarithmic determinant of network scattering matrix at specified
        wavenumber.
    """

    k = k0[0] + 1j * k0[1]
    S_ee = network.get_S_ee(None, k)
    return np.abs(np.linalg.det(S_ee))


def sweep(
    k0_min: complex, k0_max: complex, num_points: int, network: Network
) -> np.ndarray:
    k0_reals = np.linspace(k0_min.real, k0_max.real, num_points)
    k0_imags = np.linspace(k0_min.imag, k0_max.imag, num_points)
    k0_r, k0_i = np.meshgrid(k0_reals, k0_imags)

    data = np.zeros((num_points, num_points))

    for i in tqdm(range(len(k0_reals)), leave=False):
        for j in tqdm(range(len(k0_imags)), leave=False):
            k0 = k0_r[i, j] + 1j * k0_i[i, j]
            new_data = inv_factor_det(np.array([k0.real, k0.imag]), network)
            data[i, j] = new_data

    return k0_r, k0_i, data
