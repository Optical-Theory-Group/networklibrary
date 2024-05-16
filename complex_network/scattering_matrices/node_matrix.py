from typing import Any, Callable

import numpy as np
import scipy

from complex_network.components.component import Component

# -----------------------------------------------------------------------------
# Methods for constant node scattering matrices (independent of k0)
# -----------------------------------------------------------------------------


def get_constant_node_S_closure(
    S_mat_type: str, size: int, S_mat_params: dict[str, Any] | None = None
) -> Callable:
    """Generate a random node scattering matrix of a given size.

    S_mat_params must contain at least "S_mat_type". Options are
        'identity':
            identity matrix - complete reflection at each input
        'permute_identity' :
            permuted identity matrix - rerouting to next edge
        'uniform':
            each element takes a value in [0,1)
        'isotropic_unitary':
            unitary isotropic SM, implemented through DFT matrix of correct
            dimension
        'COE' :
            drawn from circular orthogonal ensemble
        'CUE' :
            drawn from circular unitary ensemble
        'unitary_cyclic':
            unitary cyclic SM constructed through specifying phases of
            eigenvalues using 'delta'
        'to_the_lowest_index':
            reroutes all energy to connected node of lowest index
        'custom' :
            Set a custom scattering matrix. Requires kwarg 'S_mat' to be set
    """

    if S_mat_params is None:
        S_mat_params = {}

    valid_S_mat_types = [
        "identity",
        "uniform_random",
        "isotropic_unitary",
        "CUE",
        "COE",
        "permute_identity",
        "custom",
        "unitary_cyclic",
    ]

    match S_mat_type:
        case "identity":
            S_mat = np.identity(size, dtype=np.complex128)

        case "gaussian_random":
            S_mat = np.random.random((size, size))

        case "isotropic_unitary":
            S_mat = scipy.linalg.dft(size) / np.sqrt(size)

        case "CUE":
            gamma = S_mat_params.get("subunitary_factor", 1.0)
            S_mat = scipy.stats.unitary_group.rvs(size) * gamma

        case "COE":
            gamma = S_mat_params.get("subunitary_factor", 1.0)
            S_mat = scipy.stats.unitary_group.rvs(size) * gamma

            S_mat = S_mat @ S_mat.T
        case "permute_identity":
            mat = np.identity(size, dtype=np.complex_)
            inds = [(i - 1) % size for i in range(size)]
            S_mat = mat[:, inds]

        case "custom":
            S_mat = S_mat_params.get("S_mat", np.array(0))
            if S_mat.shape != (size, size):
                raise ValueError(
                    "Supplied scattering matrix is of incorrect"
                    f"Given: {S_mat.shape}"
                    f"Expected: {(size, size)}"
                )

        case "unitary_cyclic":
            delta = S_mat_params.get("delta")

            if delta is not None:
                ll = np.exp(1j * delta[0:size])
            else:
                ll = np.exp(1j * 2 * np.pi * np.random.rand(size))

            s = 1 / size * scipy.linalg.dft(size) @ ll

            S_mat = np.zeros((size, size), dtype=np.complex128)
            for jj in range(0, size):
                S_mat[jj, :] = np.concatenate(
                    (s[(size - jj) : size], s[0 : size - jj])
                )

        case _:
            raise ValueError(
                f"Specified scattering matrix type is invalid. Please choose"
                f" one from {valid_S_mat_types}"
            )

    # Introduce incoherent scattering loss
    scat_loss = S_mat_params.get("scat_loss", 0.0)
    if not np.isclose(scat_loss, 0.0):
        S11 = (np.sqrt(1 - scat_loss**2)) * S_mat
        S12 = np.zeros(shape=(size, size), dtype=np.complex128)
        S21 = np.zeros(shape=(size, size), dtype=np.complex128)
        S22 = scat_loss * np.identity(size, dtype=np.complex128)
        S_mat_top_row = np.concatenate((S11, S12), axis=1)
        S_mat_bot_row = np.concatenate((S21, S22), axis=1)
        S_mat = np.concatenate((S_mat_top_row, S_mat_bot_row), axis=0)

    # We want to return a function, not a matrix (even though it's a constant
    # function)
    get_S = lambda k0: S_mat
    return get_S


def get_inverse_matrix_closure(func: Callable) -> Callable:
    """Get closure for inverse scattering matrix"""

    def get_S_inv(k0: complex) -> np.ndarray:
        return np.linalg.inv(func(k0))

    return get_S_inv


def get_zero_matrix_closure(size: int) -> Callable:
    """Get closure for zero matrices of given size.

    Mostly used for the derivative matrix in the case that the scattering
    matrix is constant."""

    def get_zero_matrix(k0, *args, **kwargs) -> np.ndarray:
        return np.zeros((size, size), dtype=np.complex128)

    return get_zero_matrix


def get_permuted_matrix_closure(
    component: Component, matrix_case: str, sorted_indices: list[int]
) -> Callable:
    """Produces closures that permute the indices of scattering matrices
    returned by a scattering matrix function.

    These are needed properly permute the matrix functions. Permutations
    are required, for example, if the relabelling of network indices results in
    a new order in a node's sorted_indices, since the node scattering matrix by
    definition acts on a vector ordered by these indices.

    matrix_case should be either "get_S", "get_S_inv" or "get_dS"."""
    func = getattr(component, matrix_case, None)
    if func is None:
        raise ValueError(f"Invalid matrix_case {matrix_case}")

    def get_permuted_matrix(k0: complex, *args, **kwargs) -> np.ndarray:
        matrix = func(k0, *args, **kwargs)
        return matrix[:, sorted_indices][sorted_indices, :]

    return get_permuted_matrix


# -----------------------------------------------------------------------------
# Fresnel node matrices used primarily for link segments
# -----------------------------------------------------------------------------


def fresnel(n1: complex, n2: complex) -> np.ndarray:
    """Fresnel scattering matrix

    n1 | n2
    """

    r = (n1 - n2) / (n1 + n2)
    r2 = (n2 - n1) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    t2 = t

    S = np.array([[r, t2], [t, r2]])
    return S
