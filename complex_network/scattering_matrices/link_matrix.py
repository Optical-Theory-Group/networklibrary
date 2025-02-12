"""Methods for returning closures that compute link scattering
matrices."""

from typing import Callable

import numpy as np

# -----------------------------------------------------------------------------
# Standard propagation matrices for single mode waveguides
# -----------------------------------------------------------------------------


def get_propagation_matrix_closure(link) -> Callable:
    """Standard propagation matrix for links"""

    def get_propagation_matrix(k0: complex) -> np.ndarray:
        return np.array(
            [
                [
                    0.0,
                    np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length),
                ],
                [
                    np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length),
                    0.0,
                ],
            ]
        )

    return get_propagation_matrix


def get_propagation_matrix_inverse_closure(link) -> Callable:
    """Standard propagation matrix inverse for links"""

    def get_propagation_matrix_inverse(k0: complex) -> np.ndarray:
        return np.array(
            [
                [
                    0.0,
                    1.0
                    / np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length),
                ],
                [
                    1.0
                    / np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length),
                    0.0,
                ],
            ]
        )

    return get_propagation_matrix_inverse


def get_propagation_matrix_derivative_closure(link) -> Callable:
    """Standard propagation matrix derivative for links"""

    def get_propagation_matrix_derivative(
        k0, variable: str = "k0"
    ) -> np.ndarray:
        VALID_VARIABLES = ["k0", "Dn"]

        matrix_part = np.array(
            [
                [
                    0.0,
                    np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length),
                ],
                [
                    np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length),
                    0.0,
                ],
            ]
        )

        match variable:
            case "k0":
                factor = (
                    1j
                    * link.length
                    * (k0 * link.dn(k0) + link.n(k0) + link.Dn)
                )
            case "Dn":
                factor = 1j * link.length * k0
            case _:
                raise ValueError(
                    f"Invalid variable {variable}. Pick one "
                    f"from {VALID_VARIABLES}"
                )

        return factor * matrix_part

    return get_propagation_matrix_derivative