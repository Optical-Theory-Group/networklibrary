"""Methods for returning closures that compute link scattering
matrices."""

from typing import Callable

import numpy as np

# -----------------------------------------------------------------------------
# Standard propagation matrices for single mode waveguides
# -----------------------------------------------------------------------------


class PropagationMatrix:
    """Callable class representing the link propagation matrix."""

    def __init__(self, link) -> None:
        self.link = link

    def __call__(self, k0: complex) -> np.ndarray:
        link = self.link
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


def get_propagation_matrix_closure(link) -> Callable:
    """Return a callable object for the link propagation matrix."""

    return PropagationMatrix(link)


class PropagationMatrixInverse:
    """Callable class for the inverse propagation matrix."""

    def __init__(self, link) -> None:
        self.link = link

    def __call__(self, k0: complex) -> np.ndarray:
        link = self.link
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


def get_propagation_matrix_inverse_closure(link) -> Callable:
    """Return a callable object for the inverse propagation matrix."""

    return PropagationMatrixInverse(link)


class PropagationMatrixDerivative:
    """Callable class for the derivative of the propagation matrix."""

    def __init__(self, link) -> None:
        self.link = link

    def __call__(self, k0: complex, variable: str = "k0") -> np.ndarray:
        link = self.link
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


def get_propagation_matrix_derivative_closure(link) -> Callable:
    """Return a callable object for the derivative of the propagation matrix."""

    return PropagationMatrixDerivative(link)
