"""Methods for returning closures that compute link scattering
matrices."""

from typing import Callable

import numpy as np

# _____________________________________________________________________________
# Standard propagation matrices for single mode waveguides
# _____________________________________________________________________________

class PropagationMatrix:
    """Callable class representing the link propgation matrix."""

    def __init__(self,link) -> None:
        self.link = link

    def __call__(self, k0: complex) -> np.ndarray:
        link = self.link
        phase_advance = np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length)

        return np.array([[0.0,phase_advance,],
                         [phase_advance,0.0,],])

def get_propagation_matrix_closure(link) -> Callable:
    """Return a callable object for link propagation matrices."""
    return PropagationMatrix(link)

class PropagationMatrixInverse:
    """Callable class representing the link propagation matrix inverse."""

    def __init__(self, link) -> None:
        self.link = link

    def __call__(self, k0: complex) -> np.ndarray:
        link = self.link
        phase_advance = np.exp(-1j * (link.n(k0) + link.Dn) * k0 * link.length)
        return np.array([[0.0, phase_advance,],
                         [phase_advance, 0.0,],])

def get_propagation_matrix_inverse_closure(link) -> Callable:
    """Standard propagation matrix inverse for links"""
    return PropagationMatrixInverse(link)

class PropagationMatrixDerivative:
    """Callable class representing the link propagation matrix derivative."""

    def __init__(self, link) -> None:
        self.link = link

    def __call__(self, k0: complex, variable: str = "k0") -> np.ndarray:
        link = self.link
        VALID_VARIABLES = ["k0", "Dn"]
        if variable not in VALID_VARIABLES:
            raise ValueError(f"Invalid variable {variable}. Pick one from {VALID_VARIABLES}")
        
        phase_advance = np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length)

        matrix_part = np.array([[0.0, phase_advance,],
                                [phase_advance, 0.0,],])
        # Match the variable to find the correct derivative factor
        match variable:
            case "k0":
                factor = (1j * link.length * (k0 * link.dn(k0) + link.n(k0) + link.Dn))
            case "Dn":
                factor = 1j * link.length * k0

        return factor * matrix_part

def get_propagation_matrix_derivative_closure(link) -> Callable:
    """Return a callable object for link propagation matrix derivatives."""
    return PropagationMatrixDerivative(link)

class PropagationMatrixDoubleDerivative:
    """Callable class representing the link propagation matrix second derivative."""

    def __init__(self, link) -> None:
        self.link = link

    def __call__(self, k0: complex, variable: str = "k0") -> np.ndarray:
        link = self.link
        VALID_VARIABLES = ["k0", "Dn"]
        if variable not in VALID_VARIABLES:
            raise ValueError(f"Invalid variable {variable}. Pick one from {VALID_VARIABLES}")
        
        phase_advance = np.exp(1j * (link.n(k0) + link.Dn) * k0 * link.length)

        matrix_part = np.array([[0.0, phase_advance,],
                                [phase_advance, 0.0,],])
        # Match the variable to find the correct second derivative factor
        match variable:
            case "k0":
                factor = (-link.length**2 * k0**2 * link.dn(k0)**2 +
                          1j * link.length * (2 * link.dn(k0) + k0 * link.d2n(k0)) +
                          -link.length**2 * (link.n(k0) + link.Dn)**2)
            case "Dn":
                factor = -link.length**2 * k0**2

        return factor * matrix_part