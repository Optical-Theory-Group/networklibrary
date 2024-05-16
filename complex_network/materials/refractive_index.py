"""Module that defines functions for calculating refractive indices and their
derivatives."""

import numpy as np


def n_sellmeier_k0(k0: float, B: np.ndarray, C: np.ndarray) -> float:
    """Returns the refractive index using the Sellmeier equation
    https://en.wikipedia.org/wiki/Sellmeier_equation

    The input is assumed to be given as a wavenumber in units 1/m."""
    k0 = np.real(k0)
    wavelength = 2.0 * np.pi / k0
    wavelength *= 1e6

    partial = 0.0
    for b, c in zip(B, C):
        partial += b / (1.0 - c / wavelength**2)

    n = np.sqrt(1 + partial)
    return n


def dn_sellmeier_k0(k0: float, B: np.ndarray, C: np.ndarray) -> float:
    """Returns the derivative of the refractive index with respect to k0
    using the Sellmeier equation
    https://en.wikipedia.org/wiki/Sellmeier_equation

    The input is assumed to be given as a wavenumber in units 1/m."""
    k0 = np.real(k0)
    # Denominator is just n multiplied by 2 because we differentiate a square
    # root
    denominator = 2 * n_sellmeier_k0(k0, B, C)

    wavelength = 2 * np.pi / k0
    wavelength *= 1e6

    partial = 0
    for b, c in zip(B, C):
        partial += (
            b * c / (wavelength / 1e6) / (np.pi * (1 - c / wavelength**2) ** 2)
        )
    dn = partial / denominator
    dn /= 1e12
    return dn
