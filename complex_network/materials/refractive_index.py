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

    # # Edit out after use:
    n = 1.5
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
        partial += (b * c / (wavelength / 1e6) / (np.pi * (1 - c / wavelength**2) ** 2))
    dn = partial / denominator
    # Adjust units from per meter 
    dn /= 1e12
    return dn

def group_refractive_index(k0: float, B: np.ndarray, C: np.ndarray) -> float:
    """Returns the group refractive index using the Sellmeier equation
        The input is assumed to be given as a wavenumber in units 1/m.
        
        Args:
            k0: vaccum wavenumber in 1/m 
            B: Sellmeier B coefficients
            C: Sellmeier C coefficients
            
        Returns:
            Group refractive index ng
    """
    n = n_sellmeier_k0(k0, B, C)
    dn_dk0 = dn_sellmeier_k0(k0, B, C)
    ng = n - k0 * dn_dk0
    return ng

# def dn2_sellmeier_k0(k0: float, B: np.ndarray, C: np.ndarray) -> float:
#     """Returns the second derivative of refractive index with respect to k0 (1/m).
#     using the Sellmeier equation
#     https://en.wikipedia.org/wiki/Sellmeier_equation"""
    
#     k0 = np.real(k0)
#     n = n_sellmeier_k0(k0, B, C)
    
#     wavelength_m = 2.0 * np.pi / k0
#     wavelength_um = wavelength_m * 1e6
    
#     partial_dP = 0.0  # Same as dn_sellmeier_k0's partial_dP (sum of 1e12 * dT_i/dk0)
#     partial_d2P = 0.0  # Sum of 2e24 * d^2T_i/dk0^2 for all i

#     for b, c in zip(B, C):
#         # Compute dT_i/dk0 scaled term (added to partial_dP)
#         denominator = wavelength_um**2 - c
#         if denominator == 0:
#             continue
#         term_dP = (b * c) / (wavelength_m * np.pi * (1 - c / wavelength_um**2)**2)
#         partial_dP += term_dP
        
#         # Compute d²T_i/dk0² scaled term (added to partial_d2P)
#         numerator_d2 = b * c * (3 * c + wavelength_um**2)
#         denominator_d2 = (wavelength_m**2) * (np.pi**2) * (1 - c / wavelength_um**2)**3
#         if denominator_d2 == 0:
#             continue
#         term_d2P = numerator_d2 / denominator_d2
#         partial_d2P += term_d2P
    
#     # Extract unscaled derivatives
#     dP_dk0 = partial_dP / 1e12  # dP/dk0 = sum(dT_i/dk0)
#     d2P_dk0_sq = partial_d2P / (2e24)  # d²P/dk0² = sum(d²T_i/dk0²)
    
#     # Second derivative of n w.r.t. k0: d²n/dk0² = (d²P/(2n)) - (dP²)/(2n³)
#     d2n_dk0_sq = (d2P_dk0_sq / (2 * n)) - (dP_dk0**2) / (2 * n**3)
    
#     return d2n_dk0_sq