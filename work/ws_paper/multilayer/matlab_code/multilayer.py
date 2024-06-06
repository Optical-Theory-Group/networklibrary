"""Python interface for the multilayer Matlab code.

The code is run through octave using oct2py."""

import numpy as np
import oct2py
import copy

# Set up oct2py environment
oc = oct2py.Oct2Py()
matlab_files_path = (
    "/home/nbyrnes/Code/networklibrary/work/ws_paper/multilayer/matlab_code"
)
oc.addpath(matlab_files_path)


def get_S(k0: float | complex, n: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return the scattering matrix for a multilayer system"""
    r, _, r2, _, t, _ = oc.calc_fresnel_multilayer(n, L, k0, 0.0, nout=6)
    S = np.array([[r, t], [t, r2]], dtype=np.complex128)
    return S


def get_S_inv(k0: float | complex, n: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return the inverse scattering matrix for a multilayer system"""
    S = get_S(k0, n, L)
    return np.linalg.inv(S)


def get_dS_dk0(
    k0: float | complex, n: np.ndarray, L: np.ndarray, dk0: float = 1e-6
) -> np.ndarray:
    k0_left = k0 - dk0 / 2
    k0_right = k0 + dk0 / 2
    S_left = get_S(k0_left, n, L)
    S_right = get_S(k0_right, n, L)
    dS = (S_right - S_left) / dk0
    return dS


def get_dS_dn(
    k0: float | complex,
    n: np.ndarray,
    L: np.ndarray,
    n_index: int,
    dn: float = 1e-9,
) -> np.ndarray:
    n_left = copy.deepcopy(n)
    n_right = copy.deepcopy(n)
    n_right[n_index] += dn

    S_left = get_S(k0, n_left, L)
    S_right = get_S(k0, n_right, L)
    dS = (S_right - S_left) / dn
    return dS


def get_ws_k0(
    k0: float | complex, n: np.ndarray, L: np.ndarray, dk0: float = 1e-4
):
    ws = -1j * get_S_inv(k0, n, L) @ get_dS_dk0(k0, n, L, dk0)
    return ws


def get_ws_n(
    k0: float | complex,
    n: np.ndarray,
    L: np.ndarray,
    n_index: int,
    dn: float = 1e-9,
):
    ws = -1j * get_S_inv(k0, n, L) @ get_dS_dn(k0, n, L, n_index, dn)
    return ws
