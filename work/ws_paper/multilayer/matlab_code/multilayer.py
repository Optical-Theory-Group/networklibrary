"""Python interface for the multilayer Matlab code.

The code is run through octave using oct2py."""

import numpy as np
import oct2py

# Set up oct2py environment
oc = oct2py.Oct2Py()
matlab_files_path = (
    "/home/nbyrnes/Code/networklibrary/work/ws_paper/multilayer/matlab_code"
)
oc.addpath(matlab_files_path)


def get_S(k0: float | complex, n: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return the scattering matrix for a multilayer system"""
    R1NS, _, T1NS, _ = oc.calc_fresnel_multilayer(n, L, k0, 0.0, nout=4)
    S = np.array([[R1NS, T1NS], [T1NS, R1NS]], dtype=np.complex128)
    return S


def get_S_inv(k0: float | complex, n: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Return the inverse scattering matrix for a multilayer system"""
    S = get_S(k0, n, L)
    return np.linalg.inv(S)
