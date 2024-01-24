import numpy as np
import scipy
from complex_network.network import Network
from typing import Any
import functools
from tqdm import tqdm

def find_pole(
    network: Network,
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
    func = functools.partial(inv_factor_det, network=network)

    out = scipy.optimize.minimize(
        func,
        np.array([k0.real, k0.imag]),
        method=method,
        options=options,
        bounds=bounds,
    )
    pole = out.x[0] + 1j * out.x[1]
    return pole


def inv_factor_det(k0: np.ndarray, network: Network) -> float:
    k = k0[0] + 1j * k0[1]
    det = network.get_inv_factor_det(None, k)
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


# # This method calculates determinant of scattering matrix of the network object.
# # Input parameters:
# # - self: An instance of Network class.
# # - kmin: Minimum wavenumber value to calculate determinant.
# # - kmax: Maximum wavenumber value to calculate determinant.
# # - npr: Number of points in the real axis.
# # - npi: Number of points in the imaginary axis.
# # - takeabs: Flag to determine whether to take the absolute value of determinant or not.
# # - progress_bar_text: Text to show in the progress bar.
# def calc_det_S(
#     self,
#     kmin: complex,
#     kmax: complex,
#     npr: int,
#     npi: Union[int, None] = None,
#     takeabs: bool = True,
#     progress_bar_text: str = "",
# ):
#     """
#     This method calculates determinant of scattering matrix of the network object.

#     Inputs:
#         kmin: complex
#             Minimum wavenumber value to calculate determinant.
#         kmax: complex
#             Maximum wavenumber value to calculate determinant.
#         npr: int
#             Number of points in the real axis.
#         npi: int, optional
#             Number of points in the imaginary axis. DEFAULT = npr
#         takeabs: bool, optional
#             Flag to determine whether to take the absolute value of determinant or not. DEFAULT = True
#         progress_bar_text: str, optional
#             Text to show in the progress bar.

#     Returns:
#         KR, KI: numpy ndarrays giving meshgrid of real and imaginary k coordinates
#         detS:   abs(|SM|) or |SM| found at each position on k grid
#         kpeaks: coarse approximation of positions of peaks in detS
#     """
#     # If npi is not given, we by default set it equal to npr.
#     if npi is None:
#         npi = npr

#     # Save the original value of k.
#     korig = self.k

#     # Get real and imaginary parts of kmin and kmax and create a meshgrid of points for real/imaginary axes.
#     krmin = kmin.real
#     kimin = kmin.imag
#     krmax = kmax.real
#     kimax = kmax.imag
#     kr = np.linspace(krmin, krmax, npr)
#     ki = np.linspace(kimin, kimax, npi)
#     KR, KI = np.meshgrid(kr, ki)

#     # Initialise array to store determinant values for each complex wavenumber
#     if takeabs is True:
#         detS = np.zeros((npr, npi))
#     else:
#         detS = np.zeros((npr, npi)) + 1j * np.zeros((npr, npi))

#     # Iterate over the grid and calculate the determinant of the scattering matrix for each point.
#     for ii in range(0, npr):
#         k_real = kr[ii]
#         for jj in range(0, npi):
#             # Update the progress bar and working value of k.
#             update_progress(
#                 (ii * npr + jj) / npr**2, status=progress_bar_text
#             )
#             k = k_real + 1j * ki[jj]

#             # Reset the network with the new k value.
#             self.reset_network(k=k)

#             # Calculate the scattering matrix and node order and store SM determinant into array.
#             sm, node_order = self.scattering_matrix_direct()
#             if takeabs is True:
#                 detS[ii, jj] = abs(np.linalg.det(sm))
#             else:
#                 detS[ii, jj] = np.linalg.det(sm)

#     # Reset the network back to the original k value.
#     self.reset_network(k=korig)

#     # Find peaks of detS.
#     peak_inds = detect_peaks(abs(detS))

#     # Create an array with the wavenumbers of the peaks.
#     kpeaks = np.array(
#         [
#             kr[peak_inds[i][0]] + 1j * ki[peak_inds[i][1]]
#             for i in range(0, len(peak_inds))
#         ]
#     )

#     return KR, KI, detS, kpeaks


# def calc_det_ImSP(self, kmin, kmax, npr, npi=None, takeabs=True):
#     if npi is None:
#         npi = npr
#     korig = self.k

#     krmin = np.real(kmin)
#     kimin = np.imag(kmin)

#     krmax = np.real(kmax)
#     kimax = np.imag(kmax)

#     kr = np.linspace(krmin, krmax, npr)
#     ki = np.linspace(kimin, kimax, npi)
#     KR, KI = np.meshgrid(kr, ki)

#     if takeabs is True:
#         detS = np.zeros((npr, npi))
#     else:
#         detS = np.zeros((npr, npi)) + 1j * np.zeros((npr, npi))

#     for ii in range(0, npr):
#         k_real = kr[ii]
#         for jj in range(0, npi):
#             update_progress((ii * npr + jj) / npr**2)
#             k_imag = ki[jj]

#             k = k_real + 1j * k_imag
#             self.reset_network(k=k)
#             S11, P11 = self.get_S11_P11()
#             sm = np.eye(S11.shape[0]) - S11 @ P11
#             if takeabs is True:
#                 detS[ii, jj] = abs(np.linalg.det(sm))
#             else:
#                 detS[ii, jj] = np.linalg.det(sm)

#     self.reset_network(k=korig)

#     peak_inds = detect_peaks(abs(detS))
#     kpeaks = np.array(
#         [
#             kr[peak_inds[i][0]] + 1j * ki[peak_inds[i][1]]
#             for i in range(0, len(peak_inds))
#         ]
#     )

#     return KR, KI, detS, kpeaks


# def detect_peaks(image):
#     """
#     Takes an image and detect the peaks using the local maximum filter.
#     Returns a boolean mask of the peaks (i.e. 1 when
#     the pixel's value is the neighborhood maximum, 0 otherwise)
#     """

#     # define an 8-connected neighborhood
#     neighborhood = generate_binary_structure(2, 2)

#     # apply the local maximum filter; all pixel of maximal value
#     # in their neighborhood are set to 1
#     local_max = maximum_filter(image, footprint=neighborhood) == image
#     # local_max is a mask that contains the peaks we are
#     # looking for, but also the background.
#     # In order to isolate the peaks we must remove the background from the mask.

#     # we create the mask of the background
#     background = image == 0

#     # a little technicality: we must erode the background in order to
#     # successfully subtract it form local_max, otherwise a line will
#     # appear along the background border (artifact of the local maximum filter)
#     eroded_background = binary_erosion(
#         background, structure=neighborhood, border_value=1
#     )

#     # we obtain the final mask, containing only peaks,
#     # by removing the background from the local_max mask (xor operation)
#     detected_peaks = local_max ^ eroded_background

#     labels = measure.label(detected_peaks)
#     props = measure.regionprops(labels)
#     peak_inds = [
#         (int(prop.centroid[0]), int(prop.centroid[1])) for prop in props
#     ]

#     return peak_inds
