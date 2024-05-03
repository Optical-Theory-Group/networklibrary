# -*- coding: utf-8 -*-
import logging
import sys
from typing import Any

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import binary_erosion, generate_binary_structure
from skimage import measure


def get_adjugate(M: np.ndarray) -> np.ndarray:
    """Find the adjugate of a matrix"""
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


def plot_colourline(x, y, c, minc=None, maxc=None):
    plt.figure(1)
    if maxc is None:
        maxc = np.max(c)
    if minc is None:
        minc = np.min(c)
    c = cm.hot((c - minc) / (maxc - minc))
    ax = plt.gca()
    for i in np.arange(len(x) - 1):
        ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], c=c[i])
    return


def convert_seconds_to_hms(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def compare_dict(dict1, dict2):
    """
    Compare two dictionaries containing numpy arrays and other data types.
    Returns True if they are equal, False otherwise.
    """
    # if len(dict1) != len(dict2):
    #     return False
    #
    # for key in dict1:
    #     if key not in dict2:
    #         return False
    #
    #     val1 = dict1[key]
    #     val2 = dict2[key]
    #
    #     if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
    #         if not np.array_equal(val1, val2):
    #             return False
    #     else:
    #         try:
    #             if val1 != val2:
    #                 return False
    #         except:
    #             print('---------ERROR {}------------'.format(key))
    #             print('---------val1------------')
    #             print(val1)
    #             print('---------val2------------')
    #             print(val2)
    # Check if the dictionaries have the same keys
    if set(dict1.keys()) != set(dict2.keys()):
        print("Different keys")
        return False

    # Compare each value in the dictionaries
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        #
        # if type(val1) != type(val2):
        #     logging.ERROR('Different types {} vs {} in {}'.format(type(val1), type(val2), key))
        #     return False

        if isinstance(val1, np.ndarray) and isinstance(val2, np.ndarray):
            if not np.array_equal(val1, val2):
                print(
                    "Unequal numpy arrays {} vs {} in {}".format(
                        val1, val2, key
                    )
                )
                return False
        elif isinstance(val1, dict) and isinstance(val2, dict):
            if not compare_dict(val1, val2):
                print("In nested dictionary {}".format(key))
                return False
        else:
            if np.array(val1 != val2).all():
                print("Unequal values {} vs {} in {}".format(val1, val2, key))
                return False

    return True


def update_progress(progress, status="", barlength=20):
    """
    Prints a progress bar to console

    Parameters
    ----------
    progress : float
        Variable ranging from 0 to 1 indicating fractional progress.
    status : TYPE, optional
        Status text to suffix progress bar. The default is ''.
    barlength : str, optional
        Controls width of progress bar in console. The default is 20.

    """
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status += " Done.\r\n"
    block = int(round(barlength * progress))
    text = "\rPercent: [{0}] {1:.2f}% {2}".format(
        "#" * block + "-" * (barlength - block), progress * 100, status
    )
    sys.stdout.write(text)
    sys.stdout.flush()


def detect_peaks(image):
    """
    Takes an image and detect the peaks using the local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2, 2)

    # apply the local maximum filter; all pixel of maximal value
    # in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood) == image
    # local_max is a mask that contains the peaks we are
    # looking for, but also the background.
    # In order to isolate the peaks we must remove the background from the mask.

    # we create the mask of the background
    background = image == 0

    # a little technicality: we must erode the background in order to
    # successfully subtract it form local_max, otherwise a line will
    # appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(
        background, structure=neighborhood, border_value=1
    )

    # we obtain the final mask, containing only peaks,
    # by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    labels = measure.label(detected_peaks)
    props = measure.regionprops(labels)
    peak_inds = [
        (int(prop.centroid[0]), int(prop.centroid[1])) for prop in props
    ]

    return peak_inds
