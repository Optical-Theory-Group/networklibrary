# -*- coding: utf-8 -*-

import sys
import numpy as np
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
from skimage import measure
import matplotlib.cm as cm
import matplotlib.pyplot as plt




def plot_colourline(x,y,c,minc=None,maxc=None):
    plt.figure(1)
    if maxc == None: maxc = np.max(c)
    if minc == None: minc = np.min(c)
    c = cm.hot((c-minc)/(maxc - minc))
    ax = plt.gca()
    for i in np.arange(len(x)-1):
        ax.plot([x[i],x[i+1]], [y[i],y[i+1]], c=c[i])
    return


def convert_seconds_to_hms(seconds):
    seconds = seconds % (24 * 3600)
    hour = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    return "%d:%02d:%02d" % (hour, minutes, seconds)


def update_progress(progress,status='',barLength=20):
    """
    Prints a progress bar to console

    Parameters
    ----------
    progress : float
        Variable ranging from 0 to 1 indicating fractional progress.
    status : TYPE, optional
        Status text to suffix progress bar. The default is ''.
    barLength : str, optional
        Controls width of progress bar in console. The default is 20.

    """
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = 'error: progress var must be float\r\n'
    if progress < 0:
        progress = 0
        status = 'Halt...\r\n'
    if progress >= 1:
        progress = 1
        status += ' Done.\r\n'
    block = int(round(barLength*progress))
    text = '\rPercent: [{0}] {1:.2f}% {2}'.format( '#'*block + '-'*(barLength-block), progress*100, status)
    sys.stdout.write(text)
    sys.stdout.flush()

def detect_peaks(image):
    """
    Takes an image and detect the peaks usingthe local maximum filter.
    Returns a boolean mask of the peaks (i.e. 1 when
    the pixel's value is the neighborhood maximum, 0 otherwise)
    """

    # define an 8-connected neighborhood
    neighborhood = generate_binary_structure(2,2)

    #apply the local maximum filter; all pixel of maximal value
    #in their neighborhood are set to 1
    local_max = maximum_filter(image, footprint=neighborhood)==image
    #local_max is a mask that contains the peaks we are
    #looking for, but also the background.
    #In order to isolate the peaks we must remove the background from the mask.

    #we create the mask of the background
    background = (image==0)

    #a little technicality: we must erode the background in order to
    #successfully subtract it form local_max, otherwise a line will
    #appear along the background border (artifact of the local maximum filter)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

    #we obtain the final mask, containing only peaks,
    #by removing the background from the local_max mask (xor operation)
    detected_peaks = local_max ^ eroded_background

    labels = measure.label(detected_peaks)
    props = measure.regionprops(labels)
    peak_inds = [(int(prop.centroid[0]),int(prop.centroid[1])) for prop in props]

    return peak_inds

