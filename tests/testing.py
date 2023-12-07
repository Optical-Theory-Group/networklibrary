from typing import Union

import matplotlib.pyplot as plt
# from igraph import Graph
import numpy as np
import scipy.spatial as sp
import shapely
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon
from shapely.ops import nearest_points
from shapely.plotting import plot_polygon

# from matplotlib.axes import Axes
# from mpl_toolkits.mplot3d.art3d import Poly3DCollection
#
# from PyQt6 import QtCore, QtWidgets  # Should work with PyQt5 / PySide2 / PySide6 as well
# from PyQt6.QtGui import QPolygonF, QPen, QBrush, QColor, QFont
# from PyQt6.QtCore import QRectF, QPointF, QTimer
# import pyqtgraph as pg


def get_polygon_maximum_r(polygon):
    polycenter = np.array(polygon.centroid.coords)[0]
    return max([sp.distance.euclidean(polycenter, point) for point in polygon.exterior.coords])


def _transform_polygon(poly, center, rot, origin="centroid"):
    poly_rot = rotate(poly, rot, origin=origin, use_radians=True)
    poly_shift = translate(poly_rot, center[0], center[1])
    return poly_shift


def _time_of_contact(p1: Polygon, p2: Polygon, v1: np.ndarray, v2: np.ndarray, av1: float, av2: float,
                     dt0: float, mintsep: float, exterior:bool = False) -> float:
    # we follow the conservative advancement algorithm discussed here:
    # https://www.toptal.com/game/video-game-physics-part-ii-collision-detection-for-solid-objects.
    # We assume linear translational and rotational motion between start and end time.
    # http://www.kuffner.org/james/software/dynamics/mirtich/mirtichThesis.pdf (sec. 2.3.2. We take omega_max = av)

    # # find closest points between two polygons
    # DT = 0
    # p1, p2 = nearest_points(poly1, poly2)
    # p1v = np.array(p1.coords)[0]
    # p2v = np.array(p2.coords)[0]
    # dc = np.array([p2v[0] - p1v[0], p2v[1] - p1v[1]])
    # d = np.linalg.norm(dc)
    # if d > minsep: # still far enough away
    #     B1 = np.dot(v1, +dc) + get_polygon_maximum_r(poly1) * np.abs(av1)
    #     B2 = np.dot(v2, -dc) + get_polygon_maximum_r(poly2) * np.abs(av1)
    #     DT = d / (B1 + B2)
    if exterior:
        p1 = p1.exterior
        p2 = p2.exterior
    p1_dt = _transform_polygon(p1, np.array([v1[0] * dt, v1[1] * dt]), av1 * dt)
    p2_dt = _transform_polygon(p2, np.array([v2[0] * dt, v2[1] * dt]), av2 * dt)


    if not p1_dt.intersects(p2_dt):
        return False

    t = dt / 2
    t_step = dt / 2
    i = 0
    while t_step > mintsep:
        i += 1
        print(i)
        p1_t = _transform_polygon(p1, np.array([v1[0] * t, v1[1] * t]), av1 * t)
        p2_t = _transform_polygon(p2, np.array([v2[0] * t, v2[1] * t]), av2 * t)

        if p1_t.intersects(p2_t):
            t_step /= 2
            t -= t_step
        else:
            t_step /= 2
            t += t_step


        if exterior:
            pp1 = Polygon(p1)
            pp2 = Polygon(p2)
            pp1_t = Polygon(p1_t)
            pp2_t = Polygon(p2_t)
            pp1_dt = Polygon(p1_dt)
            pp2_dt = Polygon(p2_dt)
        else:
            pp1 = p1
            pp2 = p2
            pp1_t = p1_t
            pp2_t = p2_t
            pp2_dt = p2_dt
            pp1_dt = p1_dt
        plt.close('all')
        plot_polygon(pp1)
        plot_polygon(pp2)
        # plot_polygon(pp1_t, color='green')
        plot_polygon(pp2_t, color='green')
        plot_polygon(pp1_dt, color='red')
        plot_polygon(pp2_dt, color='red')
        plt.show()
        input()


    return t


if __name__ == '__main__':
    poly1 = Polygon(((-4, -2), (-4, 3.5), (0, 3.5), (-1, -2)))
    poly2 = Polygon(((-1, -1), (-0.5, -0.25), (-.25, 3), (-1.7, 2)))
    center1 = np.array([0., 0.])
    center2 = np.array([-2., 0.])
    rot1 = 0
    rot2 = 0.

    polyinit1 = _transform_polygon(poly1, center1, rot1)
    polyinit2 = _transform_polygon(poly2, center2, rot2)

    v1 = np.array([0., 0.])
    v2 = np.array([7., 0.])
    av1 = 0.
    av2 = -6.1

    dt = .25
    minsep = 1e-5

    shiftcenter1 = center1 + v1 * dt
    shiftcenter2 = center2 + v2 * dt
    rotfinal1 = rot1 + av1 * dt
    rotfinal2 = rot2 + av2 * dt

    polyshift1 = _transform_polygon(poly1.exterior, shiftcenter1, rotfinal1)
    polyshift2 = _transform_polygon(poly2, shiftcenter2, rotfinal2)

    # plot_polygon(polyinit1)
    # plot_polygon(polyinit2)
    # plot_polygon(polyshift1, color='green')
    # plot_polygon(polyshift2, color='green')

    DT1 = _time_of_contact(polyinit1, polyinit2, v1, v2, av1, av2, dt, minsep, exterior=True)
    # DT1 = _time_of_contact(polyshift1, polyshift2, v1, v2, av1, av2, dt, minsep)



