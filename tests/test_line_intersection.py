import matplotlib.pyplot as plt
import numpy as np

from complexnetworklibrary.network import Network


def plot_lines(line1, line2, intersection=None):
    fig, ax = plt.subplots()
    ax.plot([line1[0][0], line1[1][0]], [line1[0][1], line1[1][1]], label='Line 1')
    ax.plot([line2[0][0], line2[1][0]], [line2[0][1], line2[1][1]], label='Line 2')
    if intersection:
        ax.plot(intersection[0], intersection[1], 'ro', label='Intersection')
    ax.legend()
    plt.show()

line1 = [(0, 0), (1, 1)]
line2 = [(1, 0), (0, 1)]

intersection = Network.intersection(line1, line2)
if intersection is not None:
    print(f"The lines intersect at {intersection}")
else:
    print("The lines do not intersect")

plot_lines(line1, line2, intersection)
