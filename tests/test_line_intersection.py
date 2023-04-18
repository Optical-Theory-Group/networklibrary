import matplotlib.pyplot as plt
import numpy as np

def intersection(line1, line2):
    """
    Find the intersection of two line segments defined by their endpoints.

    Args:
        line1 [(x1,y1),(x2,y2)]: A list containing two (x, y) coordinate tuples representing
            the endpoints of the first line segment.
        line2 [(x3,y3),(x4,y4)]: A list containing two (x, y) coordinate tuples representing
            the endpoints of the second line segment.

    Returns:
        tuple: A tuple containing the (x, y) coordinates of the intersection point,
            or None if the lines do not intersect.
    """
    # Unpack the coordinates of the line segments
    p1, p2 = line1
    p3, p4 = line2

    # Convert to numpy arrays
    p1, p2, p3, p4 = np.array(p1), np.array(p2), np.array(p3), np.array(p4)

    # Calculate the denominator of the line intersection formula
    den = np.linalg.det(np.array([p2 - p1, p4 - p3]))

    # Check if the denominator is close to 0 (i.e. lines are parallel)
    if np.isclose(den, 0):
        return None
    else:
        # Calculate the numerator of the line intersection formula
        num = np.linalg.det(np.array([p3 - p1, p4 - p3]))

        # Calculate the intersection point parameter (t)
        t = num / den

        # Check if the intersection point is within both line segments
        if t < 0 or t > 1:
            return None
        else:
            # Calculate the intersection point and return as a tuple
            return tuple(p1 + t * (p2 - p1))

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

intersection = intersection(line1, line2)
if intersection is not None:
    print(f"The lines intersect at {intersection}")
else:
    print("The lines do not intersect")

plot_lines(line1, line2, intersection)