import numpy as np

import quadpy
import matplotlib.pyplot as plt
from complex_network.networks import pole_calculator


def func(z):
    return np.sin(z) / (z**2 - z)


pole = 1.0
residue = np.sin(1.0)
res = pole_calculator.get_residue(func, pole, radius=10, degree=10)
print(f"Answer is {residue}")
print(f"Numerical calculation is {res}")
