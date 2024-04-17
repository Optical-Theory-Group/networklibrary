import numpy as np


def f(x):
    return x**2


x = 1
dx = 1e-5
y1 = f(x - dx / 2)
y2 = f(x + dx / 2)
df = (y2 - y1) / dx

print(f"Theory: {2*x}")
print(f"Numerical: {df}")