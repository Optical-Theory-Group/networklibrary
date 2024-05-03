"""Class for defining dielectric materials.

This is used for the links' optical properties."""

import functools
from typing import Callable

import numpy as np

from complex_network.materials.material import Material
from complex_network.materials.refractive_index import (
    dn_sellmeier_k0,
    n_sellmeier_k0,
)

# UPDATE THIS SET WHEN MORE MATERIALS ARE ADDED
VALID_DIELECTRICS = {"glass", "sapphire"}

# Dictionaries containing sellmeier coefficients, which are used to calculate
# refractive indices. See https://en.wikipedia.org/wiki/Sellmeier_equation
sellmeier_B = {
    "glass": np.array([1.03961212, 0.231792344, 1.01046945]),
    "sapphire": np.array([1.4313493, 0.65054713, 5.3414021]),
}
sellmeier_C = {
    "glass": np.array([0.00600069867, 0.0200179144, 103.560653]),
    "sapphire": np.array([0.0726631, 0.1193242, 18.028251]),
}


class Dielectric(Material):
    """Main class for describing dielectric network materials"""

    def __init__(self, material: str) -> None:
        if material not in VALID_DIELECTRICS:
            raise ValueError(
                f"Unknown material. Choose from {VALID_DIELECTRICS}."
            )
        self.material = material
        B = sellmeier_B.get(material)
        C = sellmeier_C.get(material)
        self.B = B
        self.C = C
        self._n = functools.partial(n_sellmeier_k0, B=B, C=C)
        self._dn = functools.partial(dn_sellmeier_k0, B=B, C=C)
        self.default_wave_param = "k0"

    @property
    def n(self) -> Callable[..., float]:
        """The refractive index function."""
        return self._n

    @property
    def dn(self) -> Callable[..., float]:
        """The derivative of the refractive index with respect to k0."""
        return self._dn
