"""Class for defining dielectric materials.

This is used for the links' optical properties."""

import functools

import numpy as np

from complex_network.materials.material import Material
from complex_network.materials.refractive_index import (
    dn_sellmeier_k0,
    n_sellmeier_k0,
)

# Dictionaries containing sellmeier coefficients, which are used to calculate
# refractive indices. See https://en.wikipedia.org/wiki/Sellmeier_equation
#
# Sources:
# glass https://refractiveindex.info/?shelf=3d&book=glass&page=BK7
# sapphire https://refractiveindex.info/?shelf=3d&book=crystals&page=sapphire

sellmeier_B = {
    "glass": np.array([1.03961212, 0.231792344, 1.01046945]),
    "sapphire": np.array([1.4313493, 0.65054713, 5.3414021]),
}
sellmeier_C = {
    "glass": np.array([0.00600069867, 0.0200179144, 103.560653]),
    "sapphire": np.array([0.0726631, 0.1193242, 18.028251]),
}

# List of valid dielectrics used in error handling
VALID_DIELECTRICS = list(sellmeier_B.keys())


class Dielectric(Material):
    """Main class for describing dielectric network materials with dispersion.

    The Dielectric class is for specific dielectric materials with known
    physical parameters. For custom materials, e.g. with constant or
    user-defined optical properties (e.g. refractive index functions),
    use the material base class instead."""

    def __init__(self, material: str) -> None:
        if material not in VALID_DIELECTRICS:
            raise ValueError(
                f"Unknown material: {material}. Choose from "
                f"{VALID_DIELECTRICS}."
            )
        self.material = material
        B = sellmeier_B.get(material)
        C = sellmeier_C.get(material)
        self.B = B
        self.C = C
        self._n = functools.partial(n_sellmeier_k0, B=B, C=C)
        self._dn = functools.partial(dn_sellmeier_k0, B=B, C=C)
        self.default_wave_param = "k0"
