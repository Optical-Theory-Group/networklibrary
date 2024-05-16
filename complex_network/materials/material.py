"""Base class for network materials. 

These define optical properties of the network links, such as refractive
index."""

from typing import Callable


class Material:
    """Base class for materials that the networks are made from.

    The material defines the optical properties of the network. Optical
    properties primarily refers to refractive index, but may be extended in the
    future. The base class should primarily be used when the user wants to
    define a custom refractive index function. For realistic materials with
    dispersion, use one of the sub classes instead, e.g. Dielectric in
    dielectric.py.


    If you want a constant refractive index, just provide n as a float. Don't
    worry about dn."""

    def __init__(
        self, n: Callable | float | complex, dn: Callable | None = None
    ) -> None:
        self.material = "custom"

        # Case where only n is given as a number, (fixed refractive index with
        # no dispersion). This logic is not fool-proof but should catch most
        # common errors.
        if isinstance(n, (float, complex)):
            n = lambda k0: n
            dn = lambda k0: 0.0
        elif dn is None:
            raise ValueError("If n is not constant, dn must be given.")

        self._n = n
        self._dn = dn
        self.default_wave_param = "k0"

    @property
    def n(self) -> Callable[..., float]:
        """The refractive index function."""
        return self._n

    @property
    def dn(self) -> Callable[..., float]:
        """The derivative of the refractive index with respect to k0."""
        return self._dn
