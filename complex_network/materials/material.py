"""Base class for materials. 

These define optical properties of the network links."""

from abc import ABC, abstractmethod


class Material(ABC):
    @property
    @abstractmethod
    def n(self):
        """Refractive index function"""
        pass

    @property
    @abstractmethod
    def dn(self):
        """Derivative of the refractive index function with respect to k0."""
        pass
