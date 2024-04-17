from abc import ABC, abstractmethod


class Material(ABC):
    @property
    @abstractmethod
    def n(self):
        pass

    @property
    @abstractmethod
    def dn(self):
        pass
