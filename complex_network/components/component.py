import numpy as np
import matplotlib.pyplot as plt
from typing import Any


class Component:
    """Base class for network components

    Defines some useful functions common to both components, such as
    printing, saving to file etc.
    """

    def __init__(
        self, index: int, nature: str, data: dict[str, Any] | None = None
    ):
        self.reset_values(data)
        self.index = index
        self.nature = nature

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if not isinstance(value, int):
            raise ValueError("Property 'index' must be an integer.")
        self._index = value

    @property
    def nature(self):
        return self._nature

    @nature.setter
    def nature(self, value):
        if value not in ["internal", "external"]:
            raise ValueError(
                f"Invalid type '{value}'. Must be 'internal' or 'external'."
            )
        self._nature = value

    @property
    def attr_names(self) -> list[str]:
        """Get a list of all the attribute names. Useful for printing and
        saving"""
        return list(self.get_default_values().keys())

    @staticmethod
    def get_default_values() -> dict[str, Any]:
        raise NotImplementedError("Must be implemented in subclasses")

    def reset_values(self, data: dict[str, Any] | None = None) -> None:
        default_values = self.get_default_values()
        if data is not None:
            default_values.update(data)
        for key, value in default_values.items():
            setattr(self, key, value)

    def reset_fields(self) -> None:
        """Reset the values of the fields to be zero"""
        for key in self.inwave.keys():
            self.inwave[key] = 0 + 0j
        for key in self.outwave.keys():
            self.outwave[key] = 0 + 0j
        self.inwave_np = np.zeros(self.inwave_np.shape, dtype=np.complex128)
        self.outwave_np = np.zeros(self.outwave_np.shape, dtype=np.complex128)

    def update(self, direction: str = "forward") -> None:
        """
        Updates output/input amplitudes according to scattering matrix

        Parameters
        ----------
        direction : str, optional
            Set to 'forward' or 'backwards' depending on recursive algorithm
            being used. The default is 'forward'.
        """
        if direction == "forward":
            # Use S matrix to find outgoing waves at node
            outwave_np = np.matmul(self.S_mat, self.inwave_np).T
            outwave = {
                str(node_id): val
                for node_id, val in zip(
                    self.sorted_connected_nodes, outwave_np
                )
            }
            self.outwave = outwave
            self.outwave_np = outwave_np
        elif direction == "backward":
            # Use inverse S matrix to find incoming waves at node
            inwave_np = np.matmul(self.iS_mat, self.outwave_np).T
            inwave = {
                str(node_id): val
                for node_id, val in zip(self.sorted_connected_nodes, inwave_np)
            }

            self.inwave = inwave
            self.inwave_np = inwave_np
        else:
            raise ValueError(
                'Unknown run direction type: must be "forward" or "backward"'
            )

    def to_dict(self) -> dict:
        """Return a dictionary of the component attributes"""
        return {
            v: getattr(self, v) for v in self.attr_names if hasattr(self, v)
        }

    def __str__(self):
        """String representation of the component. Prints all attributes."""
        attr_values = [
            f"{attr}: {getattr(self, attr)}"
            for attr in self.attr_names
            if hasattr(self, attr)
        ]
        return ",\n".join(attr_values)
