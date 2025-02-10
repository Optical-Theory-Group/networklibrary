"""Component base class module. 

These are things like nodes and links."""

from typing import Any

import numpy as np


class Component:
    """Base class for network components.

    Defines some useful functions common to all components, such as
    printing, saving to file etc.
    
    Note that "nature" referse to either "internal" or "external"."""

    def __init__(
        self, index: int, nature: str, data: dict[str, Any] | None = None
    ) -> None:
        self.reset_values(data)
        self.index = index
        self.nature = nature

    @property
    def index(self) -> int:
        """Index of the component.

        Used for identification or quick extraction."""
        return self._index

    @index.setter
    def index(self, value) -> None:
        if not isinstance(value, int):
            raise ValueError("Property 'index' must be an integer.")
        self._index = value

    @property
    def nature(self) -> str:
        """Nature refers to whether a component is internal or external."""
        return self._nature

    @nature.setter
    def nature(self, value) -> None:
        if value not in ["internal", "external"]:
            raise ValueError(
                f"Invalid type '{value}'. Must be 'internal' or 'external'."
            )
        self._nature = value

    @property
    def attr_names(self) -> list[str]:
        """Get a list of all the attribute names.

        Useful for printing and saving."""
        return list(self.get_default_values().keys())

    @staticmethod
    def get_default_values() -> dict[str, Any]:
        raise NotImplementedError("Must be implemented in subclasses")

    def reset_values(self, data: dict[str, Any] | None = None) -> None:
        """Mostly used to set default values of the component.

        Specific details are implemented in subclasses."""
        default_values = self.get_default_values()
        if data is not None:
            default_values.update(data)
        for key, value in default_values.items():
            setattr(self, key, value)

    def reset_fields(self) -> None:
        """Reset the values of the fields to be zero."""
        for key in self.inwave.keys():
            self.inwave[key] = 0 + 0j
        for key in self.outwave.keys():
            self.outwave[key] = 0 + 0j
        self.inwave_np = np.zeros(self.inwave_np.shape, dtype=np.complex128)
        self.outwave_np = np.zeros(self.outwave_np.shape, dtype=np.complex128)

    def to_dict(self) -> dict:
        """Return a dictionary of the component attributes."""
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
