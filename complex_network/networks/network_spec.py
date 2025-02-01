"""Module defining spec objects that used to specify properties of different
components of the networks.
"""

from dataclasses import dataclass
from typing import Any

from complex_network.materials.dielectric import Dielectric
from complex_network.materials.material import Material
from complex_network.components.node import Node

VALID_NETWORK_TYPES = [
        "delaunay",
        "voronoi",
        "buffon",
        # "linear",
        # "archimedean",
    ]

@dataclass
class NetworkSpec:
    """Parameters associated with the construction of the network. Note that
    values can

    Attributes:
    ----------
    network_type: str
        Type of network to be created. See network_factory for details.

    node_S_mat_type: str
        Type of scattering matrix used. See get_S_mat in network_factory for
        details.

    node_S_mat_params:
        scat_loss: float
            Specify scattering loss parameter for node, i.e. fraction of power
            lost from the network.
        subunitary_factor: float
            Used for CUE and COE cases.
        S_mat:
            Used in custom case.
        delta:
            Used in unitary_cyclic case

    """

    network_type: str
    network_shape: str
    num_internal_nodes: int | None = None
    num_external_nodes: int | None = None
    num_seed_nodes: int | None = None
    network_size: float | tuple[float, float] | None = None
    external_size: float | None = None
    external_offset: float | None = None
    node_S_mat_type: str | None = None
    node_S_mat_params: dict[str, Any] | None = None
    material: Material | None = None
    fully_connected: bool | None = None

    def __post_init__(self) -> None:
        """Set default values for the network spec if nothing specified."""
        default_flag = False
        defaults = NetworkSpec.get_default_values(self.network_type, self.network_shape)
        for key, value in defaults.items():
            if getattr(self, key) is None:
                setattr(self, key, value)
                default_flag = True

        if default_flag:
            UserWarning("Some default values have been set for the network spec.")

    @staticmethod
    def get_default_values(network_type: str, network_shape: str) -> dict[str, Any]:
        """Default values for the network spec."""

        match network_type:
            case "delaunay":
                default_values: dict[str, Any] = {
                    "num_internal_nodes": 30,
                    "num_external_nodes": 10,
                    "num_seed_nodes": 0,
                }
            case "voronoi":
                default_values: dict[str, Any] = {
                    "num_internal_nodes": 0,
                    "num_external_nodes": 4,
                    "num_seed_nodes": 50,
                }
            case "buffon":
                default_values: dict[str, Any] = {
                    "num_internal_nodes": 0,
                    "num_external_nodes": 30, 
                    "num_seed_nodes": 0,
                    "fully_connected": True,
                }
            case "linear":
                raise NotImplementedError
            case "archimedean":
                raise NotImplementedError
            case _:
                raise ValueError(
                    f"network_type '{network_type}' is invalid."
                    f"Please choose one from {VALID_NETWORK_TYPES}."
                )
            
        node_defaults = Node.get_default_values()

        default_values.update(
        {"network_type": "delaunay",
            "network_shape": network_shape,
            "network_size": 200. if network_shape == "circular" else (200., 200.),
            "external_size": 220. if network_shape == "circular" else None,
            "external_offset": None if network_shape == "circular" else 20.,
            "node_S_mat_type": node_defaults["S_mat_type"],
            "node_S_mat_params": node_defaults["S_mat_params"],
            "material": Dielectric("glass"),
        })

        return default_values