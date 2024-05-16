# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:38:08 2020

@author: Matthew Foreman

Class file for Network object. Inherits from NetworkGenerator class

"""

# setup code logging
import copy
from dataclasses import dataclass
from typing import Any
import functools

import numpy as np
from tqdm import tqdm

from complex_network import utils
from complex_network.networks import pole_calculator
from complex_network.networks.network import Network
from complex_network.scattering_matrices import node_matrix

@dataclass
class PerturbationStatus:
    """Data for keeping track of perturbations

    perturbation_type
        options:
            node_eigenvalue
                The eigenvalue of one of the network nodes is altered by a
                factor exp(1j * t), where t is the perturbation value

    perturbation_value
        how big was the perturbation? The meaning of the value depends on the
        type of perturbation. See perturbation type documentation for more info

    node_id
        What is the id of the node that was affected by the perturbation?
        (if any)

    link_id
        What is the id of the link that was affected by the perturbation?
        (if any)
    """

    perturbation_type: str | None = None
    perturbation_value: float | complex = 0.0
    node_id: int | None = None
    link_id: int | None = None


class NetworkPerturbator:
    def __init__(self, network: Network) -> None:
        self.network = network
        self.unperturbed_network = copy.deepcopy(network)
        self.perturbed_network = copy.deepcopy(network)
        self.temporary_network = copy.deepcopy(network)
        self.status = PerturbationStatus()

    def reset(self) -> None:
        """Reset the networks back to the original network"""
        reset_network = copy.deepcopy(self.network)
        self.unperturbed_network = reset_network
        self.perturbed_network = reset_network
        self.status = PerturbationStatus()

    def update(self) -> None:
        """Set the unperturbed network to be the current perturbed network"""
        current_network = copy.deepcopy(self.perturbed_network)
        self.unperturbed_network = copy.deepcopy(current_network)

    # -------------------------------------------------------------------------
    # Perturbation methods
    # -------------------------------------------------------------------------

    def perturb_link_n(self, link_index: int, value: complex) -> None:
        """Change the refractive index of a link so that it becomes
        base_n + value"""
        link = self.perturbed_network.get_link(link_index)

        # Need to copy this to avoid a recursion error
        copied_function = copy.deepcopy(link.Dn)

        def new_Dn(k0: complex) -> complex:
            return copied_function(k0) + value

        link.Dn = new_Dn

        self.status.perturbation_type = "link_n"
        self.status.perturbation_value = value
        self.status.link_id = link_index

    def perturb_segment_n(self, link_index: int, value: complex) -> None:
        """Change the refractive index of a segment link so that it becomes
        base_n + value. Update its neighbouring node scattering matrices
        according to the fresnel coefficients"""
        link = self.perturbed_network.get_link(link_index)

        # Need to copy this to avoid a recursion error
        copied_function = copy.deepcopy(link.Dn)

        def new_Dn(k0: complex) -> complex:
            return copied_function(k0) + value

        link.Dn = new_Dn

        self.status.perturbation_type = "link_n"
        self.status.perturbation_value = value
        self.status.link_id = link_index

        # Update node scattering matrices (new fresnel matrices)
        # ---------|    O     |------|     O    |---------
        # link_one | node_one | link | node_two | link_two
        node_one = self.perturbed_network.get_node(link.node_indices[0])
        node_two = self.perturbed_network.get_node(link.node_indices[1])

        # Get the right link indices and links
        first, second = node_one.sorted_connected_links
        link_one_index = first if first != link_index else second 
        link_one = self.perturbed_network.get_link(link_one_index)
        first, second = node_two.sorted_connected_links
        link_two_index = first if first != link_index else second 
        link_two = self.perturbed_network.get_link(link_two_index)

        return None

    # -------------------------------------------------------------------------
    # Methods associated with iterative perturbations
    # -------------------------------------------------------------------------

    def track_pole_link_n(
        self,
        pole: complex,
        link_index: int,
        Dn_values: np.ndarray,
    ) -> tuple[dict[str, list[complex]], dict[str, list[complex]]]:
        """Track the motion of a pole as one performs link_n perturbations

        Dn_values should be an array of values that the link's
        Dn value will go through (actual values, not changes).

        It is assumed that Dn_values[0] is the current value for the network!
        """

        # Set up list for storing poles
        poles = {
            "direct": [pole],
            "formula": [pole],
            "formula_residue": [pole],
            "formula_residue_i": [pole],
            "volume_residue": [pole],
            "volume_residue_i": [pole]
        }
        pole_shifts = {
            "direct": [],
            "formula": [],
            "formula_residue": [],
            "formula_residue_i": [],
            "volume_residue": [],
            "volume_residue_i": []
        }

        old_pole = pole
        previous_Dn_value = Dn_values[0]

        for Dn_value in tqdm(Dn_values[1:]):

            # This is the change in Dn compared to the previous value
            Dn_shift = Dn_value - previous_Dn_value
            previous_Dn_value = Dn_value

            # Do the perturbation
            self.perturb_link_n(link_index, Dn_shift)

            # 1) Find the new pole using numerical root finding
            new_pole = pole_calculator.find_pole(
                self.perturbed_network, old_pole
            )
            poles["direct"].append(new_pole)
            pole_shift = new_pole - old_pole
            pole_shifts["direct"].append(pole_shift)

            # 2) Work out pole shift from raw Wigner-Smith operators
            # (formulas and no residues)
            ws_k0 = self.unperturbed_network.get_wigner_smith(old_pole)
            ws_Dn = self.unperturbed_network.get_wigner_smith(
                old_pole, "Dn", perturbed_link_index=link_index
            )
            pole_shift = -np.trace(ws_Dn) / np.trace(ws_k0) * Dn_shift
            poles["formula"].append(poles["formula"][-1] + pole_shift)
            pole_shifts["formula"].append(pole_shift)

            # 3) Work out pole shift from Wigner-Smith operator residues
            # (formula)
            ws_k0_res = pole_calculator.get_residue(
                self.unperturbed_network.get_wigner_smith, old_pole
            )

            func = functools.partial(
                self.unperturbed_network.get_wigner_smith,
                variable="Dn",
                perturbed_link_index=link_index,
            )
            ws_Dn_res = pole_calculator.get_residue(func, old_pole)

            pole_shift = -np.trace(ws_Dn_res) / np.trace(ws_k0_res) * Dn_shift
            poles["formula_residue"].append(
                poles["formula_residue"][-1] + pole_shift
            )
            pole_shifts["formula_residue"].append(pole_shift)

            # 4) Formula residue with 1j
            pole_shift = -np.trace(ws_Dn_res) / 1j * Dn_shift
            poles["formula_residue_i"].append(
                poles["formula_residue_i"][-1] + pole_shift
            )
            pole_shifts["formula_residue_i"].append(pole_shift)


            # 5) Work out pole shift from volume Wigner-Smith operator residues
            ws_k0_res = pole_calculator.get_residue(
                self.unperturbed_network.get_wigner_smith_volume, old_pole
            )

            func = functools.partial(
                self.unperturbed_network.get_wigner_smith_volume,
                variable="Dn",
                perturbed_link_index=link_index,
            )
            ws_Dn_res = pole_calculator.get_residue(func, old_pole)

            pole_shift = -np.trace(ws_Dn_res) / np.trace(ws_k0_res) * Dn_shift
            poles["volume_residue"].append(
                poles["volume_residue"][-1] + pole_shift
            )
            pole_shifts["volume_residue"].append(pole_shift)

            # 6) Same as 5, but assume ws_k0_res has trace i
            pole_shift = -np.trace(ws_Dn_res) / 1j * Dn_shift
            
            poles["volume_residue_i"].append(
                poles["volume_residue_i"][-1] + pole_shift
            )
            pole_shifts["volume_residue_i"].append(pole_shift)


            # Update the networks
            old_pole = new_pole
            self.update()

        return poles, pole_shifts
