"""Module for perturbing the network and tracking shifts in poles"""

import copy
import functools
from dataclasses import dataclass

import numpy as np
from tqdm import tqdm

from complex_network.networks import pole_calculator
from complex_network.networks.network import Network
from complex_network.scattering_matrices import link_matrix, node_matrix


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

    def perturb_segment_n(self, link_index: int, value: complex) -> None:
        """Change the refractive index of a segment link so that it becomes
        base_n + value. Update its neighbouring node scattering matrices
        according to the fresnel coefficients"""
        link = self.perturbed_network.get_link(link_index)

        # Update refractive index
        # Copy to avoid recursion error
        Dn = link.Dn
        new_Dn = Dn + value
        link.Dn = new_Dn

        self.perturbed_network.update_segment_matrices(link)

    def change_node_scattering_matrix(
        self,
        node_index: int,
        new_node_S_mat_type: str,
        new_node_S_mat_params: dict,
    ) -> None:
        """Perturb the node scattering matrix by adding a value to the
        diagonal elements"""
        node = self.perturbed_network.get_node(node_index)

        self.perturbed_network.update_node_scattering_matrix(
            node_index, new_node_S_mat_type, new_node_S_mat_params
        )

    # -------------------------------------------------------------------------
    # Pump network with gain
    # -------------------------------------------------------------------------

    def uniform_pump(self, value: complex) -> None:
        """Pump all links in the network by altering the imaginary parts of the
        link refractive indices"""

        for link in self.perturbed_network.internal_links:
            Dn = link.Dn
            new_Dn = Dn + value
            link.Dn = new_Dn
            self.perturbed_network.update_link_matrices(link)

    def selective_pump(self, link_index: int, value: complex) -> None:
        """Pump a link in the network by altering the imaginary part of its
        refractive indices"""
        link = self.perturbed_network.get_link(link_index)
        Dn = link.Dn
        new_Dn = Dn + value
        link.Dn = new_Dn
        self.perturbed_network.update_link_matrices(link)

    def custom_pump(self, pump_profile: dict, value: complex) -> None:
        """Pump according to a pump profile dict, which tells us the relative
        amount that each link should be pumped"""

        # Clear out zeros and divide by sum
        cleaned = {key: max(val, 0) for key, val in pump_profile.items()}
        s = sum([val for _, val in cleaned.items()])
        ratios = {key: val / s for key, val in cleaned.items()}

        for link in self.perturbed_network.internal_links:
            Dn = link.Dn
            new_Dn = Dn + value * ratios[link.index]
            link.Dn = new_Dn
            self.perturbed_network.update_link_matrices(link)

    def get_Q_pump_links(self, pole: complex) -> None:
        """Work out the pole shift for each link that would result if gain
        were added to it"""

        data_dict = {}
        for link in self.unperturbed_network.internal_links:
            link_index = link.index
            func = functools.partial(
                self.unperturbed_network.get_wigner_smith,
                variable="Dn",
                perturbed_link_index=link_index,
            )
            ws_Dn_residue = pole_calculator.get_residue(func, pole)
            ws_k0_residue = pole_calculator.get_residue(
                self.unperturbed_network.get_wigner_smith, pole
            )

            pole_shift = (
                -(-1j) * np.trace(ws_Dn_residue) / np.trace(ws_k0_residue)
            )
            data_dict[link_index] = pole_shift
        return data_dict

    # -------------------------------------------------------------------------
    # Methods associated with iterative perturbations
    # -------------------------------------------------------------------------

    def get_animation_data_segment_n(
        self,
        link_index: int,
        Dn_values: np.ndarray,
        k0_min: complex,
        k0_max: complex,
        num_points: int,
    ) -> np.ndarray:

        data = []
        previous_Dn_value = Dn_values[0]

        for Dn_value in tqdm(Dn_values[1:], leave=False):

            # This is the change in Dn compared to the previous value
            Dn_shift = Dn_value - previous_Dn_value
            previous_Dn_value = Dn_value

            # Do the perturbation
            self.perturb_segment_n(link_index, Dn_shift)

            # Get the data here
            _, _, new_data = pole_calculator.sweep(
                k0_min, k0_max, num_points, self.perturbed_network
            )
            data.append(new_data)

            # Update the networks
            self.update()

        return data

    def track_pole_segment_n(
        self,
        poles: list[complex],
        link_index: int,
        Dn_values: np.ndarray,
    ) -> tuple[dict[str, list[complex]], dict[str, list[complex]]]:
        """Track the motion of a list of poles as one performs link_n
        perturbations

        Dn_values should be an array of values that the link's
        Dn value will go through (actual values, not changes).

        It is assumed that Dn_values[0] is the current value for the network!
        """

        # Set up list for storing poles
        poles_dict = {
            "direct": [[pole] for pole in poles],
            "formula": [[pole] for pole in poles],
            "volume": [[pole] for pole in poles],
        }
        pole_shifts_dict = {
            "direct": [[] for _ in poles],
            "formula": [[] for _ in poles],
            "volume": [[] for _ in poles],
        }

        previous_Dn_value = Dn_values[0]

        for Dn_value in tqdm(Dn_values[1:]):

            # This is the change in Dn compared to the previous value
            Dn_shift = Dn_value - previous_Dn_value
            previous_Dn_value = Dn_value

            # Do the perturbation
            self.perturb_segment_n(link_index, Dn_shift)

            # -------------------------------------------------------------
            # 1) Find the new pole using numerical root finding
            for i, pole_list in enumerate(poles_dict["direct"]):
                old_pole = pole_list[-1]
                new_pole = pole_calculator.find_pole(
                    self.perturbed_network, old_pole
                )
                pole_list.append(new_pole)

                # Add new pole to the data dictionary
                new_pole_shift = new_pole - old_pole
                pole_shifts_dict["direct"][i].append(new_pole_shift)

            # -------------------------------------------------------------
            # 2) Work out pole shift from Wigner-Smith operator residues
            for i, pole_list in enumerate(poles_dict["formula"]):
                old_pole = pole_list[-1]

                ws_k0_residue = pole_calculator.get_residue(
                    self.unperturbed_network.get_wigner_smith, old_pole
                )

                func = functools.partial(
                    self.unperturbed_network.get_wigner_smith,
                    variable="Dn",
                    perturbed_link_index=link_index,
                )
                ws_Dn_residue = pole_calculator.get_residue(func, old_pole)

                new_pole_shift = (
                    -np.trace(ws_Dn_residue)
                    / np.trace(ws_k0_residue)
                    * Dn_shift
                )
                new_pole = old_pole + new_pole_shift

                pole_list.append(new_pole)
                pole_shifts_dict["formula"][i].append(new_pole_shift)

            # -------------------------------------------------------------
            # 3) Work out pole shift from volume integrals
            for i, pole_list in enumerate(poles_dict["volume"]):
                old_pole = pole_list[-1]

                ws_k0_residue = pole_calculator.get_residue(
                    self.unperturbed_network.get_wigner_smith_volume, old_pole
                )

                func = functools.partial(
                    self.unperturbed_network.get_wigner_smith_volume,
                    variable="Dn",
                    perturbed_link_index=link_index,
                )
                ws_Dn_residue = pole_calculator.get_residue(func, old_pole)

                new_pole_shift = (
                    -np.trace(ws_Dn_residue)
                    / np.trace(ws_k0_residue)
                    * Dn_shift
                )
                new_pole = old_pole + new_pole_shift

                pole_list.append(new_pole)
                pole_shifts_dict["volume"][i].append(new_pole_shift)

            # Update the networks
            self.update()

        return poles_dict, pole_shifts_dict

    def track_pole_uniform_pump(
        self,
        poles: list[complex],
        Dn_values: np.ndarray,
    ) -> tuple[dict[str, list[complex]], dict[str, list[complex]]]:
        """Track the motion of a list of poles as one pumps the network
        uniformly

        Dn_values should be an array of values that the link's
        Dn value will go through (actual values, not changes).

        It is assumed that Dn_values[0] is the current value for the network!
        """

        # Set up list for storing poles
        poles_dict = {
            "direct": [[pole] for pole in poles],
        }
        pole_shifts_dict = {
            "direct": [[] for _ in poles],
        }

        previous_Dn_value = Dn_values[0]

        for Dn_value in tqdm(Dn_values[1:]):

            # This is the change in Dn compared to the previous value
            Dn_shift = Dn_value - previous_Dn_value
            previous_Dn_value = Dn_value

            # Do the perturbation
            self.uniform_pump(Dn_shift)

            # -------------------------------------------------------------
            # 1) Find the new pole using numerical root finding
            for i, pole_list in enumerate(poles_dict["direct"]):
                old_pole = pole_list[-1]
                new_pole = pole_calculator.find_pole(
                    self.perturbed_network, old_pole
                )
                pole_list.append(new_pole)

                # Add new pole to the data dictionary
                new_pole_shift = new_pole - old_pole
                pole_shifts_dict["direct"][i].append(new_pole_shift)

            # Update the networks
            self.update()

        return poles_dict, pole_shifts_dict

    def track_pole_selective_pump(
        self,
        link_index: int,
        poles: list[complex],
        Dn_values: np.ndarray,
    ) -> tuple[dict[str, list[complex]], dict[str, list[complex]]]:
        """Track the motion of a list of poles as one pumps the network
        uniformly

        Dn_values should be an array of values that the link's
        Dn value will go through (actual values, not changes).

        It is assumed that Dn_values[0] is the current value for the network!
        """

        # Set up list for storing poles
        poles_dict = {
            "direct": [[pole] for pole in poles],
        }
        pole_shifts_dict = {
            "direct": [[] for _ in poles],
        }

        previous_Dn_value = Dn_values[0]

        for Dn_value in tqdm(Dn_values[1:]):

            # This is the change in Dn compared to the previous value
            Dn_shift = Dn_value - previous_Dn_value
            previous_Dn_value = Dn_value

            # Do the perturbation
            self.selective_pump(link_index, Dn_shift)

            # -------------------------------------------------------------
            # 1) Find the new pole using numerical root finding
            for i, pole_list in enumerate(poles_dict["direct"]):
                old_pole = pole_list[-1]
                new_pole = pole_calculator.find_pole(
                    self.perturbed_network, old_pole
                )
                pole_list.append(new_pole)

                # Add new pole to the data dictionary
                new_pole_shift = new_pole - old_pole
                pole_shifts_dict["direct"][i].append(new_pole_shift)

            # Update the networks
            self.update()

        return poles_dict, pole_shifts_dict

    def track_pole_custom_pump(
        self,
        target_pole: complex,
        poles: list[complex],
        Dn_values: np.ndarray,
    ) -> tuple[dict[str, list[complex]], dict[str, list[complex]]]:
        """Track the motion of a list of poles as one pumps the network
        uniformly

        Dn_values should be an array of values that the link's
        Dn value will go through (actual values, not changes).

        It is assumed that Dn_values[0] is the current value for the network!
        """

        # Set up list for storing poles
        poles_dict = {
            "direct": [[pole] for pole in poles],
        }
        pole_shifts_dict = {
            "direct": [[] for _ in poles],
        }

        previous_Dn_value = Dn_values[0]

        for Dn_value in tqdm(Dn_values[1:]):

            # This is the change in Dn compared to the previous value
            Dn_shift = Dn_value - previous_Dn_value
            previous_Dn_value = Dn_value

            # Get the latst pump profile and do the perturbation
            Q_dict = self.get_Q_pump_links(target_pole)
            pump_profile = {key: np.imag(val) for key, val in Q_dict.items()}
            self.custom_pump(pump_profile, Dn_shift)

            # -------------------------------------------------------------
            # 1) Find the new pole using numerical root finding
            for i, pole_list in enumerate(poles_dict["direct"]):
                old_pole = pole_list[-1]
                new_pole = pole_calculator.find_pole(
                    self.perturbed_network, old_pole
                )
                pole_list.append(new_pole)

                # Add new pole to the data dictionary
                new_pole_shift = new_pole - old_pole
                pole_shifts_dict["direct"][i].append(new_pole_shift)

            # Update the networks
            self.update()

        return poles_dict, pole_shifts_dict
