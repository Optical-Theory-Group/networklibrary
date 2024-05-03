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

import numpy as np
from tqdm import tqdm

from complex_network import utils
from complex_network.networks import pole_finder
from complex_network.networks.network import Network


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

    def perturb_node_eigenvalue(
        self, node_index: int, eigenvalue_index: int, angle: complex
    ) -> None:
        """Multiply the specified eigenvalue by the factor variable"""
        node = self.perturbed_network.get_node(node_index)
        S_mat = node.S_mat

        # Multiply the appropriate eigenvalue by the given factor
        lam, w = np.linalg.eig(S_mat)
        lam[eigenvalue_index] = lam[eigenvalue_index] * np.exp(1j * angle)
        new_S = w @ np.diag(lam) @ np.linalg.inv(w)
        node.S_mat = new_S
        node.iS_mat = np.linalg.inv(new_S)

        # Leave perturbation info here too
        node.is_perturbed = True
        node.perturbation_params = {
            "eigenvalue_index": eigenvalue_index,
            "factor": angle,
        }
        self.status.perturbation_type = "node_eigenvalue"
        self.status.perturbation_value = angle
        self.status.node_id = node.index

    def perturb_pseudonode_r(self, node_index: int, dr: float) -> None:
        """Change the reflection coefficient of the pseudonode by the given
        dr"""
        node = self.perturbed_network.get_node(node_index)
        S_mat = node.S_mat

        # Get the current value of r
        previous_r = np.abs(S_mat[0, 0])
        new_r = previous_r + dr
        new_t = np.sqrt(1.0 - new_r**2)
        new_S_mat = np.array([[-new_r, new_t], [new_t, new_r]])

        node.S_mat = new_S_mat
        node.iS_mat = np.linalg.inv(new_S_mat)

        self.status.perturbation_type = "pseudonode_r"
        self.status.perturbation_value = dr
        self.status.node_id = node.index

    def perturb_pseudonode_s(self, node_index: int, ds: float) -> None:
        """Change the position of the pseudonode along the parent link by the
        amount ds"""
        node = self.perturbed_network.get_node(node_index)
        link_one_index, link_two_index = node.sorted_connected_links
        node_one_index, node_two_index = node.sorted_connected_nodes
        node_one = self.perturbed_network.get_node(node_one_index)
        node_two = self.perturbed_network.get_node(node_two_index)
        link_one = self.perturbed_network.get_link(link_one_index)
        link_two = self.perturbed_network.get_link(link_two_index)

        # Work out the current fractional position of the pseudonode
        # and get the new position
        p = node.position
        p1 = node_one.position
        p2 = node_two.position
        previous_s = (p - p1) / (p2 - p1)
        previous_s = previous_s[0]
        new_s = previous_s + ds
        new_position = p1 + new_s * (p2 - p1)
        node.position = new_position
        node.perturbation_data["s"] = new_s

        # Get new lengths
        total_length = link_one.length + link_two.length
        link_one.length = total_length * new_s
        link_two.length = total_length * (1 - new_s)
        link_one.update_S_matrices()
        link_two.update_S_matrices()

        self.status.perturbation_type = "pseudonode_s"
        self.status.perturbation_value = ds
        self.status.node_id = node.index

    def perturb_link_n(self, link_index: int, d_alpha: complex) -> None:
        """Change the refractive index of a link so that it becomes
        base_n + alpha"""
        link = self.perturbed_network.get_link(link_index)

        # Need to copy this to avoid a recursion error
        copied_function = copy.deepcopy(link.Dn)

        def new_Dn(k0: complex) -> complex:
            return copied_function(k0) + d_alpha

        link.Dn = new_Dn

        self.status.perturbation_type = "link_n"
        self.status.perturbation_value = d_alpha
        self.status.link_id = link_index

    # -------------------------------------------------------------------------
    # Methods associated with iterative perturbations
    # -------------------------------------------------------------------------

    def perturb_link_n_iterative(
        self,
        pole: complex,
        link_index: int,
        alpha_values: np.ndarray,
    ) -> tuple[dict[str, list[complex]], dict[str, list[complex]]]:
        """Change the refractive index of a link so that it becomes
        base_n + alpha"""

        # Set up list for storing poles
        poles = {"direct": [pole], "wigner": [pole], "volume": [pole]}
        pole_shifts = {"direct": [], "wigner": [], "volume": []}

        old_pole = pole
        for i, value in enumerate(tqdm(alpha_values)):
            # If we are at the first angle, dt is just that. Otherwise
            # it's the difference between the current angle and the previous
            # one
            if i == 0:
                continue
            else:
                d_alpha = value - alpha_values[i - 1]

            # Do the perturbation
            self.perturb_link_n(link_index, d_alpha)

            # Find the new pole
            new_pole = pole_finder.find_pole(self.perturbed_network, old_pole)
            poles["direct"].append(new_pole)
            pole_shift = new_pole - old_pole
            pole_shifts["direct"].append(pole_shift)

            # Work out pole shift from Wigner-Smith operators
            ws_k0 = self.unperturbed_network.get_wigner_smith_k0(old_pole)
            ws_n = self.unperturbed_network.get_wigner_smith_n(
                old_pole, link_index
            )
            pole_shift = -np.trace(ws_n) / np.trace(ws_k0) * d_alpha
            poles["wigner"].append(poles["wigner"][-1] + pole_shift)
            pole_shifts["wigner"].append(pole_shift)

            # Work out pole shift from the volume integrals
            ws_k0_vol = self.get_wigner_smith_k0_volume(old_pole)
            ws_n_vol = self.get_wigner_smith_n_volume(old_pole, link_index)
            pole_shift = -np.trace(ws_n_vol) / np.trace(ws_k0_vol) * d_alpha
            poles["volume"].append(poles["volume"][-1] + pole_shift)
            pole_shifts["volume"].append(pole_shift)

            # Test

            # Update the networks
            old_pole = new_pole
            self.update()

        return poles, pole_shifts

    def get_wigner_smith_k0_volume(self, k0):
        U_0 = self.unperturbed_network.get_U_0(k0)
        U_1 = self.unperturbed_network.get_U_1(k0)
        U_2 = self.unperturbed_network.get_U_2(k0)
        U_3 = self.unperturbed_network.get_U_3(k0, dk=1e-4)

        pre_factor = np.linalg.inv(
            np.identity(len(U_0), dtype=np.complex128) - U_0
        )

        post_factor = U_1 + U_2 + U_3
        ws_volume = pre_factor @ post_factor
        return ws_volume

    def get_wigner_smith_n_volume(self, k0, link_index):
        U_0 = self.unperturbed_network.get_U_0(k0)
        U_1 = self.unperturbed_network.get_U_1_n(k0, link_index)
        U_2 = self.unperturbed_network.get_U_2_n(k0, link_index)
        U_3 = self.get_U_3_alpha(k0)

        pre_factor = np.linalg.inv(
            np.identity(len(U_0), dtype=np.complex128) - U_0
        )

        post_factor = U_1 + U_2 + U_3
        ws_volume = pre_factor @ post_factor
        return ws_volume

    def perturb_pseudonode_r_iterative(
        self,
        pole: complex,
        node_index: int,
        r_values: np.ndarray,
    ) -> tuple[dict[str, list[complex]], dict[str, list[complex]]]:

        # Set up list for storing poles
        poles = {"direct": [pole], "wigner": [pole]}
        pole_shifts = {"direct": [], "wigner": []}

        old_pole = pole
        for i, value in enumerate(tqdm(r_values)):
            # If we are at the first angle, dt is just that. Otherwise
            # it's the difference between the current angle and the previous
            # one
            if i == 0:
                dr = value
            else:
                dr = value - r_values[i - 1]

            # Do the perturbation
            self.perturb_pseudonode_r(node_index, dr)

            # Find the new pole
            new_pole = pole_finder.find_pole(self.perturbed_network, old_pole)
            poles["direct"].append(new_pole)
            pole_shift = new_pole - old_pole
            pole_shifts["direct"].append(pole_shift)

            # Work out pole shift from Wigner-Smith operators
            ws_k0 = self.unperturbed_network.get_wigner_smith_k0(old_pole)
            ws_r = self.unperturbed_network.get_wigner_smith_r(
                old_pole, node_index
            )
            pole_shift = -np.trace(ws_r) / np.trace(ws_k0) * dr
            poles["wigner"].append(poles["wigner"][-1] + pole_shift)
            pole_shifts["wigner"].append(pole_shift)

            # Update the networks
            old_pole = new_pole
            self.update()

        return poles, pole_shifts

    def perturb_pseudonode_s_iterative(
        self,
        pole: complex,
        node_index: int,
        s_values: np.ndarray,
    ):

        # Set up list for storing poles
        poles = {"direct": [pole], "wigner": [pole]}
        pole_shifts = {"direct": [], "wigner": []}

        old_pole = pole
        for i, value in enumerate(tqdm(s_values)):
            # If we are at the first angle, dt is just that. Otherwise
            # it's the difference between the current angle and the previous
            # one
            if i == 0:
                ds = value
            else:
                ds = value - s_values[i - 1]

            # Do the perturbation
            self.perturb_pseudonode_s(node_index, ds)

            # Find the new pole
            new_pole = pole_finder.find_pole(self.perturbed_network, old_pole)
            poles["direct"].append(new_pole)
            pole_shift = new_pole - old_pole
            pole_shifts["direct"].append(pole_shift)

            # Work out pole shift from Wigner-Smith operators
            ws_k0 = self.unperturbed_network.get_wigner_smith_k0(old_pole)
            ws_s = self.unperturbed_network.get_wigner_smith_s(
                old_pole, node_index
            )
            pole_shift = -np.trace(ws_s) / np.trace(ws_k0) * ds
            poles["wigner"].append(poles["wigner"][-1] + pole_shift)
            pole_shifts["wigner"].append(pole_shift)

            # Update the networks
            old_pole = new_pole
            self.update()

        return poles, pole_shifts

    def perturb_pseudonode_r_sweep(
        self,
        node_index: int,
        num_points: int,
        k0_min: complex,
        k0_max: complex,
        r_values: np.ndarray,
    ):

        # Set up list for storing poles
        data_array = np.zeros((len(r_values), num_points, num_points))

        for i, value in enumerate(tqdm(r_values)):
            # If we are at the first angle, dt is just that. Otherwise
            # it's the difference between the current angle and the previous
            # one
            if i == 0:
                dr = value
            else:
                dr = value - r_values[i - 1]

            # Do the perturbation
            self.perturb_pseudonode_r(node_index, dr)

            # Do the complex plane sweep
            k0_r, k0_i, data = pole_finder.sweep(
                k0_min, k0_max, num_points, self.perturbed_network
            )
            data_array[i] = data
            self.update()

        return k0_r, k0_i, data

    def perturb_node_eigenvalue_iterative(
        self,
        pole: complex,
        node_index: int,
        eigenvalue_index: int,
        angles: np.ndarray,
    ):

        # Set up list for storing poles
        poles = {
            "direct": [pole],
            "network": [pole],
            "wigner": [pole],
        }
        pole_shifts = {"direct": [], "network": [], "wigner": []}

        old_pole = pole
        for i, value in enumerate(tqdm(angles)):
            # If we are at the first angle, dt is just that. Otherwise
            # it's the difference between the current angle and the previous
            # one
            if i == 0:
                dt = value
            else:
                dt = value - angles[i - 1]

            # Do the perturbation
            self.perturb_node_eigenvalue(node_index, eigenvalue_index, dt)

            # Find the new pole
            new_pole = pole_finder.find_pole(self.perturbed_network, old_pole)
            poles["direct"].append(new_pole)
            pole_shift = new_pole - old_pole
            pole_shifts["direct"].append(pole_shift)

            # Work out pole shift from network matrix derivatives
            S_ii = self.unperturbed_network.get_S_ii()
            dS_ii = self.unperturbed_network.get_dS_ii_dt(
                node_index, eigenvalue_index
            )
            P_ii = self.unperturbed_network.get_P_ii(old_pole)
            dP_ii = self.unperturbed_network.get_dP_ii_dk0(old_pole)
            ISP = np.identity(len(S_ii), dtype=np.complex128) - S_ii @ P_ii
            adj = utils.get_adjugate(ISP)
            numerator = adj @ dS_ii @ P_ii
            denominator = adj @ S_ii @ dP_ii
            pole_shift = -np.trace(numerator) / np.trace(denominator) * dt
            poles["network"].append(poles["network"][-1] + pole_shift)
            pole_shifts["network"].append(pole_shift)

            # Work out pole shift from Wigner-Smith operators
            ws_k0 = self.unperturbed_network.get_wigner_smith_k0(old_pole)
            ws_t = self.unperturbed_network.get_wigner_smith_t(
                old_pole, node_index, eigenvalue_index
            )
            pole_shift = -np.trace(ws_t) / np.trace(ws_k0) * dt
            poles["wigner"].append(poles["wigner"][-1] + pole_shift)
            pole_shifts["wigner"].append(pole_shift)

            # Update the networks
            old_pole = new_pole
            self.update()

        return poles, pole_shifts

    def get_pole_shift_node_eigenvalue_perturbation(self):
        pass

    # -------------------------------------------------------------------------
    # Methods associated with iterative perturbations
    # -------------------------------------------------------------------------

    def node_eigenvalue_perturbation_get_S(
        self, node_index: int, eigenvalue_index: int, thetas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        num_perturbations = len(thetas)

        # First index is for S and dS/dtheta
        S_matrices = np.zeros(
            (
                2,
                num_perturbations,
                self.network.internal_vector_length,
                self.network.internal_vector_length,
            ),
            dtype=np.complex128,
        )

        factors = np.exp(1j * thetas)
        for num_factor, factor in enumerate(factors):
            # Make a fresh copy from the original
            self.network_copy = copy.deepcopy(self.network)

            # Perform the perturbation
            self.network_copy.perturb_node_eigenvalue(
                node_index, eigenvalue_index, factor
            )
            new_S, new_dS = self.network_copy.get_S_ii_dS_ii()
            S_matrices[0, num_factor, :, :] = new_S
            S_matrices[1, num_factor, :, :] = new_dS
        return S_matrices

    def node_eigenvalue_perturbation_get_P(
        self, n: complex, k0: complex
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.network_copy.get_P_ii_dP_ii(n, k0)

    def node_eigenvalue_perturbation_iterative(
        self,
        node_index: int,
        eigenvalue_index: int,
        thetas: np.ndarray,
        initial_k0: complex,
        n: complex,
    ):
        # Get S matrices
        S_matrices = self.node_eigenvalue_perturbation_get_S(
            node_index, eigenvalue_index, thetas
        )

        d_theta = thetas[1] - thetas[0]
        k0s_theory = [initial_k0]
        k0s_real = [initial_k0]
        old_k0_theory = initial_k0
        old_k0_real = initial_k0

        for iteration_number, theta in enumerate(tqdm(thetas)):
            # Do real perturbation and find the actual position of the new pole
            new_k0_real = self.node_eigenvalue_perturbation_get_pole(
                node_index, eigenvalue_index, theta, old_k0_real
            )
            k0s_real.append(new_k0_real)
            old_k0_real = new_k0_real

            # Simulated position of new pole
            # Get new P and dP matrix
            # P, dP = self.node_eigenvalue_perturbation_get_P(n, old_k0_theory)

            # # Calculate derivative
            # S = S_matrices[0, iteration_number, :, :]
            # dS = S_matrices[1, iteration_number, :, :]
            # D = np.identity(len(P), dtype=np.complex128) - S @ P
            # adj = get_adjugate(D)
            # top = adj @ dS @ P
            # bottom = adj @ S @ dP
            # derivative = -np.trace(top) / np.trace(bottom)
            # new_k0_theory = old_k0_theory + derivative * d_theta
            # k0s_theory.append(new_k0_theory)
            # old_k0_theory = new_k0_theory

        return k0s_real

    # -------------------------------------------------------------------------
    # Methods associated with volume integrals (for Wigner-Smith calculations)
    # -------------------------------------------------------------------------

    def get_U_3_alpha(self, k0) -> np.ndarray:
        """Calculate the U_3 matrix associated with the perturbation parameter
        (see theory notes)"""

        # Get network matrices
        unperturbed_network_matrix = (
            self.unperturbed_network.get_network_matrix(k0)
        )
        perturbed_network_matrix = self.perturbed_network.get_network_matrix(
            k0
        )

        # Get the scattered fields for each incident field
        unperturbed_outgoing_vectors = []
        perturbed_outgoing_vectors = []

        num_externals = self.network.num_external_nodes
        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            incident_vector = np.zeros(
                (len(unperturbed_network_matrix)), dtype=np.complex128
            )
            incident_vector[num_externals : 2 * num_externals] = incident_field
            unperturbed_outgoing_vector = (
                unperturbed_network_matrix @ incident_vector
            )
            unperturbed_outgoing_vectors.append(unperturbed_outgoing_vector)
            perturbed_outgoing_vector = (
                perturbed_network_matrix @ incident_vector
            )
            perturbed_outgoing_vectors.append(perturbed_outgoing_vector)

        U_3 = np.zeros((num_externals, num_externals), dtype=np.complex128)
        internal_vector_length = int(
            (len(perturbed_network_matrix) - 2 * num_externals) / 2
        )
        internal_scattering_map = self.network.internal_scattering_map
        external_scattering_map = self.network.external_scattering_map

        for q in range(num_externals):
            for p in range(num_externals):

                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.network.internal_links:
                    length = link.length

                    n = link.n(k0)
                    Dn = link.Dn(k0)

                    # Get the field distribution associated with q illumination
                    q_vector_unperturbed = unperturbed_outgoing_vectors[q]
                    q_vector_perturbed = perturbed_outgoing_vectors[q]
                    q_o_unperturbed = q_vector_unperturbed[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    q_i_unperturbed = q_vector_unperturbed[
                        2 * num_externals + internal_vector_length :
                    ]
                    q_o_perturbed = q_vector_perturbed[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    q_i_perturbed = q_vector_perturbed[
                        2 * num_externals + internal_vector_length :
                    ]

                    # Get the field distribution associated with p illumination
                    p_vector_unperturbed = unperturbed_outgoing_vectors[p]
                    p_vector_perturbed = perturbed_outgoing_vectors[p]
                    p_o_unperturbed = p_vector_unperturbed[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    p_i_unperturbed = p_vector_unperturbed[
                        2 * num_externals + internal_vector_length :
                    ]
                    p_o_perturbed = p_vector_perturbed[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    p_i_perturbed = p_vector_perturbed[
                        2 * num_externals + internal_vector_length :
                    ]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = internal_scattering_map[key]

                    # Work out derivatives numerically
                    I_mp_before = p_i_unperturbed[index]
                    I_mq_before = q_i_unperturbed[index]
                    O_mp_before = p_o_unperturbed[index]
                    O_mq_before = q_o_unperturbed[index]
                    I_mp_after = p_i_perturbed[index]
                    I_mq_after = q_i_perturbed[index]
                    O_mp_after = p_o_perturbed[index]
                    O_mq_after = q_o_perturbed[index]
                    diff_I_mp = I_mp_after - I_mp_before
                    diff_I_mq = I_mq_after - I_mq_before
                    diff_O_mp = O_mp_after - O_mp_before
                    diff_O_mq = O_mq_after - O_mq_before
                    d_I_mp = diff_I_mp / self.status.perturbation_value
                    d_I_mq = diff_I_mq / self.status.perturbation_value
                    d_O_mp = diff_O_mp / self.status.perturbation_value
                    d_O_mq = diff_O_mq / self.status.perturbation_value

                    partial_sum += 1j * d_O_mp * np.conj(O_mq_before) * (
                        1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length)
                    ) + 1j * d_I_mp * np.conj(I_mq_before) * (
                        np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0
                    )

                # Next loop over external links
                for link in self.network.external_links:
                    length = link.length

                    n = link.n(k0)
                    Dn = link.Dn(k0)

                    # Get the field distribution associated with q illumination
                    q_vector_unperturbed = unperturbed_outgoing_vectors[q]
                    q_vector_perturbed = perturbed_outgoing_vectors[q]
                    q_o_unperturbed = q_vector_unperturbed[0:num_externals]
                    q_o_perturbed = q_vector_perturbed[0:num_externals]
                    q_i_unperturbed = q_vector_unperturbed[
                        num_externals : 2 * num_externals
                    ]
                    q_i_perturbed = q_vector_perturbed[
                        num_externals : 2 * num_externals
                    ]

                    # Get the field distribution associated with p illumination
                    p_vector_unperturbed = unperturbed_outgoing_vectors[p]
                    p_vector_perturbed = perturbed_outgoing_vectors[p]
                    p_o_unperturbed = p_vector_unperturbed[0:num_externals]
                    p_o_perturbed = p_vector_perturbed[0:num_externals]
                    p_i_unperturbed = p_vector_unperturbed[
                        num_externals : 2 * num_externals
                    ]
                    p_i_perturbed = p_vector_perturbed[
                        num_externals : 2 * num_externals
                    ]

                    # Find the fields in the link
                    # Note: node_two is always the external node
                    external_node_index = link.sorted_connected_nodes[1]
                    key = f"{external_node_index}"
                    index = external_scattering_map[key]

                    # Work out derivatives numerically
                    I_mp_before = p_i_unperturbed[index]
                    I_mq_before = q_i_unperturbed[index]
                    O_mp_before = p_o_unperturbed[index]
                    O_mq_before = q_o_unperturbed[index]
                    I_mp_after = p_i_perturbed[index]
                    I_mq_after = q_i_perturbed[index]
                    O_mp_after = p_o_perturbed[index]
                    O_mq_after = q_o_perturbed[index]
                    diff_I_mp = I_mp_after - I_mp_before
                    diff_I_mq = I_mq_after - I_mq_before
                    diff_O_mp = O_mp_after - O_mp_before
                    diff_O_mq = O_mq_after - O_mq_before
                    d_I_mp = diff_I_mp / self.status.perturbation_value
                    d_I_mq = diff_I_mq / self.status.perturbation_value
                    d_O_mp = diff_O_mp / self.status.perturbation_value
                    d_O_mq = diff_O_mq / self.status.perturbation_value

                    partial_sum += 1j * d_I_mp * np.conj(I_mq_before) * (
                        1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length)
                    ) + 1j * d_O_mp * np.conj(O_mq_before) * (
                        np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0
                    )

                U_3[q, p] = partial_sum

        return U_3
