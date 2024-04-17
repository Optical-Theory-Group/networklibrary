from typing import Any, Callable
from tqdm.notebook import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from complex_network.scattering_matrices import scattering_matrix
from complex_network.components.link import Link
from complex_network.components.node import Node
from complex_network.materials.material import Material


class Network:
    def __init__(
        self,
        nodes: dict[int, Node],
        links: dict[int, Link],
        material: Material,
        data: dict[str, Any] | None = None,
    ) -> None:
        self.reset_values(data)
        self.node_dict = nodes
        self.link_dict = links
        self.material = material
        self.reset_fields()
        self.set_matrix_calc_utils()

    def set_matrix_calc_utils(self):
        """Pre-calculate some useful quantities used in calculating the network
        scattering matrces"""

        internal_vector_length = 0
        for node in self.internal_nodes:
            internal_vector_length += node.degree
        self.internal_vector_length = internal_vector_length

        (
            internal_scattering_map,
            internal_scattering_slices,
            external_scattering_map,
        ) = self._get_network_matrix_maps()
        self.internal_scattering_map = internal_scattering_map
        self.internal_scattering_slices = internal_scattering_slices
        self.external_scattering_map = external_scattering_map

    # -------------------------------------------------------------------------
    # Basic network properties
    # -------------------------------------------------------------------------

    @property
    def n(self) -> Callable[..., float]:
        """Refractive index function"""
        return self.material.n

    @property
    def dn(self) -> Callable[..., float]:
        """Derivative of refractive index function"""
        return self.material.dn

    @property
    def nodes(self):
        """List of all nodes"""
        return list(self.node_dict.values())

    @property
    def num_nodes(self) -> int:
        """Number of nodes in the network"""
        return len(self.nodes)

    @property
    def external_nodes(self):
        """List of external nodes"""
        return [node for node in self.nodes if node.node_type == "external"]

    @property
    def num_external_nodes(self):
        """Number of external nodes"""
        return len(self.external_nodes)

    @property
    def external_vector_length(self):
        """Equivalent to number of external nodes"""
        return self.num_external_nodes

    @property
    def external_node_indices(self):
        return [node.index for node in self.external_nodes]

    @property
    def internal_nodes(self):
        return [node for node in self.nodes if node.node_type == "internal"]

    @property
    def num_internal_nodes(self):
        return len(self.internal_nodes)

    @property
    def links(self):
        return list(self.link_dict.values())

    @property
    def num_links(self):
        return len(list(self.links))

    @property
    def external_links(self):
        return [link for link in self.links if link.link_type == "external"]

    @property
    def num_external_links(self):
        return list(self.external_links)

    @property
    def internal_links(self):
        return [link for link in self.links if link.link_type == "internal"]

    @property
    def num_internal_links(self):
        return len(list(self.internal_links))

    @property
    def connections(self):
        """Alias for links"""
        return self.links

    # -------------------------------------------------------------------------
    # Network matrix properties
    # -------------------------------------------------------------------------

    @property
    def S_mat(self):
        """The scattering matrix for the external ports"""
        if hasattr(self, "_S_mat") and self._S_mat is not None:
            return self._S_mat
        else:
            return self.get_S_matrix_direct()

    @S_mat.setter
    def S_mat(self, value: np.ndarray) -> None:
        self._S_mat = value

    @property
    def network_step_matrix(self):
        """The network matrix for one step of iteration"""
        if (
            hasattr(self, "_network_step_matrix")
            and self._network_step_matrix is not None
        ):
            return self._network_step_matrix
        else:
            return self.get_network_step_matrix()

    @property
    def network_matrix(self):
        """The full network matrix"""
        if (
            hasattr(self, "_network_matrix")
            and self._network_matrix is not None
        ):
            return self._network_matrix
        else:
            return self.get_network_matrix()

    # -------------------------------------------------------------------------
    # Basic utility functions
    # -------------------------------------------------------------------------

    def get_node(self, index: int | str) -> Node:
        """Returns the node with the specified index identifying number"""
        return self.node_dict[str(index)]

    def get_link(self, index: int | str) -> Link:
        """Returns the link with the specified index identifying number"""
        return self.link_dict[str(index)]

    def reset_values(self, data: dict[str, Any] | None = None) -> None:
        """Reset values of network to defaults or those in provided data"""
        default_values = self.get_default_values()
        if data is not None:
            default_values.update(data)
        for key, value in default_values.items():
            setattr(self, key, value)

    def reset_fields(self) -> None:
        """Reset the values of the incident and outgoing fields to be zero"""
        # Reset all node and link values
        for node in self.nodes:
            node.reset_fields()
        for link in self.links:
            link.reset_fields()

        # Set up keys
        for node in self.external_nodes:
            self.inwave[str(node.index)] = 0 + 0j
            self.outwave[str(node.index)] = 0 + 0j

        # Set up np arrays
        self.inwave_np = np.zeros(len(self.inwave.keys()), dtype=np.complex128)
        self.outwave_np = np.zeros(
            len(self.inwave.keys()), dtype=np.complex128
        )

    @staticmethod
    def get_default_values() -> dict[str, Any]:
        """Default values for the network"""
        default_values: dict[str, Any] = {
            "node_dict": {},
            "link_dict": {},
            "inwave": {},
            "outwave": {},
            "inwave_np": np.zeros(0, dtype=np.complex128),
            "outwave_np": np.zeros(0, dtype=np.complex128),
            "S_mat": np.zeros(0, dtype=np.complex128),
            "iS_mat": np.zeros(0, dtype=np.complex128),
        }
        return default_values

    def update_links(
        self,
        k0: float | complex,
    ) -> None:
        """Update k0, n and the scattering matrices for all links in the
        network"""
        for link in self.links:
            link.k0 = k0
            new_n = self.n(k0)
            link.n = new_n
            link.update_S_matrices()

    # -------------------------------------------------------------------------
    #  Direct scattering methods
    # -------------------------------------------------------------------------

    def _get_network_matrix_maps(
        self,
    ) -> tuple[dict[str, int], dict[str, slice], dict[str, int]]:
        internal_scattering_slices = {}
        internal_scattering_map = {}
        external_scattering_map = {}
        i = 0

        for node in self.internal_nodes:
            # Update the map. Loop through connected nodes and work out the
            # indices
            start = i
            node_index = node.index
            for new_index in node.sorted_connected_nodes:
                internal_scattering_map[f"{node_index},{new_index}"] = i
                i += 1
            end = i
            internal_scattering_slices[f"{node_index}"] = slice(start, end)

        i = 0
        for node in self.external_nodes:
            external_scattering_map[f"{node.index}"] = i
            i += 1

        return (
            internal_scattering_map,
            internal_scattering_slices,
            external_scattering_map,
        )

    def scatter_direct(
        self,
        incident_field: np.ndarray,
        direction: str = "forward",
    ) -> None:
        """Scatter the incident field through the network using the
        network matrix"""

        # Set up the matrix product
        network_matrix = self.network_matrix
        num_externals = self.num_external_nodes
        incident_vector = np.zeros((len(network_matrix)), dtype=np.complex128)
        incident_vector[num_externals : 2 * num_externals] = incident_field
        outgoing_vector = network_matrix @ incident_vector

        # Reset fields throughout the network and set incident field
        self.reset_fields()
        self.set_network_fields(outgoing_vector)

    def set_network_fields(self, vector: np.ndarray) -> None:
        """Set the values of the fields throughout the network using a network
        vector"""
        external_vector_length = self.external_vector_length
        internal_vector_length = self.internal_vector_length

        outgoing_external = vector[0:external_vector_length]
        incoming_external = vector[
            external_vector_length : 2 * external_vector_length
        ]
        outgoing_internal = vector[
            2 * external_vector_length : 2 * external_vector_length
            + internal_vector_length
        ]
        incoming_internal = vector[
            2 * external_vector_length + internal_vector_length :
        ]

        self.set_incident_field(incoming_external)

        count = 0
        for node in self.external_nodes:
            # Set outgoing external values
            value = outgoing_external[count]
            node_index = node.index
            connected_link_index = node.sorted_connected_links[0]
            connected_link = self.get_link(connected_link_index)

            node.outwave["-1"] = value
            node.outwave_np[0] = value
            node.inwave[str(connected_link)] = value
            node.inwave[1] = value

            connected_link.outwave[str(node_index)] = value
            connected_link.outwave_np[1] = value

            # Set incoming external values
            value = incoming_external[count]
            node_index = node.index
            connected_link_index = node.sorted_connected_links[0]
            connected_link = self.get_link(connected_link_index)

            node.inwave["-1"] = value
            node.inwave_np[0] = value
            node.outwave[str(connected_link)] = value
            node.outwave[1] = value

            connected_link.inwave[str(node_index)] = value
            connected_link.inwave_np[1] = value

            count += 1

        # Set internal node values
        count = 0
        for node in self.internal_nodes:
            for i, connected_index in enumerate(node.sorted_connected_nodes):
                incoming_value = incoming_internal[count]
                node.inwave[str(connected_index)] = incoming_value
                node.inwave_np[i] = incoming_value

                outgoing_value = outgoing_internal[count]
                node.outwave[str(connected_index)] = outgoing_value
                node.outwave_np[i] = outgoing_value

                count += 1

        # Set internal link values
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            node_one = self.get_node(node_one_index)
            node_two = self.get_node(node_two_index)

            # Set link fields
            link.inwave[str(node_one_index)] = node_one.outwave[
                str(node_two_index)
            ]
            link.inwave_np[0] = node_one.outwave[str(node_two_index)]
            link.inwave[str(node_two_index)] = node_two.outwave[
                str(node_one_index)
            ]
            link.inwave_np[1] = node_two.outwave[str(node_one_index)]

            # Outwaves
            link.outwave[str(node_one_index)] = node_one.inwave[
                str(node_two_index)
            ]
            link.outwave_np[0] = node_one.inwave[str(node_two_index)]
            link.outwave[str(node_two_index)] = node_two.inwave[
                str(node_one_index)
            ]
            link.outwave_np[1] = node_two.inwave[str(node_one_index)]

        # Remaining external links values
        for link in self.external_links:
            external_index, node_index = link.node_indices
            node = self.get_node(node_index)

            # Set link fields
            link.inwave[str(node_index)] = node.outwave[str(external_index)]
            link.inwave_np[1] = node.outwave[str(external_index)]

            link.outwave[str(node_index)] = node.inwave[str(external_index)]
            link.outwave_np[1] = node.inwave[str(external_index)]

        self.update_outgoing_fields()

    def get_S_matrix_direct(self, k0: float | complex) -> np.ndarray:
        """Calculate the scattering matrix by taking the appropriate block from
        the network matrix"""
        # Update network with given wave parameters
        self.update_links(k0)

        network_matrix = self.get_network_matrix(k0)
        num_external_nodes = self.num_external_nodes
        S_external = network_matrix[
            0:num_external_nodes, num_external_nodes : 2 * num_external_nodes
        ]
        self._S_mat = S_external
        return S_external

    def get_network_matrix(self, k0: float | complex) -> np.ndarray:
        """Get the 'infinite' order network matrix"""
        step_matrix = self.get_network_step_matrix(k0)
        lam, v = np.linalg.eig(step_matrix)
        modified_lam = np.where(np.isclose(lam, 1.0 + 0.0 * 1j), lam, 0.0)
        rebuilt = v @ np.diag(modified_lam) @ np.linalg.inv(v)
        self._network_matrix = rebuilt
        return rebuilt

    def get_network_step_matrix(self, k0: float | complex) -> np.ndarray:
        """The network step matrix satisfies

        (O_e)       (0 0         |P_ei       0)(O_e)
        (I_e)       (0 1         |0          0)(I_e)
        (---)   =   (-------------------------)(---)
        (O_i)       (0 S_ii*P_ie | S_ii*P_ii 0)(O_i)
        (I_i)_n+1   (0 P_ie      | P_ii      0)(I_i)_n
        """
        self.update_links(k0)

        # Get the internal S
        internal_S = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for node in self.internal_nodes:
            node_index = node.index
            node_S_mat = node.S_mat
            new_slice = self.internal_scattering_slices[str(node_index)]
            internal_S[new_slice, new_slice] = node_S_mat

        # Get internal P
        internal_P = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            # Wave that is going into node_one
            row = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = self.internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor
            # Wave propagating the other way
            internal_P[col, row] = phase_factor

        # Get external P
        external_P = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.external_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            external_P[row, col] = phase_factor

        # Build up network matrix
        # First define all the zero matrices to keep things simpler
        z_ee = np.zeros(
            (self.external_vector_length, self.external_vector_length),
            dtype=np.complex128,
        )
        z_long = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        z_tall = z_long.T
        z_ii = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        identity = np.identity(
            self.external_vector_length, dtype=np.complex128
        )

        network_step_matrix = np.block(
            [
                [z_ee, z_ee, external_P, z_long],
                [z_ee, identity, z_long, z_long],
                [
                    z_tall,
                    internal_S @ external_P.T,
                    internal_S @ internal_P,
                    z_ii,
                ],
                [z_tall, external_P.T, internal_P, z_ii],
            ]
        )
        self._network_step_matrix = network_step_matrix
        return network_step_matrix

    def get_S_ee(self, k0: float | complex) -> np.ndarray:
        """Get the external scattering matrix from the inverse formula"""
        self.update_links(k0)

        P_ei = self.get_P_ei()
        P_ie = P_ei.T
        S_ii = self.get_S_ii()
        P_ii = self.get_P_ii()

        # Bracketed part to be inverted
        bracket = np.identity(len(S_ii), dtype=np.complex128) - S_ii @ P_ii
        inv = np.linalg.inv(bracket)

        S_ee = P_ei @ inv @ S_ii @ P_ie
        return S_ee

    def get_S_ee_inv(self, k0: float | complex) -> np.ndarray:
        """Get the external inverse scattering matrix from the inverse
        formula"""
        self.update_links(k0)

        P_ei_inv = self.get_P_ei_inv()
        P_ie_inv = P_ei_inv.T
        S_ii_inv = self.get_S_ii_inv()
        P_ii_inv = self.get_P_ii_inv()

        # Bracketed part to be inverted
        bracket = (
            np.identity(len(S_ii_inv), dtype=np.complex128)
            - P_ii_inv @ S_ii_inv
        )
        inv = np.linalg.inv(bracket)

        S_ee_inv = P_ei_inv @ S_ii_inv @ inv @ P_ie_inv
        return S_ee_inv

    def get_S_ii(self) -> np.ndarray:
        """Return the S_ii matrix formed from node scattering matrices"""

        internal_S = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for node in self.internal_nodes:
            node_index = node.index
            node_S_mat = node.S_mat
            new_slice = self.internal_scattering_slices[str(node_index)]
            internal_S[new_slice, new_slice] = node_S_mat
        return internal_S

    def get_S_ii_inv(self) -> np.ndarray:
        """Return the inverse of the S_ii matrix formed from node scattering
        matrices"""
        return np.linalg.inv(self.get_S_ii())

    def get_P_ii(self) -> np.ndarray:
        """Return P matrix calculated from internal network links"""
        # Get internal P
        internal_P = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            # Wave that is going into node_one
            row = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = self.internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor
            # Wave propagating the other way
            internal_P[col, row] = phase_factor
        return internal_P

    def get_P_ii_inv(self) -> np.ndarray:
        """Return P matrix calculated from internal network links"""

        # Get internal P
        internal_P = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            # Wave that is going into node_one
            row = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = self.internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = 1 / phase_factor
            # Wave propagating the other way
            internal_P[col, row] = 1 / phase_factor
        return internal_P

    def get_P_ei(self) -> np.ndarray:
        """Get the matrix that deals with propagation in external links"""

        external_P = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.external_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            external_P[row, col] = phase_factor
        return external_P

    def get_P_ei_inv(self) -> np.ndarray:
        """Get the matrix that deals with propagation in external links"""

        external_P = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.external_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            external_P[row, col] = 1 / phase_factor
        return external_P

    def get_P_ie(self) -> np.ndarray:
        return self.get_P_ei().T

    def get_SP(self, k0: float | complex) -> np.ndarray:
        """Get the prodct S_ii P_ii"""
        self.update_links(k0)

        S_ii = self.get_S_ii()
        P_ii = self.get_P_ii()
        return S_ii @ P_ii

    def get_inv_factor(self, k0: float | complex) -> complex:
        """Calculate I - S_ii P_ii"""
        self.update_links(k0)

        S_ii = self.get_S_ii()
        P_ii = self.get_P_ii()
        inv_factor = np.identity(len(S_ii), dtype=np.complex128) - S_ii @ P_ii
        return inv_factor

    def get_inv_factor_det(self, k0: float | complex) -> complex:
        """Calculate det(I - S_ii P_ii)"""
        self.update_links(k0)

        S_ii = self.get_S_ii()
        P_ii = self.get_P_ii()
        inv_factor = np.identity(len(S_ii), dtype=np.complex128) - S_ii @ P_ii
        return np.linalg.det(inv_factor)

    def get_wigner_smith_k0(self, k0: float | complex) -> np.ndarray:
        """Calculate directly the Wigner-Smith operator
        Q = -i * S^-1 * dS/dk0"""

        S_ee_inv = self.get_S_ee_inv(k0)
        dS_ee_dk0 = self.get_dS_ee_dk0()

        ws = -1j * S_ee_inv @ dS_ee_dk0
        return ws

    def get_wigner_smith_t(
        self,
        k0: float | complex,
        perturbed_node_index: int,
        perturbed_angle_index: int,
    ) -> np.ndarray:
        """Calculate analytically the Wigner-Smith operator
        Q = -i * S^-1 * dS/dt"""

        S_ee_inv = self.get_S_ee_inv(k0)
        dS_ee_dt = self.get_dS_ee_dt(
            perturbed_node_index, perturbed_angle_index
        )

        ws = -1j * S_ee_inv @ dS_ee_dt
        return ws

    # -------------------------------------------------------------------------
    # Volume integral methods
    # -------------------------------------------------------------------------

    def get_U_0(self, n, k0) -> np.ndarray:
        """Calculate the U_0 matrix (see theory notes)"""

        network_matrix = self.get_network_matrix(k0)

        # Get the scattered fields for each incident field
        outgoing_vectors = []
        num_externals = self.num_external_nodes
        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            incident_vector = np.zeros(
                (len(network_matrix)), dtype=np.complex128
            )
            incident_vector[num_externals : 2 * num_externals] = incident_field
            outgoing_vector = network_matrix @ incident_vector
            outgoing_vectors.append(outgoing_vector)

        U_0 = np.zeros((num_externals, num_externals), dtype=np.complex128)
        internal_vector_length = int(
            (len(network_matrix) - 2 * num_externals) / 2
        )

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.internal_links:
                    length = link.length

                    # Get the field distribution associated with q illumination
                    q_vector = outgoing_vectors[q]
                    q_o = q_vector[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    q_i = q_vector[
                        2 * num_externals + internal_vector_length :
                    ]

                    # Get the field distribution associated with p illumination
                    p_vector = outgoing_vectors[p]
                    p_o = p_vector[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    p_i = p_vector[
                        2 * num_externals + internal_vector_length :
                    ]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = self.internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    partial_sum += (
                        O_mp
                        * np.conj(O_mq)
                        * (1.0 + 0.0j - np.exp(-2 * n * np.imag(k0) * length))
                    )

                    partial_sum += (
                        I_mp
                        * np.conj(I_mq)
                        * (np.exp(2 * n * np.imag(k0) * length) - 1.0 + 0.0j)
                    )

                # Next loop over external links
                for link in self.external_links:
                    length = link.length

                    # Get the field distribution associated with q illumination
                    q_vector = outgoing_vectors[q]
                    q_o = q_vector[0:num_externals]
                    q_i = q_vector[num_externals : 2 * num_externals]

                    # Get the field distribution associated with p illumination
                    p_vector = outgoing_vectors[p]
                    p_o = p_vector[0:num_externals]
                    p_i = p_vector[num_externals : 2 * num_externals]

                    # Find the fields in the link
                    # Note: node_two is always the external node
                    external_node_index = link.sorted_connected_nodes[1]
                    key = f"{external_node_index}"
                    index = self.external_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    partial_sum += (
                        I_mp
                        * np.conj(I_mq)
                        * (1.0 - np.exp(-2 * n * np.imag(k0) * length))
                    )

                    partial_sum += (
                        O_mp
                        * np.conj(O_mq)
                        * (np.exp(2 * n * np.imag(k0) * length) - 1.0)
                    )

                U_0[q, p] = partial_sum

        return U_0

    def get_U_1(self, n, k0) -> np.ndarray:
        """Calculate the U_1 matrix (see theory notes)"""

        network_matrix = self.get_network_matrix(k0)

        # Get the scattered fields for each incident field
        outgoing_vectors = []
        num_externals = self.num_external_nodes
        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            incident_vector = np.zeros(
                (len(network_matrix)), dtype=np.complex128
            )
            incident_vector[num_externals : 2 * num_externals] = incident_field
            outgoing_vector = network_matrix @ incident_vector
            outgoing_vectors.append(outgoing_vector)

        U_1 = np.zeros((num_externals, num_externals), dtype=np.complex128)
        internal_vector_length = int(
            (len(network_matrix) - 2 * num_externals) / 2
        )

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.internal_links:
                    length = link.length

                    # Get the field distribution associated with q illumination
                    q_vector = outgoing_vectors[q]
                    q_o = q_vector[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    q_i = q_vector[
                        2 * num_externals + internal_vector_length :
                    ]

                    # Get the field distribution associated with p illumination
                    p_vector = outgoing_vectors[p]
                    p_o = p_vector[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    p_i = p_vector[
                        2 * num_externals + internal_vector_length :
                    ]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = self.internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    partial_sum += (
                        O_mp
                        * np.conj(O_mq)
                        * (np.exp(-2 * n * np.real(k0) * length))
                        + I_mp
                        * np.conj(I_mq)
                        * (np.exp(2 * n * np.real(k0) * length))
                    ) * length

                # Next loop over external links
                for link in self.external_links:
                    length = link.length

                    # Get the field distribution associated with q illumination
                    q_vector = outgoing_vectors[q]
                    q_o = q_vector[0:num_externals]
                    q_i = q_vector[num_externals : 2 * num_externals]

                    # Get the field distribution associated with p illumination
                    p_vector = outgoing_vectors[p]
                    p_o = p_vector[0:num_externals]
                    p_i = p_vector[num_externals : 2 * num_externals]

                    # Find the fields in the link
                    # Note: node_two is always the external node
                    external_node_index = link.sorted_connected_nodes[1]
                    key = f"{external_node_index}"
                    index = self.external_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    partial_sum += (
                        I_mp
                        * np.conj(I_mq)
                        * (np.exp(-2 * n * np.real(k0) * length))
                        + O_mp
                        * np.conj(O_mq)
                        * (np.exp(2 * n * np.real(k0) * length))
                    ) * length

                U_1[q, p] = partial_sum

        return U_1

    def get_U_2(self, n, k0) -> np.ndarray:
        """Calculate the U_2 matrix (see theory notes)"""

        network_matrix = self.get_network_matrix(k0)

        # Get the scattered fields for each incident field
        outgoing_vectors = []
        num_externals = self.num_external_nodes
        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            incident_vector = np.zeros(
                (len(network_matrix)), dtype=np.complex128
            )
            incident_vector[num_externals : 2 * num_externals] = incident_field
            outgoing_vector = network_matrix @ incident_vector
            outgoing_vectors.append(outgoing_vector)

        U_2 = np.zeros((num_externals, num_externals), dtype=np.complex128)
        internal_vector_length = int(
            (len(network_matrix) - 2 * num_externals) / 2
        )

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.internal_links:
                    length = link.length

                    # Get the field distribution associated with q illumination
                    q_vector = outgoing_vectors[q]
                    q_o = q_vector[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    q_i = q_vector[
                        2 * num_externals + internal_vector_length :
                    ]

                    # Get the field distribution associated with p illumination
                    p_vector = outgoing_vectors[p]
                    p_o = p_vector[
                        2 * num_externals : 2 * num_externals
                        + internal_vector_length
                    ]
                    p_i = p_vector[
                        2 * num_externals + internal_vector_length :
                    ]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = self.internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    partial_sum += (
                        O_mp
                        * np.conj(I_mq)
                        * (np.exp(2 * n * np.real(k0) * length) - 1.0)
                    )

                    partial_sum += (
                        I_mp
                        * np.conj(O_mq)
                        * (1.0 - np.exp(-2 * n * np.real(k0) * length))
                    )

                # Next loop over external links
                for link in self.external_links:
                    length = link.length

                    # Get the field distribution associated with q illumination
                    q_vector = outgoing_vectors[q]
                    q_o = q_vector[0:num_externals]
                    q_i = q_vector[num_externals : 2 * num_externals]

                    # Get the field distribution associated with p illumination
                    p_vector = outgoing_vectors[p]
                    p_o = p_vector[0:num_externals]
                    p_i = p_vector[num_externals : 2 * num_externals]

                    # Find the fields in the link
                    # Note: node_two is always the external node
                    external_node_index = link.sorted_connected_nodes[1]
                    key = f"{external_node_index}"
                    index = self.external_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    partial_sum += (
                        I_mp
                        * np.conj(O_mq)
                        * (np.exp(2 * n * np.real(k0) * length) - 1.0)
                    )

                    partial_sum += (
                        O_mp
                        * np.conj(I_mq)
                        * (1.0 - np.exp(-2 * n * np.real(k0) * length))
                    )

                U_2[q, p] = partial_sum

        return U_2

    # -------------------------------------------------------------------------
    #  Methods for altering/perturbing the network
    # -------------------------------------------------------------------------

    def perturb_node_eigenvalue(
        self, node_index: int, eigenvalue_index: int, factor: complex
    ) -> None:
        """Multiply the specified eigenvalue by the factor variable"""
        node = self.get_node(node_index)
        S_mat = node.S_mat

        # Multiply the appropriate eigenvalue by the given factor
        lam, w = np.linalg.eig(S_mat)
        lam[eigenvalue_index] = lam[eigenvalue_index] * factor
        new_S = w @ np.diag(lam) @ np.linalg.inv(w)
        node.S_mat = new_S
        node.iS_mat = np.linalg.inv(new_S)

        # Leave perturbation info here too
        node.is_perturbed = True
        node.perturbation_params = {
            "eigenvalue_index": eigenvalue_index,
            "factor": factor,
        }

    def get_dS_ee_dk0(self) -> np.ndarray:
        """Get the derivative of S_ee with respect to k0"""

        P_ei = self.get_P_ei()
        P_ie = self.get_P_ie()
        P_ii = self.get_P_ii()
        S_ii = self.get_S_ii()

        dP_ei_dk0 = self.get_dP_ei_dk0()
        dP_ie_dk0 = self.get_dP_ie_dk0()
        dP_ii_dk0 = self.get_dP_ii_dk0()
        dS_ii_dk0 = np.zeros(S_ii.shape, dtype=np.complex128)

        SP = S_ii @ P_ii
        inv = np.linalg.inv(np.identity(len(SP)) - SP)
        dinv_dk0 = inv @ (S_ii @ dP_ii_dk0 + dS_ii_dk0 @ P_ii) @ inv

        term_one = dP_ei_dk0 @ inv @ S_ii @ P_ie
        term_two = P_ei @ dinv_dk0 @ S_ii @ P_ie
        term_three = P_ei @ inv @ dS_ii_dk0 @ P_ie
        term_four = P_ei @ inv @ S_ii @ dP_ie_dk0

        dS_ee_dk0 = term_one + term_two + term_three + term_four
        return dS_ee_dk0

    def get_dS_ee_dt(
        self,
        perturbed_node_index: int,
        perturbed_angle_index: int,
    ) -> np.ndarray:
        """Get the derivative of S_ee with respect to eigenphase t
        (eigenvalue is e^it)"""

        P_ei = self.get_P_ei()
        P_ie = self.get_P_ie()
        P_ii = self.get_P_ii()
        S_ii = self.get_S_ii()

        dP_ei_dt = np.zeros(P_ei.shape, dtype=np.complex128)
        dP_ie_dt = np.zeros(P_ie.shape, dtype=np.complex128)
        dP_ii_dt = np.zeros(P_ii.shape, dtype=np.complex128)
        dS_ii_dt = self.get_dS_ii_dt(
            perturbed_node_index, perturbed_angle_index
        )

        SP = S_ii @ P_ii
        inv = np.linalg.inv(np.identity(len(SP)) - SP)
        dinv_dt = inv @ (S_ii @ dP_ii_dt + dS_ii_dt @ P_ii) @ inv

        term_one = dP_ei_dt @ inv @ S_ii @ P_ie
        term_two = P_ei @ dinv_dt @ S_ii @ P_ie
        term_three = P_ei @ inv @ dS_ii_dt @ P_ie
        term_four = P_ei @ inv @ S_ii @ dP_ie_dt

        dS_ee_dt = term_one + term_two + term_three + term_four
        return dS_ee_dt

    def get_S_ii_dS_ii(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the S_ii matrix and its derivative"""

        internal_S = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )

        internal_dS = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )

        for node in self.internal_nodes:
            node_index = node.index
            node_S_mat = node.S_mat
            new_slice = self.internal_scattering_slices[str(node_index)]
            internal_S[new_slice, new_slice] = node_S_mat

            if node.is_perturbed:
                internal_dS[new_slice, new_slice] = node.dS_mat

        return internal_S, internal_dS

    def get_P_ii_dP_ii(
        self, n: complex, k0: complex
    ) -> tuple[np.ndarray, np.ndarray]:
        self.update_links(k0)

        internal_P = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )

        internal_dP = np.zeros(
            (self.internal_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )

        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            # Wave that is going into node_one
            row = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = self.internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor
            internal_dP[row, col] = phase_factor * 1j * n * link.length
            # Wave propagating the other way
            internal_P[col, row] = phase_factor
            internal_dP[col, row] = phase_factor * 1j * n * link.length

        return internal_P, internal_dP

    def set_node_S_matrix(
        self,
        node_index: int,
        S_mat_type: str,
        S_mat_params: dict[str, Any] | None = None,
    ) -> None:
        """Replace the node scattering matrix with a new one as defined by
        the arguments"""
        if S_mat_params is None:
            S_mat_params = {}

        node = self.get_node(node_index)
        size, _ = node.S_mat.shape
        node.S_mat = scattering_matrix.get_S_mat(
            S_mat_type, size, S_mat_params
        )
        node.iS_mat = np.linalg.inv(node.S_mat)

    def get_dS_ii_dt(
        self,
        perturbed_node_index: int,
        perturbed_angle_index: int,
        internal_vector_length: int | None = None,
        internal_scattering_slices: slice | None = None,
    ) -> np.ndarray:
        """Return the S_ii matrix formed from node scattering matrices"""

        # Calculate the size of the matrix if it isn't given
        if internal_vector_length is None:
            internal_vector_length = 0
            for node in self.internal_nodes:
                internal_vector_length += node.degree

        # Get the slices array if not provided
        if internal_scattering_slices is None:
            _, internal_scattering_slices, _ = self._get_network_matrix_maps()

        internal_S = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        for node in self.internal_nodes:
            node_index = node.index

            # Only perturbed node has non-zero entry in dS
            if node_index != perturbed_node_index:
                continue

            # We are now at the perturbed node
            node_S_mat = node.S_mat
            lam, w = np.linalg.eig(node_S_mat)
            for angle_num, phase_term in enumerate(lam):
                if angle_num == perturbed_angle_index:
                    lam[angle_num] = 1j * phase_term
                else:
                    lam[angle_num] = 0.0

            node_S_mat = w @ np.diag(lam) @ np.linalg.inv(w)

            new_slice = internal_scattering_slices[str(node_index)]
            internal_S[new_slice, new_slice] = node_S_mat
        return internal_S

    def get_dP_ii_dk0(
        self,
        internal_vector_length: int | None = None,
        internal_scattering_map: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Return derivative of P_ii with respect to k0"""
        # Calculate the size of the matrix if it isn't given
        if internal_vector_length is None:
            internal_vector_length = 0
            for node in self.internal_nodes:
                internal_vector_length += node.degree

        # Get the slices array if not provided
        if internal_scattering_map is None:
            internal_scattering_map, _, _ = self._get_network_matrix_maps()

        # Get internal P
        internal_P = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            # Wave that is going into node_one
            row = internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor * 1j * link.n * link.length
            # Wave propagating the other way
            internal_P[col, row] = phase_factor * 1j * link.n * link.length
        return internal_P

    def get_dP_ii_drek0(
        self,
        internal_vector_length: int | None = None,
        internal_scattering_map: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Return derivative of P_ii with respect to the real part of k0"""
        # Calculate the size of the matrix if it isn't given
        if internal_vector_length is None:
            internal_vector_length = 0
            for node in self.internal_nodes:
                internal_vector_length += node.degree

        # Get the slices array if not provided
        if internal_scattering_map is None:
            internal_scattering_map, _, _ = self._get_network_matrix_maps()

        # Get internal P
        internal_P = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            # Wave that is going into node_one
            row = internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor * 1j * link.n * link.length
            # Wave propagating the other way
            internal_P[col, row] = phase_factor * 1j * link.n * link.length
        return internal_P

    def get_dP_ii_dimk0(
        self,
        internal_vector_length: int | None = None,
        internal_scattering_map: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Return derivative of P_ii with respect to the imaginary part of k0"""
        # Calculate the size of the matrix if it isn't given
        if internal_vector_length is None:
            internal_vector_length = 0
            for node in self.internal_nodes:
                internal_vector_length += node.degree

        # Get the slices array if not provided
        if internal_scattering_map is None:
            internal_scattering_map, _, _ = self._get_network_matrix_maps()

        # Get internal P
        internal_P = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            # Wave that is going into node_one
            row = internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = phase_factor * -1 * link.n * link.length
            # Wave propagating the other way
            internal_P[col, row] = phase_factor * -1 * link.n * link.length
        return internal_P

    def get_dP_ei_dk0(
        self,
        internal_vector_length: int | None = None,
        internal_scattering_map: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Return derivative of P_ei with respect to k0"""

        external_P = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.external_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            external_P[row, col] = phase_factor * 1j * link.n * link.length
        return external_P

    def get_dP_ie_dk0(
        self,
        internal_vector_length: int | None = None,
        internal_scattering_map: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Return derivative of P_ie with respect to k0"""

        return self.get_dP_ei_dk0().T

    # -------------------------------------------------------------------------
    # Iterative scattering methods
    # -------------------------------------------------------------------------

    def get_S_matrix_iterative(
        self,
        k0: float | complex,
        direction: str = "forward",
        max_num_steps: int = 10**5,
        tolerance: float = 1e-5,
        verbose: bool = True,
    ) -> np.ndarray:
        """Calculate the network scattering matrix iteratively"""
        # Update network with given wave parameters
        self.update_links(k0)

        matrix = np.zeros(
            (self.num_external_nodes, self.num_external_nodes),
            dtype=np.complex128,
        )
        if verbose:
            node_pbar = tqdm(
                total=self.num_external_nodes, desc="external nodes"
            )

        # Loop over external nodes
        for i in range(self.num_external_nodes):
            incident_field = np.zeros(
                self.num_external_nodes, dtype=np.complex128
            )
            incident_field[i] = 1.0 + 0j

            print(f"Scattering {incident_field}")
            scattered_field = self.scatter_iterative(
                incident_field, direction, max_num_steps, tolerance, verbose
            )

            matrix[:, i] = scattered_field
            if verbose:
                node_pbar.update(1)

        # Store matrix in network
        if direction == "forward":
            self.S_mat = matrix
            self.iS_mat = np.linalg.inv(matrix)
        else:
            self.iS_mat = matrix
            self.S_mat = np.linalg.inv(matrix)

        return matrix

    def scatter_iterative(
        self,
        incident_field: np.ndarray,
        direction: str = "forward",
        max_num_steps: int = 10**5,
        tolerance: float = 1e-5,
        verbose: bool = False,
    ) -> np.ndarray:
        """Scatter the incident field through the network"""
        # Reset fields throughout the network and set incident field
        self.reset_fields()
        self.set_incident_field(incident_field, direction)

        # Scatter recursively
        has_converged = False

        if verbose:
            total_pbar = tqdm(total=max_num_steps, desc="Steps")
            conv_pbar = tqdm(total=1, desc="Convergence")

        for i in range(max_num_steps):
            before = np.copy(self.get_outgoing_fields(direction))
            self.scatter_step(direction)
            after = np.copy(self.get_outgoing_fields(direction))

            # Update progress bar
            if verbose:
                total_pbar.update(1)

            # Give the code a couple of simulations so that it actually reaches
            # the external
            if i <= 5:
                continue

            # Check for convergence
            diff = np.linalg.norm(after - before)
            if verbose:
                frac = min(np.log10(diff) / np.log10(tolerance), 1.0)
                conv_pbar.update(frac - conv_pbar.n)

            if diff <= tolerance:
                has_converged = True
                break

        # Close pbars
        if verbose:
            conv_pbar.close()
            total_pbar.close()

        # Throw warning if field has not converged
        if not has_converged:
            warnings.warn(
                "Max number of steps has been reached, but field "
                "has not converged! Consider increasing the maximum number of "
                "steps.",
                category=UserWarning,
            )

        return after

    def set_incident_field(
        self, incident_field: np.ndarray, direction: str = "forward"
    ) -> None:
        """Sets the incident field to the inwaves/outwaves"""
        # Check size
        if len(incident_field) != self.num_external_nodes:
            raise ValueError(
                f"Incident field has incorrect size. "
                f"It should be of size {self.num_external_nodes}."
            )

        # Set values to nodes and network dictionaries
        for i, external_node in enumerate(self.external_nodes):
            if direction == "forward":
                self.inwave[str(external_node.index)] = incident_field[i]
                external_node.inwave["-1"] = incident_field[i]
                external_node.inwave_np[0] = incident_field[i]
            elif direction == "backward":
                self.outwave[str(external_node.index)] = incident_field[i]
                external_node.outwave["-1"] = incident_field[i]
                external_node.outwave_np[0] = incident_field[i]

        # Set values to network
        if direction == "forward":
            self.inwave_np = incident_field
        if direction == "backward":
            self.outwave_np = incident_field

    def get_outgoing_fields(self, direction: str = "forward") -> np.ndarray:
        """Get the current outgoinf field on the basis of the given
        direction"""
        if direction == "forward":
            return self.outwave_np
        else:
            return self.inwave_np

    def update_outgoing_fields(self, direction: str = "forward") -> None:
        """Update the fields from the external nodes and put them into the network
        inwave/outwaves"""
        for i, node in enumerate(self.external_nodes):
            if direction == "forward":
                self.outwave[str(node.index)] = node.outwave["-1"]
                self.outwave_np[i] = node.outwave["-1"]
            if direction == "backward":
                self.inwave[str(node.index)] = node.inwave["-1"]
                self.inwave_np[i] = node.inwave["-1"]

    def scatter_step(self, direction: str = "forward") -> None:
        """Perform one step of scattering throughout the network.

        This involves scattering once at the nodes and once in the links."""

        # Scatter at nodes
        for node in self.nodes:
            node.update(direction)

        # Transfer fields to links
        self.nodes_to_links(direction)

        # Scatter in links.
        for link in self.links:
            link.update(direction)

        # Transfer fields to nodes
        self.links_to_nodes(direction)

        # Update network outgoing fields
        self.update_outgoing_fields(direction)

    def nodes_to_links(self, direction: str = "forward") -> None:
        """Transfer fields from nodes to links in given direction"""
        # Give outwaves of nodes to links
        for link in self.links:
            node_one_index, node_two_index = link.node_indices
            node_one_index = str(node_one_index)
            node_two_index = str(node_two_index)
            node_one = self.get_node(node_one_index)
            node_two = self.get_node(node_two_index)

            # Loop through link inwaves
            # Note, note inwave/outwaves are to OTHER NODES, NOT LINKS
            if direction == "forward":
                link.inwave[node_one_index] = node_one.outwave[node_two_index]
                link.inwave[node_two_index] = node_two.outwave[node_one_index]
                link.inwave_np[0] = link.inwave[node_one_index]
                link.inwave_np[1] = link.inwave[node_two_index]

            elif direction == "backward":
                link.outwave[node_one_index] = node_one.inwave[node_two_index]
                link.outwave[node_two_index] = node_two.inwave[node_one_index]
                link.outwave_np[0] = link.outwave[node_one_index]
                link.outwave_np[1] = link.outwave[node_two_index]

    def links_to_nodes(self, direction: str = "forward") -> None:
        """Transfer fields from links to nodes in given direction"""
        # Give outwaves of links to nodes
        for link in self.links:
            node_one_index, node_two_index = link.node_indices
            node_one_index = str(node_one_index)
            node_two_index = str(node_two_index)
            node_one = self.get_node(node_one_index)
            node_two = self.get_node(node_two_index)

            # Loop through link inwaves
            # Note, note inwave/outwaves are to OTHER NODES, NOT LINKS
            if direction == "forward":
                node_one.inwave[node_two_index] = link.outwave[node_one_index]
                node_two.inwave[node_one_index] = link.outwave[node_two_index]

                # Get the indices appropriate for the numpy arrays that
                # correspond to the correct wave
                np_one = node_one.sorted_connected_nodes.index(
                    int(node_two_index)
                )
                np_two = node_two.sorted_connected_nodes.index(
                    int(node_one_index)
                )
                node_one.inwave_np[np_one] = node_one.inwave[node_two_index]
                node_two.inwave_np[np_two] = node_two.inwave[node_one_index]

            elif direction == "backward":
                node_one.outwave[node_two_index] = link.inwave[node_one_index]
                node_two.outwave[node_one_index] = link.inwave[node_two_index]

                # Get the indices appropriate for the numpy arrays that
                # correspond to the correct wave
                np_one = node_one.sorted_connected_nodes.index(
                    int(node_two_index)
                )
                np_two = node_two.sorted_connected_nodes.index(
                    int(node_one_index)
                )
                node_one.outwave_np[np_one] = node_one.outwave[node_two_index]
                node_two.outwave_np[np_two] = node_two.outwave[node_one_index]

    # -------------------------------------------------------------------------
    # Plotting methods
    # -------------------------------------------------------------------------

    def draw(
        self,
        show_indices: bool = False,
        show_external_indices: bool = False,
        equal_aspect: bool = False,
    ) -> None:
        """Draw network"""
        fig, ax = plt.subplots()
        if equal_aspect:
            ax.set_aspect("equal")

        # Plot links
        for link in self.links:
            node_1_index, node_2_index = link.node_indices
            node_1_pos = self.get_node(node_1_index).position
            node_2_pos = self.get_node(node_2_index).position
            link.draw(ax, node_1_pos, node_2_pos, show_indices)

        # Plot nodes
        for node in self.nodes:
            node.draw(ax, show_indices, show_external_indices)

    def plot_fields(
        self,
        highlight_nodes: list[int] | None = None,
        title: str = "Field distribution",
        max_val: float | None = None,
    ) -> None:
        """Show internal field distribution in the network"""
        if highlight_nodes is None:
            highlight_nodes = []

        fig, ax = plt.subplots()
        # ax.set_aspect("equal")
        ax.set_title(title)
        # Get link intensities to define the colormap
        power_diffs = [link.power_diff for link in self.links]
        vmax = max(power_diffs) if max_val is None else max_val
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap = plt.cm.viridis

        # Plot links
        for link in self.links:
            node_1_index, node_2_index = link.node_indices
            node_1_pos = self.get_node(node_1_index).position
            node_2_pos = self.get_node(node_2_index).position
            color = cmap(norm(link.power_diff))
            link.draw(ax, node_1_pos, node_2_pos, color=color)

            length = link.length
            x_1, x_2 = node_1_pos[0], node_2_pos[0]
            y_1, y_2 = node_1_pos[1], node_2_pos[1]
            cx = (x_1 + x_2) / 2
            cy = (y_1 + y_2) / 2
            if link.power_direction == 1:
                dx = x_2 - x_1
                dy = y_2 - y_1
            else:
                dx = x_1 - x_2
                dy = y_1 - y_2
            dx /= 100
            dy /= 100

            ax.quiver(
                cx,
                cy,
                dx,
                dy,
                scale=0.1,
                scale_units="xy",
                angles="xy",
                color=color,
            )

            # arrow = FancyArrowPatch(
            #     (cx, cy),  # Starting point of the arrowhead
            #     (cx + dx, cy + dy),  # Ending point of the arrowhead
            #     arrowstyle='->',  # Arrow style
            #     mutation_scale=1e-6,  # Increase this value for a chunkier arrowhead
            #     color=color,  # Arrow color
            # )

            # # Add the arrow to the axis
            # ax.add_patch(arrow)
            # arrowhead_size = 0.0000005 * length
            # ax.arrow(
            #     cx,
            #     cy,
            #     dx,
            #     dy,
            #     lw=0.0,
            #     head_width=arrowhead_size,
            #     head_length=arrowhead_size,
            #     color=color,
            # )

        # Add the colorbar
        sm = plt.cm.ScalarMappable(norm, cmap)
        cbar = fig.colorbar(sm, ax=ax, orientation="vertical")

        # background color
        # bg_color = cmap(0)
        # ax.set_facecolor(bg_color)

        # external nodes
        for node in self.external_nodes:
            node.draw(ax, color="white")

        # Highlight nodes
        for node_index in highlight_nodes:
            node = self.get_node(node_index)
            node.draw(ax, color="red")
