
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
            perturbed_node_index, perturbed_angle_index, k0
        )

        ws = -1j * S_ee_inv @ dS_ee_dt
        return ws


    def get_wigner_smith_r(
        self,
        k0: float | complex,
        perturbed_node_index: int,
    ) -> np.ndarray:
        """Calculate analytically the Wigner-Smith operator
        Q = -i * S^-1 * dS/dr"""

        S_ee_inv = self.get_S_ee_inv(k0)
        dS_ee_dr = self.get_dS_ee_dr(k0, perturbed_node_index)

        ws = -1j * S_ee_inv @ dS_ee_dr
        return ws

    def get_wigner_smith_s(
        self,
        k0: float | complex,
        perturbed_node_index: int,
    ) -> np.ndarray:
        """Calculate analytically the Wigner-Smith operator
        Q = -i * S^-1 * dS/ds"""

        S_ee_inv = self.get_S_ee_inv(k0)
        dS_ee_ds = self.get_dS_ee_ds(k0, perturbed_node_index)

        ws = -1j * S_ee_inv @ dS_ee_ds
        return ws

    def get_dS_ee_dt(
        self, perturbed_node_index: int, perturbed_angle_index: int, k0
    ) -> np.ndarray:
        """Get the derivative of S_ee with respect to eigenphase t
        (eigenvalue is e^it)"""

        P_ei = self.get_P_ei(k0)
        P_ie = self.get_P_ie(k0)
        P_ii = self.get_P_ii(k0)
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

    def get_dS_ee_dr(
        self, k0: float | complex, perturbed_node_index: int
    ) -> np.ndarray:
        """Get the derivative of S_ee with respect to the reflection
        coefficient r"""

        P_ei = self.get_P_ei(k0)
        P_ie = self.get_P_ie(k0)
        P_ii = self.get_P_ii(k0)
        S_ii = self.get_S_ii()

        dP_ei_dr = np.zeros(P_ei.shape, dtype=np.complex128)
        dP_ie_dr = np.zeros(P_ie.shape, dtype=np.complex128)
        dP_ii_dr = np.zeros(P_ii.shape, dtype=np.complex128)
        dS_ii_dr = self.get_dS_ii_dr(perturbed_node_index)

        SP = S_ii @ P_ii
        inv = np.linalg.inv(np.identity(len(SP)) - SP)
        dinv_dr = inv @ (S_ii @ dP_ii_dr + dS_ii_dr @ P_ii) @ inv

        term_one = dP_ei_dr @ inv @ S_ii @ P_ie
        term_two = P_ei @ dinv_dr @ S_ii @ P_ie
        term_three = P_ei @ inv @ dS_ii_dr @ P_ie
        term_four = P_ei @ inv @ S_ii @ dP_ie_dr

        dS_ee_dr = term_one + term_two + term_three + term_four
        return dS_ee_dr

    def get_dS_ee_ds(
        self, k0: float | complex, perturbed_node_index: int
    ) -> np.ndarray:
        """Get the derivative of S_ee with respect to the fractional
        position s"""

        P_ei = self.get_P_ei(k0)
        P_ie = self.get_P_ie(k0)
        P_ii = self.get_P_ii(k0)
        S_ii = self.get_S_ii()

        dP_ei_ds = np.zeros(P_ei.shape, dtype=np.complex128)
        dP_ie_ds = np.zeros(P_ie.shape, dtype=np.complex128)
        dP_ii_ds = self.get_dP_ii_ds(k0, perturbed_node_index)
        dS_ii_ds = np.zeros(S_ii.shape, dtype=np.complex128)

        SP = S_ii @ P_ii
        inv = np.linalg.inv(np.identity(len(SP)) - SP)
        dinv_ds = inv @ (S_ii @ dP_ii_ds + dS_ii_ds @ P_ii) @ inv

        term_one = dP_ei_ds @ inv @ S_ii @ P_ie
        term_two = P_ei @ dinv_ds @ S_ii @ P_ie
        term_three = P_ei @ inv @ dS_ii_ds @ P_ie
        term_four = P_ei @ inv @ S_ii @ dP_ie_ds

        dS_ee_ds = term_one + term_two + term_three + term_four
        return dS_ee_ds


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

    def get_P_ii_dP_ii(self, k0: complex) -> tuple[np.ndarray, np.ndarray]:
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
            n = link.n(k0)

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

    def get_dS_ii_dr(
        self,
        perturbed_node_index: int,
        internal_vector_length: int | None = None,
        internal_scattering_slices: slice | None = None,
    ) -> np.ndarray:
        """Return the derivative of S_ii with respect to the perturbed node's
        reflection coefficeint"""

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
            r = -node_S_mat[0, 0]
            fac = -r / np.sqrt(1 - r**2)
            derivative = np.array([[-1.0, fac], [fac, 1.0]])

            new_slice = internal_scattering_slices[str(node_index)]
            internal_S[new_slice, new_slice] = derivative
        return internal_S

    def get_dP_ii_ds(
        self,
        k0: float | complex,
        perturbed_node_index: int,
        internal_vector_length: int | None = None,
        internal_scattering_map: dict[str, int] | None = None,
    ) -> np.ndarray:
        """Return derivative of P_ii with respect to k0"""
        self.update_links(k0)

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

        # Determine the two links that are relevant
        perturbed_node = self.get_node(perturbed_node_index)
        link_one_index, link_two_index = perturbed_node.sorted_connected_links
        link_one = self.get_link(link_one_index)
        link_two = self.get_link(link_two_index)
        total_length = link_one.length + link_two.length

        for i, link in enumerate([link_one, link_two]):
            sign = 1.0 if i == 0 else -1.0
            n = link.n(k0)
            dn = link.dn(k0)
            Dn = link.Dn(k0)
            dDn = link.dDn(k0)

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
            internal_P[row, col] = (
                sign * phase_factor * 1j * k0 * (n + Dn) * total_length
            )
            # Wave propagating the other way
            internal_P[col, row] = (
                sign * phase_factor * 1j * k0 * (n + Dn) * total_length
            )

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

    def U_0(self ,k0):
        "s""Calculate the U_0 matrix (see theory notes)"""
        self.update_links(k0)
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
                    n = link.n(k0)
                    Dn = link.Dn(k0)

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
                        * (1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                    )

                    partial_sum += (
                        I_mp
                        * np.conj(I_mq)
                        * (np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0)
                    )

                # Next loop over external links
                for link in self.external_links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)

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
                        * (1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                    )

                    partial_sum += (
                        O_mp
                        * np.conj(O_mq)
                        * (np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0)
                    )

                U_0[q, p] = partial_sum

        return U_0



    def get_U_1_n(self, k0, link_index) -> np.ndarray:
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

                    # Skip unless we are at the key link
                    if link.index != link_index:
                        continue

                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)

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
                        k0
                        * (
                            O_mp
                            * np.conj(O_mq)
                            * (np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                            + I_mp
                            * np.conj(I_mq)
                            * (np.exp(2 * (n + Dn) * np.imag(k0) * length))
                        )
                        * length
                    )

                # Next loop over external links
                for link in self.external_links:

                    # Skip unless we are at the key link
                    if link.index != link_index:
                        continue

                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)

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
                        k0
                        * (
                            I_mp
                            * np.conj(I_mq)
                            * (np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                            + O_mp
                            * np.conj(O_mq)
                            * (np.exp(2 * (n + Dn) * np.imag(k0) * length))
                        )
                        * length
                    )

                U_1[q, p] = partial_sum

        return U_1



    def get_U_2_n(self, k0, link_index) -> np.ndarray:
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
                    if link.index != link_index:
                        continue

                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)

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
                        0.5
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            O_mp
                            * np.conj(I_mq)
                            * (
                                np.exp(2j * (n + Dn) * np.real(k0) * length)
                                - 1.0
                            )
                        )
                    )

                    partial_sum += (
                        0.5
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            I_mp
                            * np.conj(O_mq)
                            * (
                                1.0
                                - np.exp(-2j * (n + Dn) * np.real(k0) * length)
                            )
                        )
                    )

                # Next loop over external links
                for link in self.external_links:
                    if link.index != link_index:
                        continue

                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)

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
                        0.5
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            I_mp
                            * np.conj(O_mq)
                            * (
                                np.exp(2j * (n + Dn) * np.real(k0) * length)
                                - 1.0
                            )
                        )
                    )

                    partial_sum += (
                        0.5
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            O_mp
                            * np.conj(I_mq)
                            * (
                                1.0
                                - np.exp(-2j * (n + Dn) * np.real(k0) * length)
                            )
                        )
                    )

                U_2[q, p] = partial_sum

        return U_2



    def get_U_2_old(self, k0) -> np.ndarray:
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
                    n = link.n(k0)
                    Dn = link.Dn(k0)
                    dn = link.dn(k0)
                    dDn = link.dDn(k0)

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
                        0.5
                        * (dn + dDn)
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            O_mp
                            * np.conj(I_mq)
                            * (
                                np.exp(2j * (n + Dn) * np.real(k0) * length)
                                - 1.0
                            )
                        )
                    )

                    partial_sum += (
                        0.5
                        * (dn + dDn)
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            I_mp
                            * np.conj(O_mq)
                            * (
                                1.0
                                - np.exp(-2j * (n + Dn) * np.real(k0) * length)
                            )
                        )
                    )

                # Next loop over external links
                for link in self.external_links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)
                    dn = link.dn(k0)
                    dDn = link.dDn(k0)

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
                        0.5
                        * (dn + dDn)
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            I_mp
                            * np.conj(O_mq)
                            * (
                                np.exp(2j * (n + Dn) * np.real(k0) * length)
                                - 1.0
                            )
                        )
                    )

                    partial_sum += (
                        0.5
                        * (dn + dDn)
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            O_mp
                            * np.conj(I_mq)
                            * (
                                1.0
                                - np.exp(-2j * (n + Dn) * np.real(k0) * length)
                            )
                        )
                    )

                U_2[q, p] = partial_sum

        return U_2





    def get_U_3_old(self, k0: float | complex, dk: float = 1e-5) -> np.ndarray:
        """Calculate the U_3 matrix associated with the wavenumber
        (see theory notes)"""
        # Perturbation parameters
        k1 = k0 + dk

        # Get network matrices
        unperturbed_network_matrix = self.get_network_matrix(k0)
        perturbed_network_matrix = self.get_network_matrix(k1)

        # Get the scattered fields for each incident field
        unperturbed_outgoing_vectors = []
        perturbed_outgoing_vectors = []

        num_externals = self.num_external_nodes
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
        internal_scattering_map = self.internal_scattering_map
        external_scattering_map = self.external_scattering_map

        for q in range(num_externals):
            for p in range(num_externals):

                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.internal_links:
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

                    d_I_mp = diff_I_mp / dk
                    d_I_mq = diff_I_mq / dk
                    d_O_mp = diff_O_mp / dk
                    d_O_mq = diff_O_mq / dk

                    partial_sum += 1j * d_O_mp * np.conj(O_mq_before) * (
                        1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length)
                    ) + 1j * d_I_mp * np.conj(I_mq_before) * (
                        np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0
                    )

                # Next loop over external links
                for link in self.external_links:
                    continue
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

                    d_I_mp = diff_I_mp / dk
                    d_I_mq = diff_I_mq / dk
                    d_O_mp = diff_O_mp / dk
                    d_O_mq = diff_O_mq / dk

                    partial_sum += 1j * d_I_mp * np.conj(I_mq_before) * (
                        1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length)
                    ) + 1j * d_O_mp * np.conj(O_mq_before) * (
                        np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0
                    )

                U_3[q, p] = partial_sum

        return U_3






    def get_wigner_smith_k0(self, k0: float | complex) -> np.ndarray:
        """Calculate directly the wavenumber Wigner-Smith operator
        Q = -i * S^-1 * dS/dk0"""
        S_ee_inv = self.get_S_ee_inv(k0)
        dS_ee_dk0 = self.get_dS_ee_dk0(k0)
        ws = -1j * S_ee_inv @ dS_ee_dk0
        return ws




    def get_wigner_smith_Dn(
        self,
        k0: float | complex,
        perturbed_link_index: int,
    ) -> np.ndarray:
        """Calculate analytically the Wigner-Smith operator
        Q = -i * S^-1 * dS/dDn associated with a link refractive index
        perturbation"""
        S_ee_inv = self.get_S_ee_inv(k0)
        dS_ee_dDn = self.get_dS_ee_dDn(k0, perturbed_link_index)
        ws = -1j * S_ee_inv @ dS_ee_dDn
        return ws


    # -------------------------------------------------------------------------
    #  Methods for altering/perturbing the network
    # -------------------------------------------------------------------------

    def get_dS_ee_dk0(self, k0) -> np.ndarray:
        """Get the derivative of S_ee with respect to k0"""

        P_ei = self.get_P_ei(k0)
        P_ie = self.get_P_ie(k0)
        P_ii = self.get_P_ii(k0)
        S_ii = self.get_S_ii()

        dP_ei_dk0 = self.get_dP_ei_dk0(k0)
        dP_ie_dk0 = self.get_dP_ie_dk0(k0)
        dP_ii_dk0 = self.get_dP_ii_dk0(k0)
        dS_ii_dk0 = np.zeros(S_ii.shape, dtype=np.complex128)

        inv = np.linalg.inv(np.identity(len(S_ii)) - S_ii @ P_ii)
        dinv_dk0 = inv @ (dS_ii_dk0 @ P_ii + S_ii @ dP_ii_dk0) @ inv

        term_one = dP_ei_dk0 @ inv @ S_ii @ P_ie
        term_two = P_ei @ dinv_dk0 @ S_ii @ P_ie
        term_three = P_ei @ inv @ dS_ii_dk0 @ P_ie
        term_four = P_ei @ inv @ S_ii @ dP_ie_dk0

        dS_ee_dk0 = term_one + term_two + term_three + term_four
        return dS_ee_dk0

    def get_dS_ee_dDn(
        self, k0: float | complex, perturbed_link_index: int
    ) -> np.ndarray:
        """Get the derivative of S_ee with respect to change in refractive
        index alpha"""

        P_ei = self.get_P_ei(k0)
        P_ie = self.get_P_ie(k0)
        P_ii = self.get_P_ii(k0)
        S_ii = self.get_S_ii()

        dP_ei_dDn = np.zeros(P_ei.shape, dtype=np.complex128)
        dP_ie_dDn = np.zeros(P_ie.shape, dtype=np.complex128)
        dP_ii_dDn = self.get_dP_ii_dDn(k0, perturbed_link_index)
        dS_ii_dDn = np.zeros(S_ii.shape, dtype=np.complex128)

        SP = S_ii @ P_ii
        inv = np.linalg.inv(np.identity(len(SP)) - SP)
        dinv_dDn = inv @ (S_ii @ dP_ii_dDn + dS_ii_dDn @ P_ii) @ inv

        term_one = dP_ei_dDn @ inv @ S_ii @ P_ie
        term_two = P_ei @ dinv_dDn @ S_ii @ P_ie
        term_three = P_ei @ inv @ dS_ii_dDn @ P_ie
        term_four = P_ei @ inv @ S_ii @ dP_ie_dDn

        dS_ee_dDn = term_one + term_two + term_three + term_four
        return dS_ee_dDn

    def get_dS_ie_dk0(self, k0) -> np.ndarray:
        """Get the derivative of S_ie with respect to k0"""

        P_ie = self.get_P_ie(k0)
        P_ii = self.get_P_ii(k0)
        S_ii = self.get_S_ii()

        dP_ie_dk0 = self.get_dP_ie_dk0(k0)
        dP_ii_dk0 = self.get_dP_ii_dk0(k0)
        dS_ii_dk0 = np.zeros(S_ii.shape, dtype=np.complex128)

        # Top matrix (O part)
        inv = np.linalg.inv(
            np.identity(len(S_ii), np.complex128) - S_ii @ P_ii
        )
        dinv_dk0 = inv @ (dS_ii_dk0 @ P_ii + S_ii @ dP_ii_dk0) @ inv

        term_one = dinv_dk0 @ S_ii @ P_ie
        term_two = inv @ dS_ii_dk0 @ P_ie
        term_three = inv @ S_ii @ dP_ie_dk0
        top = term_one + term_two + term_three

        # Bottom matrix (I part)
        inv = np.linalg.inv(
            np.identity(len(P_ii), dtype=np.complex128) - P_ii @ S_ii
        )
        dinv_dk0 = inv @ (dP_ii_dk0 @ S_ii + P_ii @ dS_ii_dk0) @ inv

        term_one = dinv_dk0 @ P_ie
        term_two = inv @ dP_ie_dk0
        bottom = term_one + term_two

        dS_ie_dk0 = np.block([[top], [bottom]])
        return dS_ie_dk0

    def get_dS_ie_dDn(
        self, k0: complex, perturbed_link_index: int
    ) -> np.ndarray:
        """Get the derivative of S_ie with respect to k0"""

        P_ie = self.get_P_ie(k0)
        P_ii = self.get_P_ii(k0)
        S_ii = self.get_S_ii()

        dP_ie_dDn = np.zeros(P_ie.shape, dtype=np.complex128)
        dP_ii_dDn = self.get_dP_ii_dDn(k0, perturbed_link_index)
        dS_ii_dDn = np.zeros(S_ii.shape, dtype=np.complex128)

        # Top matrix (O part)
        inv = np.linalg.inv(
            np.identity(len(S_ii), np.complex128) - S_ii @ P_ii
        )
        dinv_dDn = inv @ (dS_ii_dDn @ P_ii + S_ii @ dP_ii_dDn) @ inv

        term_one = dinv_dDn @ S_ii @ P_ie
        term_two = inv @ dS_ii_dDn @ P_ie
        term_three = inv @ S_ii @ dP_ie_dDn
        top = term_one + term_two + term_three

        # Bottom matrix (I part)
        inv = np.linalg.inv(
            np.identity(len(P_ii), dtype=np.complex128) - P_ii @ S_ii
        )
        dinv_dDn = inv @ (dP_ii_dDn @ S_ii + P_ii @ dS_ii_dDn) @ inv

        term_one = dinv_dDn @ P_ie
        term_two = inv @ dP_ie_dDn
        bottom = term_one + term_two

        dS_ie_dDn = np.block([[top], [bottom]])
        return dS_ie_dDn

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

    def get_dP_ii_dk0(
        self,
        k0: float | complex,
    ) -> np.ndarray:
        """Return derivative of P_ii with respect to k0"""
        self.update_links(k0)

        # Get internal P
        internal_P = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.internal_links:
            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]

            n = link.n(k0)
            dn = link.dn(k0)
            Dn = link.Dn(k0)
            dDn = link.dDn(k0)

            # Wave that is going into node_one
            row = internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            col = internal_scattering_map[
                f"{str(node_two_index)},{str(node_one_index)}"
            ]
            internal_P[row, col] = (
                phase_factor * 1j * link.length * (n + Dn + k0 * (dn + dDn))
            )
            # Wave propagating the other way
            internal_P[col, row] = (
                phase_factor * 1j * link.length * (n + Dn + k0 * (dn + dDn))
            )
        return internal_P

    def get_dP_ii_dDn(
        self,
        k0: float | complex,
        perturbed_link_index: int,
    ) -> np.ndarray:
        """Return derivative of P_ii with respect to alpha"""
        self.update_links(k0)

        # Get internal P
        internal_P = np.zeros(
            (internal_vector_length, internal_vector_length),
            dtype=np.complex128,
        )

        # Determine the two links that are relevant
        perturbed_link = self.get_link(perturbed_link_index)
        link_S_mat = perturbed_link.S_mat
        phase_factor = link_S_mat[0, 1]
        length = perturbed_link.length

        # Get joining nodes
        node_one_index, node_two_index = perturbed_link.node_indices

        # Wave that is going into node_one
        row = internal_scattering_map[
            f"{str(node_one_index)},{str(node_two_index)}"
        ]
        col = internal_scattering_map[
            f"{str(node_two_index)},{str(node_one_index)}"
        ]
        internal_P[row, col] = phase_factor * 1j * k0 * length
        # Wave propagating the other way
        internal_P[col, row] = phase_factor * 1j * k0 * length

        return internal_P

    def get_dP_ei_dk0(
        self,
        k0: float | complex,
    ) -> np.ndarray:
        """Return derivative of P_ei with respect to k0"""
        self.update_links(k0)

        external_P = np.zeros(
            (self.external_vector_length, self.internal_vector_length),
            dtype=np.complex128,
        )
        for link in self.external_links:
            n = link.n(k0)
            dn = link.dn(k0)
            Dn = link.Dn(k0)
            dDn = link.dDn(k0)

            node_one_index, node_two_index = link.node_indices
            link_S_mat = link.S_mat
            phase_factor = link_S_mat[0, 1]
            row = self.external_scattering_map[f"{str(node_two_index)}"]
            col = self.internal_scattering_map[
                f"{str(node_one_index)},{str(node_two_index)}"
            ]
            external_P[row, col] = (
                phase_factor * 1j * link.length * (n + Dn + k0 * (dn + dDn))
            )
        return external_P

    def get_dP_ie_dk0(
        self,
        k0: float | complex,
    ) -> np.ndarray:
        """Return derivative of P_ie with respect to k0"""
        return self.get_dP_ei_dk0(k0).T



    def get_U_1_k0(self, k0: float | complex) -> np.ndarray:
        """Calculate the U_1 matrix relevant to the k0 Wigner-Smith operator
        (see theory notes)"""
        self.update_links(k0)

        num_externals = self.num_external_nodes
        U_1 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)
                    dn = link.dn(k0)
                    dDn = link.dDn(k0)

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]

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
                        (n + Dn + k0 * (dn + dDn))
                        * (
                            O_mp
                            * np.conj(O_mq)
                            * (np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                            + I_mp
                            * np.conj(I_mq)
                            * (np.exp(2 * (n + Dn) * np.imag(k0) * length))
                        )
                        * length
                    )

                U_1[q, p] = partial_sum

        return U_1

    def get_U_1_Dn(
        self, k0: float | complex, pertutbed_link_index: int
    ) -> np.ndarray:
        """Calculate the U_1 matrix (see theory notes)"""
        self.update_links(k0)

        num_externals = self.num_external_nodes
        U_1 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.links:
                    if link.index != pertutbed_link_index:
                        continue

                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)
                    dn = link.dn(k0)
                    dDn = link.dDn(k0)

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]

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
                        k0
                        * (
                            O_mp
                            * np.conj(O_mq)
                            * (np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                            + I_mp
                            * np.conj(I_mq)
                            * (np.exp(2 * (n + Dn) * np.imag(k0) * length))
                        )
                        * length
                    )

                U_1[q, p] = partial_sum

        return U_1




    def get_U_2_k0(self, k0: float | complex) -> np.ndarray:
        """Calculate the U_2 matrix (see theory notes)"""
        self.update_links(k0)

        num_externals = self.num_external_nodes
        U_2 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j

                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)
                    dn = link.dn(k0)
                    dDn = link.dDn(k0)

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]

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
                        0.5
                        * (dn + dDn)
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            O_mp
                            * np.conj(I_mq)
                            * (
                                np.exp(2j * (n + Dn) * np.real(k0) * length)
                                - 1.0
                            )
                            + I_mp
                            * np.conj(O_mq)
                            * (
                                1.0
                                - np.exp(-2j * (n + Dn) * np.real(k0) * length)
                            )
                        )
                    )

                U_2[q, p] = partial_sum

        return U_2

    def get_U_2_Dn(
        self, k0: float | complex, perturbed_link_index: int
    ) -> np.ndarray:
        """Calculate the U_2 matrix (see theory notes)"""
        self.update_links(k0)

        num_externals = self.num_external_nodes
        U_2 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)

        for q in range(num_externals):
            for p in range(num_externals):
                # First loop over internal links
                partial_sum = 0.0 + 0.0j
                for link in self.internal_links:
                    if link.index != perturbed_link_index:
                        continue

                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)
                    dn = link.dn(k0)
                    dDn = link.dDn(k0)

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]

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
                        0.5
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            O_mp
                            * np.conj(I_mq)
                            * (
                                np.exp(2j * (n + Dn) * np.real(k0) * length)
                                - 1.0
                            )
                        )
                    )

                    partial_sum += (
                        0.5
                        / (n + Dn)
                        * np.imag(k0)
                        / np.real(k0)
                        * (
                            I_mp
                            * np.conj(O_mq)
                            * (
                                1.0
                                - np.exp(-2j * (n + Dn) * np.real(k0) * length)
                            )
                        )
                    )

                U_2[q, p] = partial_sum

        return U_2



    def get_U_3_k0(self, k0: float | complex) -> np.ndarray:
        """Calculate the U_3 matrix associated with the wavenumber
        (see theory notes)"""
        self.update_links(k0)

        num_externals = self.num_external_nodes
        U_3 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)
        dS_ie_dk0 = self.get_dS_ie_dk0(k0)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []
        dO_int_vectors = []
        dI_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            doutgoing_vector = dS_ie_dk0 @ incident_field

            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            new_dO = doutgoing_vector[: int(len(outgoing_vector) / 2)]
            new_dI = doutgoing_vector[int(len(outgoing_vector) / 2) :]

            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)
            dO_int_vectors.append(new_dO)
            dI_int_vectors.append(new_dI)

        internal_scattering_map = self.internal_scattering_map

        for q in range(num_externals):
            for p in range(num_externals):

                partial_sum = 0.0 + 0.0j
                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    dq_o = dO_int_vectors[q]
                    dq_i = dI_int_vectors[q]

                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]
                    dp_o = dO_int_vectors[p]
                    dp_i = dI_int_vectors[p]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    dI_mp = dp_i[index]
                    dI_mq = dq_i[index]
                    dO_mp = dp_o[index]
                    dO_mq = dq_o[index]

                    partial_sum += 1j * (
                        dO_mp
                        * np.conj(O_mq)
                        * (1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                        + dI_mp
                        * np.conj(I_mq)
                        * (np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0)
                    )

                U_3[q, p] = partial_sum

        return U_3

    def get_U_3_Dn(
        self, k0: float | complex, perturbed_link_index: int
    ) -> np.ndarray:
        """Calculate the U_3 matrix associated with the wavenumber
        (see theory notes)"""
        self.update_links(k0)

        num_externals = self.num_external_nodes
        U_3 = np.zeros((num_externals, num_externals), dtype=np.complex128)

        S_ie = self.get_S_ie(k0)
        dS_ie_dk0 = self.get_dS_ie_dDn(k0, perturbed_link_index)

        # Get the scattered fields for each incident field
        O_int_vectors = []
        I_int_vectors = []
        dO_int_vectors = []
        dI_int_vectors = []

        for i in range(num_externals):
            incident_field = np.zeros(num_externals, dtype=np.complex128)
            incident_field[i] = 1.0

            # Full length interior field vector
            outgoing_vector = S_ie @ incident_field
            doutgoing_vector = dS_ie_dk0 @ incident_field

            new_O = outgoing_vector[: int(len(outgoing_vector) / 2)]
            new_I = outgoing_vector[int(len(outgoing_vector) / 2) :]
            new_dO = doutgoing_vector[: int(len(outgoing_vector) / 2)]
            new_dI = doutgoing_vector[int(len(outgoing_vector) / 2) :]

            O_int_vectors.append(new_O)
            I_int_vectors.append(new_I)
            dO_int_vectors.append(new_dO)
            dI_int_vectors.append(new_dI)

        internal_scattering_map = self.internal_scattering_map

        for q in range(num_externals):
            for p in range(num_externals):

                partial_sum = 0.0 + 0.0j
                for link in self.links:
                    length = link.length
                    n = link.n(k0)
                    Dn = link.Dn(k0)

                    q_o = O_int_vectors[q]
                    q_i = I_int_vectors[q]
                    dq_o = dO_int_vectors[q]
                    dq_i = dI_int_vectors[q]

                    p_o = O_int_vectors[p]
                    p_i = I_int_vectors[p]
                    dp_o = dO_int_vectors[p]
                    dp_i = dI_int_vectors[p]

                    # Find the fields in the link
                    node_one_index = link.sorted_connected_nodes[0]
                    node_two_index = link.sorted_connected_nodes[1]
                    key = f"{node_one_index},{node_two_index}"
                    index = internal_scattering_map[key]

                    I_mp = p_i[index]
                    I_mq = q_i[index]
                    O_mp = p_o[index]
                    O_mq = q_o[index]

                    dI_mp = dp_i[index]
                    dI_mq = dq_i[index]
                    dO_mp = dp_o[index]
                    dO_mq = dq_o[index]

                    partial_sum += 1j * (
                        dO_mp
                        * np.conj(O_mq)
                        * (1.0 - np.exp(-2 * (n + Dn) * np.imag(k0) * length))
                        + dI_mp
                        * np.conj(I_mq)
                        * (np.exp(2 * (n + Dn) * np.imag(k0) * length) - 1.0)
                    )

                U_3[q, p] = partial_sum

        return U_3