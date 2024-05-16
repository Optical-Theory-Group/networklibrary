
    def update_S_matrices(self, k0: float | complex) -> None:
        """Function to set the scattering matrix of the link."""
        length = self.length
        n = self.n(k0)
        Dn = self.Dn(k0)
        self.S = np.array()
        self.S_mat = np.array(
            [
                [0, np.exp(1j * (n + Dn) * k0 * length)],
                [np.exp(1j * (n + Dn) * k0 * length), 0],
            ]
        )
        self.iS_mat = np.array(
            [
                [0, np.exp(-1j * (n + Dn) * k0 * length)],
                [np.exp(-1j * (n + Dn) * k0 * length), 0],
            ]
        )
