    def update(self, direction: str = "forward") -> None:
        """
        Updates output/input amplitudes according to scattering matrix

        Parameters
        ----------
        direction : str, optional
            Set to 'forward' or 'backwards' depending on recursive algorithm
            being used. The default is 'forward'."""
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
