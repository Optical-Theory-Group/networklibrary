    @staticmethod
    def get_default_properties():
        default_values = {
            "number": None,  # unique numerical id for node
            "position": None,  # position of node (2d assumed)
            "node_type": "internal",  # 'internal' or 'exit' specifying nature of node
            "n_connect": 0,  # number of connecting links/edges
            "sorted_connected_nodes": [],
            # list of ids of nodes that are connected. order matches that of S_mat
            "Smat_type": None,  # string identifier for nature of scattering matrix
            "scat_loss": 0,  # parameter describing fractional scattering loss
            "Smat_params": {},  # dictionary of any additional parameters used to generate S_mat
            "inwave": {},  # dictionary of inwave amplitudes {node id: amplitude, ... }
            "outwave": {},  # dictionary of outwave amplitudes {node id: amplitude, ...,
            #                                   'loss node id': loss amplitude, ...}
            "inwave_np": None,  # array of inwave amplitudes
            "outwave_np": None,  # array of outwave amplitudes
            "S_mat": None,  # numpy array specifying scattering matrix
            "iS_mat": None,  # numpy array specifying inverse scattering matrix
        }

        return default_values
