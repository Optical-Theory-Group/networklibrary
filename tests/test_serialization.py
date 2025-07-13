import pickle

from complex_network.networks.network_spec import NetworkSpec
from complex_network.networks.network_factory import generate_network


def main():
    spec = NetworkSpec(network_type="delaunay", network_shape="circular", num_internal_nodes=1, num_external_nodes=2)
    net = generate_network(spec)
    data = pickle.dumps(net)
    net2 = pickle.loads(data)
    assert net2.num_nodes == net.num_nodes

if __name__ == "__main__":
    main()
