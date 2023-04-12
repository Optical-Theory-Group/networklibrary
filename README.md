# networklibrary: Python library for complex network class.

---

This library defines the Network Class, which can be used to generate and simulate the properties of various different types of complex networks.

---

## Structure
A network/graph is a collection of nodes connected by edges. 

| File | Contents |
| ---- | -------- |
| network.py       | Core Network class with network related properties/methods/functions |
| node.py          | Node class with node related properties/methods/functions |
| link.py          | Link class with edge related properties/methods/functions | 
| ensemble.py      | A helper class to generate an ensemble of random networks and store in HDF5 format. |
| util.py          | Some custom utility functions used in other library classes |
| _generator.py    | base class inherited by Network class. Generator functions for different network topologies |
| _numpy_json.py    | json custom encoder to handle complex numpy arrays |
| _dict_hdf5        | custom dict <--> hdf5 convertor to handle complex numpy arrays |


## Package versions
The code was written and tested in Python 3.9.13 using 
numpy 
scipy
matplotlib 
seaborn
