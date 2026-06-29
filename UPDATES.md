# link.py

## added a property called is_perturbed to links, added other properties called isinternal, isexternal to tag internal and external links. Added explanations to power_diff property

## added a method called get_port to find which end of the link is connected to a specified node. 

# node.py

## added is_perturbed, isinternal, isexternal property to the node object
## get_default_S, get_default_S_inv, get_default_dS are static methods and were defined as lambda function with a self. call . This was causing issues with pickling the network object. They are now defined as static methods. 

## made the default scattering mode of nodes "neumann" from "COE"

## added the show_internal_index variable to the draw function.

# Added two more folders called detection and interferometry (coming soon) probably not in this update.

# network_factory.py 
## alot of changes applied by dev-niall as part of clean up not explained here.

## stored the get_S matrices as np.arrays instead of lambda functions. This was part of the earlier cleanup in node.py file. This lamda function interfered with the network pickling which is key for multiprocessed coding.

## added more explanations about how external offsets work for slab and circular geometries of networks

## added a random_seed method to the network_factory for all the network generation methods so that the method is repeatable.

## added a method called symmetric_external which placed the nodes on opposite sides in slab Delaunay networks. DONE for a study of FI values depending on node positions.

## VORONI SLAB still not fixed :/

## Update the _generate_buffon_node_links to a new logic so that the code is much cleaner the math is based on a vectorized implementation that finds the intersection points to M randomly oriented lines.

## Linear networks commented out due to lack of testing

## Defined a class called UnionFind that handles the fully_connected argument for buffon_networks. Implemented as a class inorder to fix a previous get_S_ee error

## Added network_path_search an entire library for finding multiple scattered paths: (Comming soon)


# network.py

## __init__ function can now also now store the networkspec object if provided in the argument. This can be usefull to know the properties of the network if needed called by network.network_spec

## _reset_fields set the initial inwave and outwave indices as strings. changed them to integer indices for faster processing.

## added two more lines inside _set_matrix_calc_utils: internal_link_indices_A_to_B and internal_link_indices_B_to_A that precomputes the internal link fields.

## Applied the changes in _reset_fields to _get_network_matrix_maps that expected dicts with strings as keys and gave them the integer keys. These changes are also applied to reset_dict_indices, add_node_to_link, _set_network_fields, _set_incident_field, _update_outgoing_fields, _get_network_step_matrix, get_P_ii, get_S_ii, get_P_ii_inv, get_P_ei, get_P_ei_inv, get_dP_ei for consistency

## added a property called spec that returned the NetworkSpec of a network object.

## Added network properties to network
### sigma: small world coefficient defined as (C/C_r)/(L/L_r) 
### Omega: another parameter that quantifies the small world nature of the network and is defined as L_r/L - C/C_l.

## Added a function called get_connecting_nodes that returns a list of nodes that are connected to the node under observation

## minor variable changes and comments added to add_segment_to_link function

## Added some explanation to update_node_scattering_matrix

## Added a function add_node that adds a new node to a defined spatial position with defined connections. STILL need to make a warning that looks at whether the new node produces crossing links that are invalid.

## Added a function add_link that adds a new links between [(node_1,node_2), ...]. Again, need to write code to see if they cross, currently network.draw can be used to manually verify.

## modified get_S_ee, get_S_ee_inv, to use np.linalg.solve to find the inverses instead of np.linalg.inv which is a more expensive opeations. used np.eye to generate identity matrices instead of np.identity for a more cleaner code.

## defined get_RT_matrix for slab geometries that returns the scattering matrix in the classic submatrix form [[r,t][t',r']] and defined their helper functions get_reflection_matrix, get_transmission_matrix.

## defined a function get_internal_link_fields that computes the fields along the internal links given an exciation vector on the ports I_e.


## defined a function get_all_link_energy_densities (from dev-niall branch)

## added definitions to existing functions and comments explaining some logic

## Rewrote _set_network_fields more cleanly no change in logic

## modified .draw method: added show_internal_indices and show_external_indices that only shows the internal and external indices to reduce visual clutter. added highlight_perturbed_nodes and highlight_perturbed_links

## modified some plotting functions so that fig and ax can be passed on as arguments so that plots can be plotted on top of other plots if you wanted. added the save_dir and hide_axes arguments to some plotting functions that missed it.

## Added a scientific plotter that make the plots more cleanly. By scientific plotter the xticks are written as 10^-4 instead of 0.0001 for cleaner axis ticks

## changed the name from plot_fields to plot_power_flow to more capture the idea of what it is plotting

## Wrote a method called plot_network_heatmap, which when given the spatial data for all links in the form {(link_index, fractional_ratio): value} plots heatmap along the link. Usefull for visualizing spatial variation of measurements along the network

## defined spatial_position_within_link for finding the spatial location that corresponds to a (link_index, frac_location) more easily

## defined functions such as weighted_adjacency_matrix, weighted_lapalcain, resolvent, effective_optical_distance that are network properties useful for correlating how certain measurements with network properties

# Modified breadth_first_search_simple_paths to work with multiprocessing to find the paths different paths between two nodes in a network. Still need to solve the memory overflow issue when dealing with big network and try to write a GPU version of code if needed.

## defined helper functions like get_path_length, _get_lengths_along_path that returns the length of a path. get_optical_path_length that retuns the optical path length traversed by light when travelling along that path. other functions include _init_pool: helps with multiprocessing, _bfs_worker, _bfs_cpu, _make_frontier: multiprocessing helper for BFS algorithm, _safe_norm: helper for the plotting function so that negative values are not provided to log plots


# Added a new file called scattering_ensemble.py which is an updated version of ensemble.py which runs the network in a multiprocessed way and stores the data in .h5 files for better compression.

# Added more files to the example .ipynb files.

# Made changes to scattering_matrices node_matrix.py and link_matrices.py to make the network object picklable.

