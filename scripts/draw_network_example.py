
# -*- coding: utf-8 -*-
"""
Created on Mon 18 Oct 2021

@author: mforeman

Draw an example of a network

"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as ptc

from complexnetworklibrary.network import Network

wavelength = 1050e-9
k=(2*np.pi)/wavelength
n=1

# linear
SM_type = 'isotropic_unitary'
seed = np.random.randint(2**10)
print(seed)
qualifier = ''

seed = 2+4408
case = 'voronoi slab'

if case == 'delaunay':
    # delaunay circular
    network_type = 'delaunay'
    network_spec = {'internal_nodes': 60,
                    'exit_nodes': 20,
                    'network_size': 100e-6,
                    'exit_size': 110e-6,
                    'wavenumber': k,
                    'refractive_index': n,
                    'shape': 'circular'}
elif case == 'delaunay slab'  :
    # delaunay slab
    network_type = 'delaunay'
    network_spec = {'internal_nodes': 100,
                    'exit_nodes': 10,
                    'network_size': (100e-6,20e-6),
                    'exit_size': 110e-6,
                    'left_exit_fraction': 0.5,
                    'wavenumber': k,
                    'refractive_index': n,
                    'shape': 'slab'}
elif case == 'voronoi'  :
    # voronoi circular
    network_type = 'voronoi'
    network_spec = {'seed_nodes': 100,
                    'network_size': 100e-6,
                    'exit_size': 110e-6,
                    'wavenumber': k,
                    'refractive_index': n,
                    'shape': 'circular'}
elif case == 'voronoi slab'  :
    # voronoi slab
    network_type = 'voronoi'
    network_spec = {'seed_nodes': 100,
                    'network_size': (100e-6,20e-6),
                    'left_exit_fraction': 0.5,
                    'exit_nodes': 20,
                    'exit_size': 110e-6,
                    'wavenumber': k,
                    'refractive_index': n,
                    'curvature_factor': 10,
                    'shape': 'slab'}
elif case == 'buffon'    :
    # buffon
    network_type = 'buffon'
    network_spec = {'lines': 20,
              'network_size': 100e-6,
              'wavenumber': k,
              'refractive_index': n,
              'fully_connected': True,
              'shape': 'circular',
              }
elif case == 'buffon slab':
    # # buffon slab
    network_type = 'buffon'
    network_spec = {'lines': 20,
              'network_size': (100e-6,20e-6),
              'wavenumber': k,
              'refractive_index': n,
              'fully_connected': True,
              'shape': 'slab',
              }



# network_type = 'archimedean'
# network_spec = {'num_layers':3,
#                 'scale': 100e-6,
#                 'type': 'square',
#                 'exit_nodes': 5} # square,triangular, honeycomb
# qualifier = network_spec['type']

node_spec = {'Smat_type': SM_type,
             'scat_loss': 0,
             'subunitary_factor': 1
             }


# network_type = 'voronoi'
# network_spec = {'seed_nodes': 50,
#                 'exit_nodes': 50,
#                 'network_size': (100e-6,30e-6),
#                 'exit_size': 110e-6,
#                 'left_exit_fraction': 0.5,
#                 'wavenumber': k,
#                 'refractive_index': n,
#                 'shape': 'slab'}


network = Network(network_type,
                  network_spec,
                  node_spec,
                  seed_number=seed)   #CREATE A NETWORK OBJECT

# input_amp = [0]* network.exit_nodes
# input_amp[0] = 1
# int_nodes = network.internal_nodes
# network.initialise_network(SM_type,losses,input_amp)

# smr,node_order = network.scattering_matrix_recursive()
# network.run_network()

# for ii in range(0,50):
#     network.update_network()    #updates and iterate
#     # with PdfPages('network_{}_iterative_frame_{}.pdf'.format(network_type,ii)) as export_pdf:
#     network.draw('intensity')
#     plt.savefig('network_{}_iterative_frame_{}.png'.format(network_type,ii),dpi=300)
#         # network.draw('blank')
#         # export_pdf.savefig()
#     plt.gca.clear()
#     plt.close()

network.draw('')
ax=plt.gca().set_aspect('equal', 'box')


# if case == 'voronoi slab'  :
#     rec1 = ptc.Rectangle((-network_spec['network_size'][0]/2,-network_spec['network_size'][1]/2),
#                     network_spec['network_size'][0],
#                     network_spec['network_size'][1],
#                     edgecolor = 'blue',
#                     fill=False,
#                     )
#     rec2 = ptc.Rectangle((-network_spec['exit_size']/2,-network_spec['network_size'][1]/2),
#                     network_spec['exit_size'],
#                     network_spec['network_size'][1],
#                     edgecolor = 'red',
#                     fill=False,
#                     )
#     ax=plt.gca()
#     # ax.add_patch(rec1)
#     # ax.add_patch(rec2)
#     # plt.xlim(-network_spec['exit_size']*0.6,network_spec['exit_size']*0.6)
#     # plt.ylim(-network_spec['network_size'][1]*0.6,network_spec['network_size'][1]*0.6)
#     plt.show()

exit_size = 1.1*network_spec['exit_size'] if 'exit_size' in network_spec.keys() else 1.1*network_spec['network_size']
# plt.xlim([-exit_size,exit_size])
plt.ylim([-exit_size,exit_size])
plt.savefig("output/network_{}{}_example.png".format(case,qualifier),dpi=300)
plt.savefig("output/network_{}{}_example.pdf".format(case,qualifier),dpi=300)

# %%
# def AnimationFunction(frame):
#     plt.gca().clear()
#     network.update_network()    #updates and iterate
#     # with PdfPages('network_{}_iterative_frame_{}.pdf'.format(network_type,ii)) as export_pdf:
#     network.draw('intensity')

#     plt.title('Iteration {}'.format(frame))

# fig = plt.figure(1)
# network.update_network()
# network.draw('intensity')
# anim_created = FuncAnimation(fig, AnimationFunction, frames=50)

# writer = PillowWriter(fps=3)
# anim_created.save("network_recursive_{}.gif".format(network_type), writer=writer)

# plt.savefig("network_recursive_{}_frame_50.png".format(network_type),dpi=300)
# for jj in range(0,600):
#     network.update_network()    #updates and iterate
# fig = plt.figure(1)
# plt.gca().clear()
# network.draw('intensity')
# plt.savefig("network_recursive_{}_frame_650.png".format(network_type),dpi=300)
# #
# for jj in range(0,600):
#     network.update_network()    #updates and iterate
# fig = plt.figure(1)
# plt.gca().clear()
# network.draw('intensity')
# plt.savefig("network_recursive_{}_frame_1250.png".format(network_type),dpi=300)
# #
# plt.close()