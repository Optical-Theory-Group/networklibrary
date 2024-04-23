# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 09:31:03 2023

@author: mforeman

Generates a single random network and calculates the |S| complex spectrum 
as a specified node is perturbed

"""
import os

import dill
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FFMpegFileWriter
from matplotlib.collections import LineCollection
from scipy.optimize import Bounds

from complex_network.networks.network import Network
from complex_network.utils import update_progress

    # #################################################
# %% define simulation parameters
# #################################################
wavelength = 500e-9
k=(2*np.pi)/wavelength
n=1
seed = 65565

case ='delaunay'
if case == 'delaunay':
    # delaunay slab
    network_type = 'delaunay'
    network_spec = {'internal_nodes': 80,
                    'exit_nodes': 30,
                    'network_size': (100e-6),
                    'exit_size': 110e-6,
                    'wavenumber': k,
                    'refractive_index': n,
                    'shape':'circular'
                    }
    
    node_number = 9 # which node do we peturb?
    lmin = 700e-9
    lmax = 701e-9
    Qmin = 250# 25000
    kimax = 0*k/Qmin
    kimin = -6000#k/Qmin
 
# node specification

scattering_loss = 0
node_spec = {'S_mat_type': 'unitary_cyclic',
             'scat_loss': scattering_loss,
             # leave 'delta' out to get random phases across all nodes
             }
length_scal_factor = 1

# #################################################
# %%parameters for data files
# #################################################
runid = 'perturb_node_{}_{}_i{}_e{}_dim{}_lf{}_merge'.format(network_type,
                                        node_spec['S_mat_type'],
                                        network_spec['internal_nodes'],
                                        network_spec['exit_nodes'],
                                        np.array(network_spec['network_size'])*1e6,
                                        length_scal_factor)
outputfiles = True
calcdetS   = False
calc_modes = False
refine_roots = False

# make data folders if they dont exist
datadir = 'data/{}'.format(runid)
outputdir = 'output/{}'.format(runid)
if not os.path.exists(datadir):
    os.makedirs(datadir)
if not os.path.exists(outputdir):
    os.makedirs(outputdir)

# #################################################
# %% Construct coordinates in complex plane
# #################################################
kmin = (2*np.pi/lmax) + 1j*kimin
kmax = (2*np.pi/lmin) + 1j*kimax
Nkr = 150
Nki = 150

krmin = np.real(kmin)
kimin = np.imag(kmin)

krmax = np.real(kmax)
kimax = np.imag(kmax)

kr = np.linspace(krmin,krmax,Nkr)
ki = np.linspace(kimin,kimax,Nki)

# #################################################
# %% Initialise the network
# #################################################
network = Network(network_type,network_spec,node_spec,seed_number=seed)   #CREATE A NETWORK OBJECT

input_amp = [0]* network_spec['exit_nodes']
input_amp[0] = 1
internal_nodes = network.internal_nodes
external_nodes = network.exit_nodes
network.initialise_network(node_spec,input_amp)   # input my scattering matrix and the input amplitudes

network.draw()
ax = plt.gca()
ax.set_aspect('equal')

# #####################################################################
# %% Set up properties for tuning properties of a single node
# #####################################################################

parameter_values = np.linspace(0,2*np.pi,150) # eigenvalue phase to scan over
index = 0 # initial value of delta2
nodeindextochange = 1

# reset node to initial parameter setting
dvec = np.zeros((network.nodes[node_number].n_connect))
dvec[nodeindextochange] = parameter_values[index]

new_node_spec = {'S_mat_type': 'unitary_cyclic',
                 'scat_loss': 0,
                 'delta':dvec,
                 }

# reset the scattering matrix of the given node
network.initialise_node_S_mat(node_number,**node_spec)

# Initialise arrays to store results
parameter_scan_results = {}
rcoeff = np.zeros((len(parameter_values)),dtype=np.complex_)
detS = np.zeros( (Nkr, Nki, len(parameter_values)),dtype=np.float64)

# #################################################
# %% Tune node and calculate detS landscape and
#    numerically find poles
# #################################################
indmin = 0
indmax = 1#len(parameter_values)

# %%
for index in range(indmin,indmax):
    update_progress((index-indmin) / (indmax-indmin), '{}/{}   '.format((index-indmin),(indmax-indmin)))
    
    results = {}
    dvec = np.zeros((network.nodes[node_number].n_connect))
    
    dvec[nodeindextochange] = parameter_values[index] # for four connections delta2 seems to give range of 0 <= |r| <= 1
    results['delta'] = dvec
    results['parameter'] = parameter_values[index]
    results['index'] = index
    new_node_spec = {'S_mat_type': 'unitary_cyclic',
                     'scat_loss': 0,
                     'delta':dvec,
                     }

    # reset the scattering matrix of the given node
    network.initialise_node_S_mat(node_number,**new_node_spec)
    rcoeff[index] = network.nodes[node_number].S_mat[nodeindextochange,nodeindextochange]
    results['rcoeff'] = rcoeff[index]

    # # %% -- calculate determinant landscape
    filename = 'data/{}/{}_index_results_{}.pkl'.format(runid,runid,index)
    # check to see if data file exists and we don't want to calculate detS
    if ((not calcdetS) and os.path.exists(filename)) : 
        with open(filename, 'rb') as fpkl:
            temporary_results = dill.load(fpkl)
            detS_temp = temporary_results['detS']
            kpeaks_temp = temporary_results['kpeaks_coarse']
            newk = np.zeros((len(kpeaks_temp)),dtype='complex128') if 'kpeaks_fine' not in temporary_results.keys() else temporary_results['kpeaks_fine']
    else: # calculate detS landscape
        kr,ki,detS_temp,kpeaks_temp = network.calc_det_S(kmin,kmax,Nkr,Nki,
                                                          progress_bar_text='{}/{}'.format((index-indmin),(indmax-indmin)))
        newk = np.zeros((len(kpeaks_temp)),dtype='complex128') 
        
        # if (os.path.exists(filename)) : # if we are updating detS
        #     with open(filename, 'rb') as fpkl:
        #         results = dill.load(fpkl)
        
    detS[:,:,index] = detS_temp
    results['detS'] = detS_temp
    results['kpeaks_fine'] = newk
    
    # network.inverse_scattering_matrix_direct()
    # %% -- refine estimate of new poles
    
    if refine_roots:
        # newk = np.zeros((len(kpeaks_temp)),dtype='complex128')
        # mtd = 'L-BFGS-B'#
        mtd = 'Nelder-Mead'
        opts = {'disp':True,
                'xatol': 1e-3,
                'maxiter':1500,
                'maxfev':3000}
        dkr = (krmax - krmin)/Nkr
        dki = (kimax - kimin)/Nki
        dkrbound = 11 # number of bounding pixels 
        dkibound = 11
        for kindex,k0 in enumerate(kpeaks_temp):
            k0r = np.real(k0)
            k0i = np.imag(k0)
            bounds = [(k0r-dkrbound*dkr , k0r+dkrbound*dkr),
                      (k0i-dkibound*dki , k0i+dkibound*dki)]
            newk[kindex]  = network.find_pole(k0,mtd,opts,bounds)
            print([index,kindex,2*np.pi/np.real(k0)*1e9, np.imag(k0), 
                   2*np.pi/np.real(newk[kindex])*1e9,np.imag(newk[kindex])])
                
    # %% -- find poles and associated input mode distributions
    kpoles0 = np.array(newk,dtype='complex128')
    pole_inputs = np.zeros(shape=(network.exit_nodes,len(kpoles0)),dtype='complex128')
    
    if calc_modes:
        for kindex,k0 in enumerate(kpoles0):
            # network.reset_network(k=np.real(k0))
            network.reset_network(k=k0)
    
            # find eigenvectors/values of scattering matrix
            sm,sorted_nodes = network.scattering_matrix_direct()
            val, vec = np.linalg.eig(sm)
    
            zerovals = np.argmin(1./np.abs(val))
            pole_inputs[:,kindex] = vec[:,zerovals]
    
            network.reset_network(k=np.real(k0))
            network.reset_network(input_amp=list(vec[:,zerovals]))
            network.run_network()
            # plt.figure()
            # network.draw('intensity')
            # plt.title(np.round(2*np.pi/k0 * 1e9,0))
            # plt.savefig("output/{}_eigenmode_{}_{}_int_{}_ext_{}_loss_{}.png".format(runid,index,network_type,internal_nodes,external_nodes,scattering_loss),dpi=300)
            # plt.close()

    results['pole_inputs'] = pole_inputs
    results['kpeaks_coarse'] = kpeaks_temp
    results["kpeaks_fine"] = np.array(kpoles0,dtype='complex128')
    results['num_peaks'] = len(kpeaks_temp)
    results["kanal"] = results["kpeaks_fine"]
    results["dkanal"] = 0*results["kpeaks_fine"]
    results["dkanal_startpoint"] = results["kpeaks_fine"]

    # store result
    parameter_scan_results[parameter_values[index]] = results
    with open(filename, 'wb') as fpkl:
        dill.dump(results,fpkl)

# %% Plot detS landscape
fig = plt.figure()
plt.pcolor(2*np.pi/kr * 1e9,ki,np.log10(detS_temp.T),shading='auto',cmap=
            sns.color_palette("rocket", as_cmap=True))

ind = 0
# for kpeak in kpeaks_temp:#kpoles0:
#     if np.real(kpeak) > 0:
#     # plt.scatter(2*np.pi/np.real(kpeaks0)*1e9,np.imag(kpeaks0), marker='s', s=50, color='yellow')
#         plt.scatter(2*np.pi/np.real(kpeak)*1e9,np.imag(kpeak), marker='x', s=50, color='red')
#         plt.text( 2*np.pi/np.real(kpeak)*1e9,np.imag(kpeak), "{}".format(ind))
#     ind += 1

plt.xlabel("Wavelength (nm)")
plt.ylabel("Im[k]")
plt.title('log(|S|)')
plt.xlim(lmin * 1e9,lmax * 1e9)
# plt.ylim(kimin,kimax)
plt.show()

# plt.savefig("output/tier1_detS.pdf".format(runid,runid,index,network_type,internal_nodes,external_nodes,scattering_loss),dpi=300)


raise ValueError
# #################################################
# %% Tune node and calculate analytic pole shifts
# #################################################
dk = 1e-3*(1e-3 + 1j*1e-3) # small change to get numerical derivative

for index in range(indmin+1,indmax):
    update_progress(index/ (indmax - indmin-1),'Calculating analytic pole shifts...')
    # get poles for previous parameter value
    refpoleindex =  index - 1 #0
    with open('data/{}/{}_index_results_{}.pkl'.format(runid,runid,refpoleindex), 'rb') as fpkl:
        previous_results = dill.load(fpkl)

    kpoles_iter = previous_results['kpeaks_fine']

    # retrieve previous stored variables for current
    with open('data/{}/{}_index_results_{}.pkl'.format(runid,runid,index), 'rb') as fpkl:
        results = dill.load(fpkl)

    # reset the scattering matrix of the given node
    dvec = np.zeros((network.nodes[node_number].n_connect))
    dvec[nodeindextochange] = parameter_values[index] 
    new_node_spec = {'S_mat_type': 'unitary_cyclic',
                     'scat_loss': 0,
                     'delta':dvec,
                     }

    # reset the scattering matrix of the given node
    network.initialise_node_S_mat(node_number,**new_node_spec)

    # calculate resonance shifts
    dkanal_temp = np.zeros((len(kpoles_iter)),dtype=np.complex_)
    for kindex,k0 in enumerate(kpoles_iter):
        # calcalate scattering matrix at previous position of poles
        network.reset_network(k=k0)
        sm1,sorted_nodes = network.scattering_matrix_direct()
        ism1 = np.linalg.inv(sm1)

        network.reset_network(k= k0+dk )
        sm2,sorted_nodes = network.scattering_matrix_direct()
        ism2 = np.linalg.inv(sm2)

        # print(- 1/np.trace(np.matmul( sm1, (ism2-ism1) / np.abs(dk))))
        dkanal_temp[kindex] = - 1/np.trace(np.matmul( sm1, (ism2-ism1) / dk))

    results["dkanal"] = dkanal_temp
    results["dkanal_startpoint"] = kpoles_iter
    results["kanal"] = kpoles_iter + dkanal_temp

    # store result
    parameter_scan_results[parameter_values[index]] = results
    with open('data/{}/{}_index_results_{}.pkl'.format(runid,runid,index), 'wb') as fpkl:
        dill.dump(results,fpkl)


# %% Plot det S landscape for each parameter value
kpoles_new = {}
kpoles_new2 = {}
kpoles_new3 = {}
kpoles_new4 = {}
dkpoles_new = {}
rcoeff_new = [0]*len(parameter_values)

for index in range(indmin,indmax):
    with open('data/{}/{}_index_results_{}.pkl'.format(runid,runid,index), 'rb') as fpkl:
        results = dill.load(fpkl)

    if 'rcoeff' in results.keys():              rcoeff_new[index] = results['rcoeff']
    if 'kpeaks_coarse' in results.keys():       kpoles_new[index] = results['kpeaks_coarse']
    if 'kpeaks_fine' in results.keys():         kpoles_new2[index] = results['kpeaks_fine']
    # if index != 0:
    if 'dkanal_startpoint' in results.keys():   kpoles_new3[index] = results['dkanal_startpoint']
    kpoles_new4[index] = results['kanal']
    dkpoles_new[index] = results['dkanal']

kpoles_all_coarse = np.concatenate([val for val in kpoles_new.values() ]).ravel()
kpoles_all_fine   = np.concatenate([val for val in kpoles_new2.values()]).ravel()
kpoles_anal_start = np.concatenate([val for val in kpoles_new3.values()]).ravel()
kpoles_anal_end   = np.concatenate([val for val in kpoles_new4.values()]).ravel()
dkanal   = np.concatenate([val for val in dkpoles_new.values()]).ravel()

kpoles_anal_x1 = 2*np.pi/np.real(kpoles_anal_start) * 1e9
kpoles_anal_x2 = 2*np.pi/np.real(kpoles_anal_end) * 1e9
kpoles_anal_y1 = np.imag(kpoles_anal_start)
kpoles_anal_y2 = np.imag(kpoles_anal_end)
linecol = [ [(xx1,yy1), (xx2,yy2) ] for (xx1,xx2,yy1,yy2) in zip(kpoles_anal_x1,kpoles_anal_x2,kpoles_anal_y1,kpoles_anal_y2)]


if outputfiles:
    fig = plt.figure()
    plt.plot(parameter_values,np.abs(rcoeff_new))
    plt.xlabel("$\delta$")
    plt.ylabel("$|r|$")
    plt.savefig("output/{}/{}_rcoeff_{}_int_{}_ext_{}_loss_{}.png".format(runid,runid,index,network_type,internal_nodes,external_nodes,scattering_loss),dpi=300)
    fig.clear()


if True: # outputfiles
    fig = plt.figure()
    for index in range(indmin,indmax):
        print(index)
        # retrieve previous stored variables for current
        with open('data/{}/{}_index_results_{}.pkl'.format(runid,runid,index), 'rb') as fpkl:
            results = dill.load(fpkl)

        plt.pcolor(2*np.pi/kr * 1e9,ki,np.log10(results['detS'].T),shading='auto',cmap=
                sns.color_palette("rocket", as_cmap=True))

        if outputfiles: plt.savefig("output/{}/{}_detS_delta_val_{}_{}_int_{}_ext_{}_loss_{}.png".format(runid,runid,index,parameter_values[index],network_type,internal_nodes,external_nodes,scattering_loss),dpi=300)
        fig.clear()

        plt.pcolor(2*np.pi/kr * 1e9,ki,np.log10(np.array(results['detS'].T)),shading='auto',cmap=
            sns.color_palette("rocket", as_cmap=True))

        plt.ylim([np.min(ki), np.max(ki)])
        plt.xlim([np.min(2*np.pi/np.real(kr)* 1e9), np.max(2*np.pi/np.real(kr)* 1e9)])
        # plt.xlim([445, 450])
        # plt.ylim([np.min(ki),-40000])

       # # plt.plot(2*np.pi/np.real(kpoles_all_coarse)* 1e9,np.imag(kpoles_all_coarse),'.',markersize=1)
        plt.plot(2*np.pi/np.real(kpoles_all_fine)* 1e9,np.imag(kpoles_all_fine),'b.',markersize=.3)

        line_segments = LineCollection(linecol,
                                        linewidth=0.50,
                                        color='w',
                                        )
        plt.gca().add_collection(line_segments)
        plt.title('d = {:0.3f} * 2$\pi$'.format(parameter_values[index]/(2*np.pi)))

        if outputfiles: plt.savefig("output/{}/{}_detS_delta_val_{}_{}_int_{}_ext_{}_loss_{}.png".format(runid,runid,index,parameter_values[index],network_type,internal_nodes,external_nodes,scattering_loss),dpi=1200)
        # fig.clear()

# %% Make animation of landscape

fig = plt.figure()
## %%
def AnimationFunction(frame):
    update_progress(frame/len(parameter_values),'Generating animation...')
    with open('data/{}/{}_index_results_{}.pkl'.format(runid,runid,frame), 'rb') as fpkl:
        results = dill.load(fpkl)

    ax = plt.gca()
    ax.clear()
    plt.pcolor(2*np.pi/kr * 1e9,ki,np.log10(np.array(results['detS'].T)),shading='auto',cmap=
            sns.color_palette("rocket", as_cmap=True))



    plt.ylim([np.min(ki), np.max(ki)])
    plt.xlim([np.min(2*np.pi/np.real(kr)* 1e9), np.max(2*np.pi/np.real(kr)* 1e9)])
    # plt.xlim([440, 460])
    # plt.ylim([-00000,-55000])

    # plt.xlim([445, 450])
    # plt.ylim([np.min(ki),-40000])

   # # plt.plot(2*np.pi/np.real(kpoles_all_coarse)* 1e9,np.imag(kpoles_all_coarse),'.',markersize=1)
    plt.plot(2*np.pi/np.real(kpoles_all_fine)* 1e9,np.imag(kpoles_all_fine),'b.',markersize=.3)

    line_segments = LineCollection(linecol,
                                    linewidth=0.50,
                                    color='w',
                                    )
    plt.gca().add_collection(line_segments)



    plt.title('d = {:0.3f} * 2$\pi$'.format(parameter_values[frame]/(2*np.pi)))
    # fig.show()

if True: # outputfiles
    # writer = PillowWriter(fps=8,
    #                   bitrate=300)
    # anim_created = FuncAnimation(fig, AnimationFunction, frames=95)#(len(parameter_values)-3))

    # anim_created.save("output/{}/{}_detS_markers_{}_int_{}_ext_{}_loss_{}.gif".format(runid,runid,network_type,internal_nodes,external_nodes,scattering_loss),
    #                   writer=writer,
    #                   dpi=300)

    writer = FFMpegFileWriter(fps=10)
    # writer.setup(fig,
    #                   "output/{}/{}_detS_markers_{}_int_{}_ext_{}_loss_{}.mp4".format(runid,runid,network_type,internal_nodes,external_nodes,scattering_loss),
    #                     dpi=300)
    with writer.saving(fig,
                       "output/{}/{}_detS_markers_{}_int_{}_ext_{}_loss_{}.mp4".format(runid,runid,network_type,internal_nodes,external_nodes,scattering_loss),
                       dpi=600):
        for frame in range(0,len(parameter_values)):
            AnimationFunction(frame)
            writer.grab_frame()



# from PIL import Image,ImageFilter

# images = []
# aname = "output/{}/{}_detS_{}_int_{}_ext_{}_loss_{}.gif".format(runid,runid,network_type,internal_nodes,external_nodes,scattering_loss)

# for frame in range(0,len(parameter_values)):
#     update_progress(frame/len(parameter_values),'Generating animation...')
#     fname = "output/{}/{}_detS_delta_val_{}_{}_int_{}_ext_{}_loss_{}.png".format(runid,runid,frame,parameter_values[frame],network_type,internal_nodes,external_nodes,scattering_loss)

#     exec('a'+str(frame)+'=Image.open("'+fname+'")')
#     images.append(eval('a'+str(frame)))
# images[0].save(aname,
#             save_all=True,
#             append_images=images[1:],
#             duration=100,

# # %%
# fig = plt.figure(figsize=(8,10))

# for kindex in range(0,len(kpoles0)):
#     fig.add_subplot( len(kpoles0) , 2, 2*kindex + 1)
#     plt.plot(2*np.pi/(np.real(dknum[:,kindex])+np.real(kpoles0[kindex])) * 1e9)
#     plt.plot(2*np.pi/(np.real(dkanal[:,kindex])+np.real(kpoles0[kindex])) * 1e9)
#     # plt.ylim([np.min(2*np.pi/kr * 1e9), np.max(2*np.pi/kr
#             loop=0) * 1e9)])
#     fig.add_subplot( len(kpoles0) , 2, 2*kindex + 2)
#     plt.plot(np.imag(dknum[:,kindex]))
#     plt.plot(np.imag(dkanal[:,kindex]))
#     if kindex == 0:plt.legend(['num','anal'])

