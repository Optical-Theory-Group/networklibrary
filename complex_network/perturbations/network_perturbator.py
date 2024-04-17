# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 14:38:08 2020

@author: Matthew Foreman

Class file for Network object. Inherits from NetworkGenerator class

"""

# setup code logging
import logging
import copy
from typing import Any
import warnings
from tqdm.notebook import tqdm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
from scipy import optimize
from scipy.linalg import dft, null_space
from matplotlib.patches import FancyArrowPatch
import logconfig
from tqdm import tqdm

from .._dict_hdf5 import load_dict_from_hdf5, save_dict_to_hdf5
from .._generator import NetworkGenerator
from complex_network.networks import network_factory
from complex_network.components.link import Link
from complex_network.components.node import Node
from complex_network.poles import pole_finder
from complex_network.networks.network import Network
from ..util import detect_peaks, plot_colourline, update_progress


def get_adjugate(M: np.ndarray) -> np.ndarray:
    """Find the adjugate of a matrix"""
    n_row, n_col = np.shape(M)
    adjugate = np.zeros((n_row, n_col), dtype=np.complex128)

    for i in range(n_row):
        for j in range(n_col):
            modified = np.copy(M)
            modified[i, :] = np.zeros(n_col)
            modified[:, j] = np.zeros(n_row)
            modified[i, j] = 1.0
            adjugate[i, j] = np.linalg.det(modified)
    return adjugate.T


class NetworkPerturbator:
    def __init__(self, network: Network) -> None:
        self.network = network
        self.network_copy = copy.deepcopy(network)

    def node_eigenvalue_perturbation_get_pole(
        self,
        node_index: int,
        eigenvalue_index: int,
        theta: float,
        old_pole: complex,
    ) -> complex:
        """Perturb the network and find the position of the new pole.
        Assumed to be close to the old pole"""

        self.network_copy = copy.deepcopy(self.network)

        # Perform the perturbation
        self.network_copy.perturb_node_eigenvalue(
            node_index, eigenvalue_index, np.exp(1j * theta)
        )
        new_pole = pole_finder.find_pole(self.network_copy, old_pole)
        return new_pole

    def node_eigenvalue_perturbation_get_S(
        self, node_index: int, eigenvalue_index: int, thetas: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        num_perturbations = len(thetas)

        # First index is for S and dS/dtheta
        S_matrices = np.zeros(
            (
                2,
                num_perturbations,
                self.network.internal_vector_length,
                self.network.internal_vector_length,
            ),
            dtype=np.complex128,
        )

        factors = np.exp(1j * thetas)
        for num_factor, factor in enumerate(factors):
            # Make a fresh copy from the original
            self.network_copy = copy.deepcopy(self.network)

            # Perform the perturbation
            self.network_copy.perturb_node_eigenvalue(
                node_index, eigenvalue_index, factor
            )
            new_S, new_dS = self.network_copy.get_S_ii_dS_ii()
            S_matrices[0, num_factor, :, :] = new_S
            S_matrices[1, num_factor, :, :] = new_dS
        return S_matrices

    def node_eigenvalue_perturbation_get_P(
        self, n: complex, k0: complex
    ) -> tuple[np.ndarray, np.ndarray]:
        return self.network_copy.get_P_ii_dP_ii(n, k0)

    def node_eigenvalue_perturbation_iterative(
        self,
        node_index: int,
        eigenvalue_index: int,
        thetas: np.ndarray,
        initial_k0: complex,
        n: complex,
    ):
        # Get S matrices
        S_matrices = self.node_eigenvalue_perturbation_get_S(
            node_index, eigenvalue_index, thetas
        )

        d_theta = thetas[1] - thetas[0]
        k0s_theory = [initial_k0]
        k0s_real = [initial_k0]
        old_k0_theory = initial_k0
        old_k0_real = initial_k0

        for iteration_number, theta in enumerate(tqdm(thetas)):
            # Do real perturbation and find the actual position of the new pole
            new_k0_real = self.node_eigenvalue_perturbation_get_pole(
                node_index, eigenvalue_index, theta, old_k0_real
            )
            k0s_real.append(new_k0_real)
            old_k0_real = new_k0_real

            # Simulated position of new pole
            # Get new P and dP matrix
            # P, dP = self.node_eigenvalue_perturbation_get_P(n, old_k0_theory)

            # # Calculate derivative
            # S = S_matrices[0, iteration_number, :, :]
            # dS = S_matrices[1, iteration_number, :, :]
            # D = np.identity(len(P), dtype=np.complex128) - S @ P
            # adj = get_adjugate(D)
            # top = adj @ dS @ P
            # bottom = adj @ S @ dP
            # derivative = -np.trace(top) / np.trace(bottom)
            # new_k0_theory = old_k0_theory + derivative * d_theta
            # k0s_theory.append(new_k0_theory)
            # old_k0_theory = new_k0_theory

        return k0s_real
