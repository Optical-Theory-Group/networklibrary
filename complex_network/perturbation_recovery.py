import numpy as np
from abc import ABC, abstractmethod
from scipy.linalg import hadamard
from complex_network.networks.network import Network
from typing import Union

class ProbeGenerator(ABC):
    @abstractmethod
    def generate(self, Ne:int,m:int=None)-> list[np.ndarray]:
        "Ne is the size of the entry/exit ports"
        pass

class UnitaryProbe(ProbeGenerator):
    def generate(self,Ne:int,m:int=None,source_indices:list=None)-> list[np.ndarray]:
        """Generates m probe measuremnents such that one entry/exit port is illuminated in each measurement
        
        output = [[1,0,0....],[0,1,0,0...]...]
        
        parameters
            Ne: Total number of ports
            m: Number of measurements taken
            source_indices: indices of the sources you want to illuminate, if not set, it will default to the first m ports"""

        if source_indices:
            if np.any(np.array(source_indices)>Ne):
                raise ValueError("Source indices cannot exceed number of ports")
            probes = []
            for index in source_indices:
                array = np.zeros(Ne)
                array[index] = 1
                probes.append(array)
        else:
            if m>Ne:
                raise ValueError("Number of measurements cannot exceed number of ports")
            probes = []
            for i in range(m):
                array = np.zeros(Ne)
                array[i] = 1
                probes.append(array)

        return probes

class RandomProbe(ProbeGenerator):
    def generate(self,Ne:int,m:int)-> list[np.ndarray]:
        """Generates m random probe measurements
        
        output = m measurements of size Ne
        
        parameters
            Ne: Total number of ports
            m: Number of measurements taken"""
        
        if m>Ne:
            raise ValueError("Number of measurements cannot exceed number of ports")
        probes = []
        for i in range(m):
            array = np.random.uniform(0,2*np.pi,Ne)
            array = np.exp(1j*array)
            probes.append(array)
        return probes
    
class HardamardProbe(ProbeGenerator):
    def generate(self,Ne:int,m:int)-> list[np.ndarray]:
        """Generates m Hadamard probe measurements
        
        output = m measurements of size Ne
        
        parameters
            Ne: Total number of ports
            m: Number of measurements taken"""
        
        if m>Ne:
            raise ValueError("Number of measurements cannot exceed number of ports")
        probes = []
        array = hadamard(Ne)
        for i in range(m):
            probes.append(array[i])
        return probes

class RecoveryMethod(ABC):
    @abstractmethod
    def recover(self,probes:list[np.ndarray],measurements:list[np.ndarray])-> np.ndarray:
        """Recovers the state of the system from the measurements"""
        pass

class LeastSquaresRecovery(RecoveryMethod):
    def __init__(self, rank_multiplier:int = 2):
        """Initializes the least squares recovery method
        
        parameters
            rank_multiplier: A multiplier to determine the rank of the matrix to be used in the recovery
                             for each k defect, the rank of the matrix will be k*rank_multiplier
                             In this system, each defect will change the rank of the matrix by 2"""
        self.rank_multiplier = rank_multiplier


    def recover(self, A:np.ndarray, y:np.ndarray, Ni:int, expected_defects:int)-> np.ndarray:
        """Recovers the state of the system from the measurements

        parameters
            A: Matrix of measurements
            y: Vector of measurements
            Ni: Number of expected defects"""
        
        w_hat, *_ = np.linalg.lstsq(A, y, rcond=None)
        # w_hat is the vector of weights
        # # Isolate the real and  imaginary parts
        w_hat_real = w_hat[:Ni**2].reshape((Ni, Ni))
        w_hat_imag = w_hat[Ni**2:].reshape((Ni, Ni))

        # # Combine the real and imaginary parts
        w_tot = w_hat_real + 1j*w_hat_imag

        rank_trunc = int(np.ceil(self.rank_multiplier*expected_defects))

        if rank_trunc < min(w_tot.shape):
            """When we do an SVD, the diagonal elements are sorted in the descending order.
               We only need to consider the first rank_trunc elements of the diagonal matrix
               There rest of the elements are error terms"""
            U, S, Vh = np.linalg.svd(w_tot, full_matrices=False)
            w_trunc = (U[:,:rank_trunc] * S[:rank_trunc]) @ Vh[:rank_trunc,:]
            return w_trunc
        else:
            # if the rank is too high, we just return the original matrix
            return w_tot

class BasisPursuitRecovery(RecoveryMethod):
    def __init__(self, rank_multiplier:int = 2):
        """Initializes the basis pursuit recovery method
        
        parameters
            rank_multiplier: A multiplier to determine the rank of the matrix to be used in the recovery
                             for each k defect, the rank of the matrix will be k*rank_multiplier
                             In this system, each defect will change the rank of the matrix by 2"""
        self.rank_multiplier = rank_multiplier

    def recover(self, A:np.ndarray, y:np.ndarray, Ni:int, expected_defects:int)-> np.ndarray:
        """Recovers the state of the system from the measurements

        parameters
            A: Matrix of measurements
            y: Vector of measurements
            Ni: Number of expected defects"""
            
        pass
    
class DefectDetector:
    def __init__(self,
                 probe_generator:ProbeGenerator,
                 recovery_method:RecoveryMethod):
        """Initializes the defect detector
        parameters
            probe_generator: The probe generator to be used
            recovery_method: The recovery method to be used"""
        self.probe_generator = probe_generator
        self.recovery_method = recovery_method

    def _assemble_A_y(self,L,Q,reference_S,probes,perturbed_signal):
        A_rows, Y_vec  = [], []
        for signal,output in zip(probes,perturbed_signal):
            # Find the difference between the perturbed signal and the reference signal
            delta_o = output - reference_S@signal
            A_j = np.kron((Q@signal).T, L)
            A_rows.extend([np.hstack([A_j.real,-A_j.imag]),np.hstack([A_j.imag,A_j.real])])
            Y_vec.extend([delta_o.real,delta_o.imag])

        A = np.vstack(A_rows)
        Y = np.hstack(Y_vec)
        return A, Y
        
    def detect_defects(self,unperturbed_network:Network,
                            perturbed_network:Network,
                            wavevector: Union[float,complex,list[float],list[complex]],
                            expected_defects:int,
                            m_per_defect:int,
                            top_predictions:int=5):
        """Detects defects in the network using the given probe generator and recovery method
        parameters
            unperturbed_network: The unperturbed network (Network object)
            perturbed_network: The perturbed network (Network object)
            wavevector: The wavevector to be used, can be a single value or a list of values
            expected_defects: The expected number of defects in the network
            m_per_defect: The number of measurements per defect
            top_predictions: The number of top predictions to be returned"""
        
        # Make sure the wavevector is a list
        if isinstance(wavevector, (int, float,complex)):
            wavevector = [wavevector]

        # Probe generation
        S0 = unperturbed_network.get_S_ee(wavevector[0])
        Ne = S0.shape[0]
        probes = self.probe_generator.generate(Ne, m_per_defect*expected_defects)

        # A_matrix and Y vector assembly
        A_blocks, Y_blocks = [], []
        for k in wavevector:
            S_ref = unperturbed_network.get_S_ee(k)
            outputs =[perturbed_network.get_S_ee(k)@probe for probe in probes]
            P_ei = unperturbed_network.get_P_ei(k)
            P_ie = P_ei.T
            S_ii = unperturbed_network.get_S_ii(k)
            P_ii = unperturbed_network.get_P_ii(k)
            Ni = S_ii.shape[0]

            D = np.linalg.inv(np.eye(Ni)-S_ii@P_ii)
            L,Q = P_ei@D, D@P_ie
            A_k, Y_k = self._assemble_A_y(L,Q,S_ref,probes,outputs)
            A_blocks.append(A_k)
            Y_blocks.append(Y_k)
        
        A_all = np.vstack(A_blocks)
        Y_all = np.hstack(Y_blocks)

        # total number of internal links
        num_links = unperturbed_network.internal_vector_length

        W_tot = self.recovery_method.recover(A_all, Y_all, num_links, expected_defects)

        scores = self._calculate_scores(W_tot, unperturbed_network)

        return dict(sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_predictions])
    
    def _calculate_scores(self, W_tot:np.ndarray, unperturbed_network:Network)-> dict:
        """Calculates the scores of the defects in the network
        parameters
            W_tot: The total matrix of weights
            unperturbed_network: The unperturbed network (Network object)"""
        
        scores  = {}
        idx_A = unperturbed_network.internal_link_indices_A_to_B
        idx_B = unperturbed_network.internal_link_indices_B_to_A

        for link, ia, ib in zip(unperturbed_network.internal_links, idx_A, idx_B):
            
            scores[link.index] = (np.linalg.norm(W_tot[ia]**2) + np.linalg.norm(W_tot[:,ia]**2)
                                  + np.linalg.norm(W_tot[ib]**2) + np.linalg.norm(W_tot[:,ib]**2)).real
        return scores