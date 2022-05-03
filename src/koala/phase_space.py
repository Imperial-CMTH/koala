from .lattice import Lattice
import numpy as np
from numpy import linalg as la

def k_hamiltonian_generator(lattice:Lattice, coloring:np.ndarray, ujk: np.ndarray, J: np.ndarray):
    """Generates a bloch Hamiltonian for a translational system with the unit cell given by lattice for a given k value

    :param lattice: The lattice object defining the unit cell
    :type lattice: lattice
    :param coloring: a coloring for the lattice (if None then you just use J[0])
    :type coloring: np.ndarray
    :param J: the J-values for the x y and z bonds
    :type J: np.ndarray
    :param ujk: the edges that define the flux sector
    :type ujk: np.ndarray
    :return: a function that describes the Hamiltonian for a gven k
    :rtype: function
    """
    j_vals = J[coloring] if coloring is not None else J[0]

    def k_hamiltonian(k: np.ndarray):
        """given a k value, returns a Bloch Hamiltonian

        :param k: point in momentum space
        :type k: np.ndarray
        :return: Hamiltonian
        :rtype: np.ndarray
        """ 
        H = np.zeros((lattice.n_vertices,lattice.n_vertices), dtype = 'complex')
        cross = lattice.edges.crossing
        phases = np.sum(cross*k, axis = -1)
        hoppings = 0.5j*j_vals*ujk*np.exp(1j*phases)
        H[lattice.edges.indices[:,1], lattice.edges.indices[:,0]] = hoppings
        H[lattice.edges.indices[:,0], lattice.edges.indices[:,1]] = hoppings.conj()

        return H

    return k_hamiltonian


def analyse_hk(Hk, k_num: int) -> tuple:
    """Given a k-dependent Hamiltonian, this code samples over k-space to find the energy of the ground state and tells you the gap size

    :param Hk: K-dependent Hamiltonian  
    :type Hk: function
    :param k_num: number of k states in the x and y direction that you want to sample   
    :type k_num: int
    :return: the ground state energy per site, gap size 
    :rtype: tuple
    """

    k_values = np.arange(k_num)*2*np.pi/k_num
    KX,KY = np.meshgrid(k_values,k_values)
    kx = KX.flatten()
    ky = KY.flatten()
    k_list = np.array([[x,y] for x,y in zip(kx,ky)])
    k_number = k_list.shape[0]
    n_states = Hk(np.array([0,0])).shape[0]
    energies = np.zeros((k_number,n_states//2))

    for n, ks in enumerate(k_list):
        H = Hk(ks)
        e = la.eigvalsh(H)
        energies[n]= e[:n_states//2]

    ground_state_per_site = 2*np.sum(energies)/(k_number*n_states)
    gap_size = np.min(np.abs(energies))

    return ground_state_per_site, gap_size


def gap_over_phase_space(Hk, k_num: int) -> tuple:
    """given a k-dependent hamiltonian, returns an array of the gap size over a k-lattice

    :param Hk: k dependent hamiltonian
    :type Hk: function
    :param k_num: number of k states in the x and y direction that you want to sample   
    :type k_num: int
    :return: 
    :rtype: np.ndarray
    """

    k_values = np.arange(k_num)*2*np.pi/k_num
    KX,KY = np.meshgrid(k_values,k_values)
    k_vals = np.concatenate([KX[:,:,np.newaxis],KY[:,:,np.newaxis]],axis=2)

    def find_gap(k):
        h = Hk(k)
        vals = la.eigvalsh(h)
        return np.min(np.abs(vals))
    gaps = np.apply_along_axis(find_gap,2,k_vals)
    return gaps