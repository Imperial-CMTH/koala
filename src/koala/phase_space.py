from .lattice import Lattice
import numpy as np

def k_hamiltonian_generator(lattice:Lattice, coloring:np.ndarray, J: np.ndarray, ujk: np.ndarray):
    """Generates a bloch Hamiltonian for a translational system with the unit cell given by lattice for a given k value

    :param lattice: The lattice object defining the unit cell
    :type lattice: lattice
    :param coloring: a colouring for the lattice
    :type coloring: np.ndarray
    :param J: the J-values for the x y and z bonds
    :type J: np.ndarray
    :param ujk: the edges that define the flux sector
    :type ujk: np.ndarray
    :return: a function that describes the Hamiltonian for a gven k
    :rtype: function
    """
    j_vals = J[coloring]

    def k_hamiltonian(k: np.ndarray):
        """given a k value, returns a Bloch Hamiltonian

        :param k: point in momentum space
        :type k: np.ndarray
        :return: Hamiltonian
        :rtype: np.ndarray
        """ 
        H = np.zeros((lattice.n_vertices,lattice.n_vertices), dtype = 'complex')
        for n, edge_indices in enumerate(lattice.edges.indices):
            cross = lattice.edges.crossing[n]
            H[edge_indices[1],edge_indices[0]] = 2  * j_vals[n] * ujk[n] * np.exp(1j*np.sum(cross*k))
        return (H - H.T)*1j

    return k_hamiltonian