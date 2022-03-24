from .lattice import Lattice
import numpy as np

def k_hamiltonian_generator(lattice:Lattice, coloring:np.ndarray, ujk: np.ndarray, J: np.ndarray):
    """Generates a bloch Hamiltonian for a translational system with the unit cell given by lattice for a given k value

    :param lattice: The lattice object defining the unit cell
    :type lattice: lattice
    :param coloring: a colouring for the lattice (if None then you just use J[0])
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