from .lattice import Lattice
import numpy as np

def k_hamiltonian_generator(lattice:Lattice, coloring:np.ndarray, J: np.ndarray, ujk: np.ndarray):

    j_vals = J[coloring]

    def k_hamiltonian(k: np.ndarray):
        H = np.zeros((lattice.n_vertices,lattice.n_vertices), dtype = 'complex')
        for n, edge_indices in enumerate(lattice.edges.indices):
            cross = lattice.edges.crossing[n]
            H[edge_indices[1],edge_indices[0]] = 2  * j_vals[n] * ujk[n] * np.exp(1j*np.sum(cross*k))
        return (H - H.T)*1j

    return k_hamiltonian

    