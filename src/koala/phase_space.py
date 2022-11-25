from .lattice import Lattice
import numpy as np
from numpy import linalg as la


def k_hamiltonian_generator(lattice: Lattice, coloring: np.ndarray,
                            ujk: np.ndarray, J: np.ndarray):
    """Generates a bloch Hamiltonian for a translational system with the unit cell given by lattice for a given k value

    Args:
        lattice (lattice): The lattice object defining the unit cell
        coloring (np.ndarray): a coloring for the lattice (if None then
            you just use J[0])
        J (np.ndarray): the J-values for the x y and z bonds
        ujk (np.ndarray): the edges that define the flux sector

    Returns:
        function: a function that describes the Hamiltonian for a gven k
    """
    j_vals = J[coloring] if coloring is not None else J[0]

    def k_hamiltonian(k: np.ndarray):
        """given a k value, returns a Bloch Hamiltonian

        :param k: point in momentum space
        :type k: np.ndarray
        :return: Hamiltonian
        :rtype: np.ndarray
        """
        H = np.zeros((lattice.n_vertices, lattice.n_vertices), dtype='complex')
        cross = lattice.edges.crossing
        phases = np.sum(cross * k, axis=-1)
        hoppings = 0.5j * j_vals * ujk * np.exp(1j * phases)
        H[lattice.edges.indices[:, 1], lattice.edges.indices[:, 0]] = hoppings
        H[lattice.edges.indices[:, 0],
          lattice.edges.indices[:, 1]] = hoppings.conj()

        return H

    return k_hamiltonian


def analyse_hk(Hk, k_num, return_all_results=False) -> tuple:
    """Given a k-dependent Hamiltonian, this code samples over k-space to find the energy of the ground state and tells you the gap size

    Args:
        Hk (function): K-dependent Hamiltonian
        k_num (int): number of k states in the x and y direction that
            you want to sample, can be a list for x and y separately or an integer for both
        return_all_results (Bool): If true, return the energies over all
            of phase space

    Returns:
        tuple: the ground state energy per site, gap size
    """

    if hasattr(k_num, "__len__"):
        k_values_x = np.arange(k_num[0]) * 2 * np.pi / k_num[0]
        k_values_y = np.arange(k_num[1]) * 2 * np.pi / k_num[1]
        kx_grid, ky_grid = np.meshgrid(k_values_x, k_values_y)
    else:
        k_values = np.arange(k_num) * 2 * np.pi / k_num
        kx_grid, ky_grid = np.meshgrid(k_values, k_values)

    kx = kx_grid.flatten()
    ky = ky_grid.flatten()
    k_list = np.array([[x, y] for x, y in zip(kx, ky)])
    k_number = k_list.shape[0]
    n_states = Hk(np.array([0, 0])).shape[0]
    energies = np.zeros((k_number, n_states // 2))

    for n, ks in enumerate(k_list):
        H = Hk(ks)
        e = la.eigvalsh(H)
        energies[n] = e[:n_states // 2]

    ground_state_per_site = 2 * np.sum(energies) / (k_number * n_states)
    gap_size = np.min(np.abs(energies))

    if return_all_results:
        out = (ground_state_per_site, gap_size, k_list, energies)
    else:
        out = (ground_state_per_site, gap_size)

    return out


def gap_over_phase_space(Hk, k_num: int, return_k_values=False) -> tuple:
    """given a k-dependent hamiltonian, returns an array of the gap size over a k-lattice

    Args:
        Hk (function): k dependent hamiltonian
        k_num (int): number of k states in the x and y direction that
            you want to sample

    Returns:
        np.ndarray
    """

    k_values = np.arange(k_num) * 2 * np.pi / k_num
    KX, KY = np.meshgrid(k_values, k_values)
    k_vals = np.concatenate([KX[:, :, np.newaxis], KY[:, :, np.newaxis]],
                            axis=2)

    def find_gap(k):
        h = Hk(k)
        vals = la.eigvalsh(h)
        return np.min(np.abs(vals))

    gaps = np.apply_along_axis(find_gap, 2, k_vals)
    if return_k_values:
        return gaps, k_vals
    else:
        return gaps
