from matplotlib import pyplot as plt
import koala.phase_space as ps
from koala.example_graphs import *
import koala.plotting as pl
from koala.graph_color import color_lattice
from matplotlib import cm
from koala.flux_finder import fluxes_from_bonds,fluxes_to_labels
from numpy import linalg as la
from koala.hamiltonian import generate_majorana_hamiltonian


def test_phase_and_regular_energies():
    # create the regular quantum system
    # generate the lattice and fluxes
    system_size = 2

    lattice = generate_hex_square_oct(system_size)
    coloring = color_lattice(lattice)
    j_vals = np.array([1,1,1])
    ujk = np.ones(lattice.n_edges)
    # flip a few fluxes
    flip_selection = slice(4,23,6)
    ujk[flip_selection] = -1
    # generate the Hamiltonian and solve
    h_big = generate_majorana_hamiltonian(lattice,coloring, ujk, j_vals)
    energies1 = la.eigvalsh(h_big)


    # create the phase space quantum system

    # generate a minimal lattice unit cell
    lattice = generate_hex_square_oct(1)
    coloring = color_lattice(lattice)
    ujk = np.ones(lattice.n_edges)
    j_vals = np.array([1,1,1])

    ujk[flip_selection] = -1

    # pick your k states
    k_values = np.arange(system_size)*2*np.pi/system_size
    KX,KY = np.meshgrid(k_values,k_values)
    kx = KX.flatten()
    ky = KY.flatten()

    # generate the H(k) bloch Hamiltonian
    h_k = ps.k_hamiltonian_generator(lattice,coloring, ujk, j_vals)

    # solve the Hamiltonian at every k value
    energies2 = np.array([])
    for k in zip(kx,ky):
        e = la.eigvalsh(h_k(k))
        energies2 = np.append( energies2,e )


    e_dif = energies1 - np.sort(energies2)
    assert(np.max(np.abs(e_dif)) <1e-10 )


def test_phase_space_analysers():
    system_size = 5
    lattice = generate_hex_square_oct(1)
    coloring = color_lattice(lattice)
    ujk = np.ones(lattice.n_edges)
    j_vals = np.array([1,1,1])

    h_k = ps.k_hamiltonian_generator(lattice,coloring, ujk, j_vals)

    ps.analyse_hk(h_k, system_size)
    ps.gap_over_phase_space(h_k, system_size)