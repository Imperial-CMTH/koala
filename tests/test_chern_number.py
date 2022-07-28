from koala.pointsets import generate_random
from koala.voronization import generate_lattice
from koala.graph_color import color_lattice
from koala.hamiltonian import generate_majorana_hamiltonian
from numpy import linalg as la
import numpy as np
import matplotlib
from koala import chern_number as cn

def test_chern_and_crosshair():
    # define the lattice system
    number_of_plaquettes = 50
    points = generate_random(number_of_plaquettes)
    lattice = generate_lattice(points)

    # color the lattice
    coloring = color_lattice(lattice)

    # parameters
    ujk = np.full(lattice.n_edges, 1)
    J = np.array([1,1,1])

    # solve system
    H_maj = generate_majorana_hamiltonian(lattice, coloring, ujk, J)
    eigs, vecs = la.eigh (H_maj)
    lowest_diag = np.array([1]*(lattice.n_vertices//2) + [0]*(lattice.n_vertices//2) )
    P = vecs @ np.diag(lowest_diag) @ vecs.conj().T

    # find crosshair / chern number
    crosshair_position = np.array([0.5,0.5])
    crosshair_marker = cn.crosshair_marker(lattice, P, crosshair_position)
    chern_marker = cn.chern_marker(lattice, P)
