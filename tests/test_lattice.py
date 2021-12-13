import numpy as np
from koala import lattice
from koala.lattice import Lattice
from koala.voronization import generate_lattice
from koala.pointsets import generate_random
from koala.example_graphs import *

def test_undirected():
    """
    Also check that two sites are allowed to be connected by multiple edges 
    as long as they differ by a non-contractable loop on the torus
    """
    vertices = np.array([[.2,.2],[.8,.8]])
    edge_indices = np.array([[0,1], [1,0]])
    edge_crossing= np.array([[0,0], [0,1]])

    ordered_edge_indices = np.array([[0,1], [0,1]])
    ordered_edge_crossing = np.array([[0,0], [0,-1]])

    #make an directed lattice
    directed = Lattice(vertices, edge_indices, edge_crossing)
    assert(np.all(directed.edges.indices == edge_indices))
    assert(np.all(directed.edges.crossing == edge_crossing))
    #plot_lattice(lattice, edge_arrows = True)

def test_lattice_class():
    lattice1 = tutte_graph()
    lattice2 = tri_square_pent()
    points = generate_random(30)
    lattice3 = generate_lattice(points)

    assert (len(lattice3.plaquettes) == 30)
