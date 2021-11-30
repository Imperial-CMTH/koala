import numpy as np

from koala.plotting import plot_vertex_indices, plot_degeneracy_breaking, plot_lattice
from koala.graph_utils import vertex_neighbours, clockwise_edges_about
from koala.voronization import Lattice

g = Lattice(
    vertices = np.array([[0.5,0.5], [0.1,0.1], [0.5,0.9], [0.9,0.1]]),
    edge_indices = np.array([[0,1],[0,2],[0,3]]),
    edge_crossing = np.array([[0,0],[0,0],[0,0]]),
    )

def test_vertex_neighbours():
    plot_lattice(g, edge_arrows = True, edge_index_labels = True, vertex_labels = 0)
    plot_vertex_indices(g)
    plot_degeneracy_breaking(0, g)

    vertex_indices, edge_indices = vertex_neighbours(0, g.edges.indices)
    assert(np.all(vertex_indices == (np.array([1, 2, 3]),)))
    assert(np.all(edge_indices == np.array([0,1,2])))

def test_clockwise_edges_about():
    assert(np.all(clockwise_edges_about(vertex_i = 0, g = g) == np.array([1,0,2])))