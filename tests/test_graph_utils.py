import numpy as np
import pickle as pkl

from koala.plotting import plot_vertex_indices, plot_degeneracy_breaking, plot_lattice
from koala import example_graphs as eg
from koala.pointsets import uniform
from koala.graph_utils import vertex_neighbours, clockwise_edges_about, adjacent_plaquettes,remove_trailing_edges, make_dual, vertices_to_polygon
from koala.lattice import Lattice
from koala.voronization import generate_lattice


g = Lattice(
    vertices = np.array([[0.5,0.5], [0.1,0.1], [0.5,0.9], [0.9,0.1]]),
    edge_indices = np.array([[0,1],[0,2],[0,3]]),
    edge_crossing = np.array([[0,0],[0,0],[0,0]]),
    )


def test_vertex_neighbours():
    plot_lattice(g, edge_arrows = True, edge_index_labels = True, vertex_labels = 0)
    plot_vertex_indices(g)
    plot_degeneracy_breaking(0, g)

    vertex_indices, edge_indices = vertex_neighbours(g, 0)
    assert np.all(vertex_indices == (np.array([1, 2, 3]),))
    assert np.all(edge_indices == np.array([0,1,2]))


def test_clockwise_edges_about():
    assert np.all(clockwise_edges_about(vertex_index = 0, g = g) == np.array([1,0,2]))


def test_adjacent_plaquettes_function():
    l = eg.tri_square_pent()
    
    p, e = adjacent_plaquettes(l, 1)

    assert np.all(p == [0,2])
    assert np.all(e == [2,5])


def test_trailing_edges():
    with open('tests/data/trailing_lattice_example.pickle', 'rb') as f:
        lattice = pkl.load(f)
    l_out = remove_trailing_edges(lattice)
    assert l_out.n_edges == 13
    assert l_out.n_vertices == 11
    assert l_out.n_plaquettes == 3

def test_dual():
    points = uniform(30)
    weird_graphs = [
                eg.tri_square_pent(),
                eg.two_triangles(),
                eg.tutte_graph(),
                eg.n_ladder(6,True),
                eg.bridge_graph(),
                generate_lattice(points),
                eg.cut_boundaries(generate_lattice(points), [False,True]),
                eg.cut_boundaries(generate_lattice(points), [True,True]),
                eg.honeycomb_lattice(12)
    ]

    for lat in weird_graphs:
        make_dual(lat)

def test_vertices_to_polygon():
    points = uniform(30)
    lattice = generate_lattice(points)

    sets = [
        np.random.randint(lattice.n_vertices),
        np.arange(np.random.randint(lattice.n_vertices)//2),
        np.arange(lattice.n_vertices)
    ]

    for s in sets:
        vertices_to_polygon(lattice, s)