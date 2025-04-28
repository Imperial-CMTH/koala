import numpy as np
import pickle as pkl

from koala.plotting import plot_vertex_indices, plot_degeneracy_breaking, plot_lattice
from koala import example_graphs as eg
from koala.pointsets import uniform
from koala.graph_utils import (
    lloyd_relaxation,
    vertex_neighbours,
    clockwise_edges_about,
    adjacent_plaquettes,
    remove_trailing_edges,
    make_dual,
    vertices_to_polygon,
    dimerise,
    remove_vertices,
    tile_unit_cell,
    untile_unit_cell,
    shift_vertex,
    dimer_collapse,
    com_relaxation,
    edge_spanning_tree,
    plquette_spanning_tree
)
from koala.lattice import Lattice
from koala.voronization import generate_lattice

# two test lattices
g = Lattice(
    vertices=np.array([[0.5, 0.5], [0.1, 0.1], [0.5, 0.9], [0.9, 0.1]]),
    edge_indices=np.array([[0, 1], [0, 2], [0, 3]]),
    edge_crossing=np.array([[0, 0], [0, 0], [0, 0]]),
)
points = uniform(30)
lattice = generate_lattice(points)

weird_graphs = [
    eg.tri_square_pent(),
    eg.two_triangles(),
    eg.tutte_graph(),
    eg.n_ladder(6, True),
    eg.bridge_graph(),
    generate_lattice(points),
    eg.cut_boundaries(generate_lattice(points), [False, True]),
    eg.cut_boundaries(generate_lattice(points), [True, True]),
    eg.honeycomb_lattice(12),
]
weird_graphs_simplified = [
    eg.tri_square_pent(),
    eg.two_triangles(),
    eg.tutte_graph(),
    eg.n_ladder(6, True),
    generate_lattice(points),
    eg.cut_boundaries(generate_lattice(points), [False, True]),
    eg.cut_boundaries(generate_lattice(points), [True, True]),
    eg.honeycomb_lattice(12),
]


def test_remove_vertices():
    remove_vertices(g, [0, 1])
    remove_vertices(g, [1], True)
    remove_vertices(lattice, [1, 2, 3])
    remove_vertices(lattice, [1, 2, 3], True)


def test_vertex_neighbours():
    plot_lattice(g, edge_arrows=True, edge_index_labels=True, vertex_labels=0)
    plot_vertex_indices(g)
    plot_degeneracy_breaking(0, g)

    vertex_indices, edge_indices = vertex_neighbours(g, 0)
    assert np.all(vertex_indices == (np.array([1, 2, 3]),))
    assert np.all(edge_indices == np.array([0, 1, 2]))


def test_clockwise_edges_about():
    assert np.all(clockwise_edges_about(vertex_index=0, g=g) == np.array([1, 0, 2]))


def test_adjacent_plaquettes_function():
    lat = eg.tri_square_pent()
    p, e = adjacent_plaquettes(lat, 1)
    assert np.all(p == [0, 2])
    assert np.all(e == [2, 5])


def test_trailing_edges():
    with open("tests/data/trailing_lattice_example.pickle", "rb") as f:
        lattice = pkl.load(f)
    l_out = remove_trailing_edges(lattice)
    assert l_out.n_edges == 13
    assert l_out.n_vertices == 11
    assert l_out.n_plaquettes == 3


def test_dual():
    for lat in weird_graphs:
        make_dual(lat)


def test_vertices_to_polygon():
    sets = [
        np.random.randint(lattice.n_vertices),
        np.arange(np.random.randint(lattice.n_vertices) // 2),
        np.arange(lattice.n_vertices),
    ]
    for s in sets:
        vertices_to_polygon(lattice, s)


def test_dimerisation():
    dimerise(lattice)


def test_lloyds():
    lloyd_relaxation(lattice)

def test_com():
    com_relaxation(lattice)


def test_tile_unit_cell():
    lat_params = (
        lattice.vertices.positions,
        lattice.edges.indices,
        lattice.edges.crossing,
    )

    l1 = tile_unit_cell(*lat_params, [1, 1])
    l2 = tile_unit_cell(*lat_params, [2, 1])
    l3 = tile_unit_cell(*lat_params, [2, 2])

    l1_original = untile_unit_cell(
        l1.vertices.positions, l1.edges.indices, l1.edges.crossing, [1, 1]
    )

    l2_original = untile_unit_cell(
        l2.vertices.positions, l2.edges.indices, l2.edges.crossing, [2, 1]
    )

    l3_original = untile_unit_cell(
        l3.vertices.positions, l3.edges.indices, l3.edges.crossing, [2, 2]
    )

    assert np.allclose(l1_original[0], lattice.vertices.positions)
    assert np.allclose(l2_original[0], lattice.vertices.positions)
    assert np.allclose(l3_original[0], lattice.vertices.positions)


def test_shift_vertex():
    lat_data = (
        lattice.vertices.positions,
        lattice.edges.indices,
        lattice.edges.crossing,
        lattice.vertices.adjacent_edges
    )

    shift_vertex(lat_data, 0, [0.1, 0.1])
    shift_vertex(lat_data, 1, [0.1, 0.1])
    shift_vertex(lat_data, 1, [0.9, 0.9])

def test_collapse_dimer():
    for lat in weird_graphs:
        
        boundaries_are_crossed = np.any(lat.edges.crossing != 0, axis=0)
        if not np.all(boundaries_are_crossed):
            continue
        
        dimer = dimerise(lat)
        dimer_collapse(lat, dimer)

def test_spanning_trees():
    def _sp_tree_test(lattice:Lattice):

        for s_edge in [True, False]:
            for c_bound in [True, False]:
                sp = edge_spanning_tree(lattice,s_edge, c_bound)
                vertices_in_sp = np.unique(lattice.edges.indices[sp])
                assert len(vertices_in_sp) == lattice.n_vertices

    def _plaq_tree_test(lattice:Lattice):

        for s_edge in [True, False]:
            for c_bound in [True, False]:
                sp = plquette_spanning_tree(lattice,s_edge, c_bound)
                plaqs_in_sp = np.unique(lattice.edges.adjacent_plaquettes[sp])
                assert len(plaqs_in_sp) == lattice.n_plaquettes

    for graph in weird_graphs_simplified:
        _sp_tree_test(graph)
        _plaq_tree_test(graph)


