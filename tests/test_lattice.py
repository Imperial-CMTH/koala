import numpy as np
import pytest
from koala import pointsets
from koala import voronization
from koala.graph_utils import cut_boundaries
from koala.lattice import Lattice, LatticeException
from koala.example_graphs import *
from koala.example_graphs import single_plaquette

def test_lattice_class():
    # generate a ton of weird graphs
    points2 = pointsets.bluenoise(30,3,3)
    weird_graphs = [
                tri_square_pent(),
                two_triangles(),
                tutte_graph(),
                n_ladder(6,True),
                bridge_graph(),
                voronization.generate_lattice(points2),
                cut_boundaries(voronization.generate_lattice(points2), [False,True]),
                cut_boundaries(voronization.generate_lattice(points2), [True,True]),
                honeycomb_lattice(12)
    ]

    # run the plaquette code by accessing the plaquette property
    for graph in weird_graphs:
        graph.plaquettes
        graph.edges.adjacent_plaquettes
        graph.vertices.adjacent_plaquettes
        
        if len(graph.plaquettes) > 0:
            p = graph.plaquettes[0]
            assert(p.edges.shape[0] == p.n_sides)

    # test the cut boundaries functions
    for graph in weird_graphs:
        cut_boundaries(graph, [False,True])
        cut_boundaries(graph, [True,True])
    
    def check_neighbouring_plaquettes(lattice: Lattice):
        for n, plaquette in enumerate(lattice.plaquettes):
            edges = plaquette.edges
            edge_plaquettes = lattice.edges.adjacent_plaquettes[edges]
            roll_vals = np.where(edge_plaquettes != n)[1]
            other_plaquettes =  edge_plaquettes[np.arange(len(roll_vals)), roll_vals]

            print(other_plaquettes, lattice.plaquettes[n].adjacent_plaquettes)

            for a,b in zip(other_plaquettes, lattice.plaquettes[n].adjacent_plaquettes):
                    assert( a == b)

                    
    for graph in weird_graphs:
        check_neighbouring_plaquettes(graph)



def test_multigraphs():
    "check that the plaquette code fails gracefully on multigraphs"
    with pytest.raises(LatticeException):
        m = multi_graph()
        m.plaquettes

def test_cache_order():
    "check that it still works if graph.edges.neighbouring_plaquettes is accessed before graph.plaquettes"
    g = tri_square_pent()
    g.edges.adjacent_plaquettes
    
    g = tri_square_pent()
    g.vertices.adjacent_plaquettes


def test_higher_coordination_number():
    for x in range(3,10):
        l2 = higher_coordination_number_example(x)
        assert len(l2.plaquettes) == x