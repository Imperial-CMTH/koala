import numpy as np
import pytest
from matplotlib import pyplot as plt

from koala import pointsets
from koala import voronization
from koala.lattice import cut_boundaries, LatticeException
from koala.example_graphs import *


def test_lattice_class():
    # generate a ton of weird graphs
    points2 = pointsets.generate_bluenoise(30,3,3)
    weird_graphs = [
                tri_square_pent(),
                two_tri(),
                tutte_graph(),
                n_ladder(6,True),
                bridge_graph(),
                voronization.generate_lattice(points2),
                cut_boundaries(voronization.generate_lattice(points2), [False,True]),
                cut_boundaries(voronization.generate_lattice(points2), [True,True])
    ]

    # run the plaquette code by accessing the plaquette property
    for graph in weird_graphs:
        graph.plaquettes

    # test the cut boundaries functions
    for graph in weird_graphs:
        cut_boundaries(graph, [False,True])
        cut_boundaries(graph, [True,True])

def test_multigraphs():
    "check that the plaquette code fails gracefully on multigraphs"
    with pytest.raises(LatticeException):
        m = multi_graph()
        m.plaquettes
