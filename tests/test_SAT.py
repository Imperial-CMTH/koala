#TODO 
#   - test for all_solutions argument 
#   - test for unsolveable models and returning the minimum failing graph.


import numpy as np
import pytest
from koala.pointsets import generate_random
from koala.voronization import generate_pbc_voronoi_adjacency
from koala.graph_color import vertex_color, edge_color
from koala.graph_utils import edge_neighbours

pytestmark = pytest.mark.filterwarnings("ignore:numpy")

def test_vertex_coloring():
    for n in range(4, 50, 5):
        points = generate_random(n)
        g = generate_pbc_voronoi_adjacency(points)    
        solveable, solution = vertex_color(g.edges.indices, n_colors = 3)
        if solveable:
            colors = np.array(['orange', 'b', 'k'])[solution]

            cedges = colors[g.edges.indices]
            assert(not np.any(cedges[:, 0] == cedges[:, 1]))


def test_edge_coloring():
    for n in range(4, 50):
        points = generate_random(n)
        g = generate_pbc_voronoi_adjacency(points)
        solveable, solution = edge_color(g.edges.indices, n_colors = 3)
        
        if solveable:
            colors = np.array(['r', 'g', 'b'])[solution]
        
            for i in range(colors.shape[0]):
                neighbouring_edges = edge_neighbours(i, g.edges.indices)
                assert(colors[i] not in colors[neighbouring_edges])

def test_unsolveable():
    pass