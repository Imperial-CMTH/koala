#TODO 
#   - test for all_solutions argument 
#   - test for unsolveable models and returning the minimum failing graph.


import numpy as np
import pytest
from koala.pointsets import generate_random
from koala.voronization import generate_lattice
from koala.graph_color import vertex_color, edge_color, color_lattice
from koala.graph_utils import edge_neighbours, clockwise_edges_about

pytestmark = pytest.mark.filterwarnings("ignore:numpy")

def test_vertex_coloring():
    for n in range(4, 50, 5):
        points = generate_random(n)
        g = generate_lattice(points)    
        solveable, solution = vertex_color(g.edges.indices, n_colors = 3)
        if solveable:
            colors = np.array(['orange', 'b', 'k'])[solution]

            cedges = colors[g.edges.indices]
            assert(not np.any(cedges[:, 0] == cedges[:, 1]))


def test_edge_coloring():
    for n in [4,10,30,50]:
        points = generate_random(n)
        lattice = generate_lattice(points)
        solveable, solution = edge_color(lattice, n_colors = 3)
        
        if solveable:
            colors = np.array(['r', 'g', 'b'])[solution]
        
            for i in range(colors.shape[0]):
                neighbouring_edges = edge_neighbours(lattice, i)
                assert(colors[i] not in colors[neighbouring_edges])

            

def test_unsolveable():
    pass

def test_color_lattice():
    from koala.example_graphs import two_triangles, tri_square_pent, tutte_graph
    graphs = [two_triangles(), tri_square_pent(), tutte_graph()]
    for g in graphs:
        fixed = enumerate(clockwise_edges_about(vertex_index = 0, g=g))
        solveable, solution = edge_color(g, n_colors = 3, fixed = fixed)
        if solveable:
            solution2 = color_lattice(g)
            assert(np.all(solution == solution2))