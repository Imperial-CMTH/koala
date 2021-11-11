import numpy as np
from koala.graph_utils import edge_neighbours, clockwise_edges_about
from koala.SAT import edge_color
from koala.voronization import generate_pbc_voronoi_adjacency


def test_small_multigraph():
    

    #a horrible graph where two nodes are joined by two edges 
    # that wind different ways around the torus
    points = np.array([[0.40494994, 0.09533812], [0.31187785, 0.43756783], [0.16634927, 0.49970101], [0.39096236, 0.92426087]])

    g = generate_pbc_voronoi_adjacency(points)
    assert(np.all(np.bincount(g.adjacency.flatten()) == 3))

    fixed = list(enumerate(clockwise_edges_about(vertex_i = 0, g=g)))
    solveable, solutions = edge_color(g.adjacency, n_colors = 3, fixed = fixed, all_solutions = True)
    assert(solveable)
    assert(solutions.shape[0] == 2)

    solveable, solutions = edge_color(g.adjacency, n_colors = 3, fixed = fixed, n_solutions = 1)
    assert(solveable)
    assert(solutions.shape[0] == 1)

    colors = np.array(['r', 'g', 'b'])[solutions[0]]

    for i in range(colors.shape[0]):
        neighbouring_edges = edge_neighbours(i, g.adjacency)
        assert(colors[i] not in colors[neighbouring_edges])