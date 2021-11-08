#TODO 
#   - test for all_solutions argument 
#   - test for unsolveable models and returning the minimum failing graph.

def test_vertex_coloring():
    n = 40
    from koala.pointsets import generate_random
    points = generate_random(n)

    from koala.voronization import generate_pbc_voronoi_adjacency
    vertices, adjacency = generate_pbc_voronoi_adjacency(points)
        
    # from koala.plotting import plot_lattice
    # plot_lattice(vertices, adjacency)

    from koala.SAT import vertex_color
    solution = vertex_color(adjacency, n_colors = 3)
    
    # colors = np.array(['orange', 'b', 'k'])[solution]
    # plot_lattice(vertices, adjacency, scatter_args = dict(color = colors))

    cedges = colors[adjacency]
    assert(not np.any(cedges[:, 0] == cedges[:, 1]))


def test_edge_coloring():
    n = 40
    from koala.pointsets import generate_random
    points = generate_random(n)

    from koala.voronization import generate_pbc_voronoi_adjacency
    vertices, adjacency = generate_pbc_voronoi_adjacency(points)

    from koala.SAT import edge_color
    solution = edge_color(adjacency, n_colors = 3)

    colors = np.array(['orange', 'b', 'k'])[solution]
    # plot_lattice(vertices, adjacency, edge_colors = colors)

    from koala.graph_utils import edge_neighbours


    ns = np.array([edge_neighbours(i, adjacency) for i in range(adjacency.shape[0])])
    cs = colors[ns]
    assert(not np.any([colors[i] in cs[i] for i in range(colors.shape[0])]))


"""
unsolveable = PBC_Voronoi(vertices=array([[0.21475886, 0.25115016],
       [0.15074368, 0.28779049],
       [0.7637336 , 0.04206213],
       [0.77362058, 0.08284606],
       [0.60367343, 0.85501055],
       [0.20977552, 0.54240834],
       [0.44615917, 0.7539095 ],
       [0.57293506, 0.78291032],
       [0.67351609, 0.42529479],
       [0.64434817, 0.64459786]]), adjacency=array([[0, 1],
       [8, 0],
       [8, 3],
       [2, 3],
       [5, 6],
       [7, 6],
       [7, 9],
       [8, 9],
       [1, 5],
       [4, 7],
       [4, 0],
       [4, 2],
       [9, 5],
       [0, 4],
       [1, 3],
       [3, 1],
       [5, 9],
       [6, 2]]), adjacency_crossing=array([[ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  0],
       [ 0,  1],
       [ 0,  1],
       [ 1,  0],
       [ 0, -1],
       [-1,  0],
       [ 1,  0],
       [-1,  0],
       [-1,  1]]))
"""