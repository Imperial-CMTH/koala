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