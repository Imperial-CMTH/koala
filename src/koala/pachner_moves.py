import numpy as np
from koala.lattice import Lattice, _sorted_vertex_adjacent_edges
from koala import graph_utils as gu
from koala import plotting as pl
from matplotlib import pyplot as plt

def bipartite_1_plus(lattice: Lattice, plaquette: int, central_vertex: int, debug_plot=False):
    """Performs a 1+ bipartite bistellar flip on a given trivalent bipartite lattice

    Args:
        lattice (Lattice): _description_
        plaquette (int): _description_
        central_vertex (int): _description_
        debug_plot (bool, optional): _description_. Defaults to False.

    Returns:
        Lattice: _description_
    """

    # make sure the lattice is bipartite
    sides = np.array([p.n_sides for p in lattice.plaquettes])
    assert np.all(sides%2 == 0)

    plaq = lattice.plaquettes[plaquette]
    assert plaq.n_sides > 4, "You cant contract a square"
    assert central_vertex in plaq.vertices, "The vertex must be on the plaquette given"

    central_pos = lattice.vertices.positions[central_vertex]

    _ = np.where(plaq.vertices == central_vertex)[0][0]
    vertices_around = plaq.vertices[np.arange(_ - 2, _ + 3) % plaq.n_sides]
    edges_around = plaq.edges[np.arange(_ - 2, _ + 2) % plaq.n_sides]
    directions_around = plaq.directions[np.arange(_ - 2, _ + 2) % plaq.n_sides]
    directions_around[:2] *= -1  # make sure they emanate from the central vertex
    vectors = lattice.edges.vectors[edges_around] * directions_around[:, None]
    crossing_around = lattice.edges.crossing[edges_around] * directions_around[:, None]

    p1 = 0.6 * np.sum(vectors[1:-1], axis=0)
    p2 = 0.4 * np.sum(vectors, axis=0)

    i1 = lattice.n_vertices
    i2 = lattice.n_vertices + 1

    edges_to_add = [
        [i1, vertices_around[1]],
        [i1, vertices_around[3]],
        [i2, vertices_around[0]],
        [i2, vertices_around[4]],
        [i1, i2],
    ]

    crossing_to_add = [
        crossing_around[1],
        crossing_around[2],
        crossing_around[1] + crossing_around[0],
        crossing_around[2] + crossing_around[3],
        [0, 0],
    ]

    edges_to_remove = edges_around[np.array([0, 3])]

    new_vertices = np.concatenate(
        [lattice.vertices.positions, central_pos[None, :], central_pos[None, :]]
    )
    new_edges = np.concatenate([lattice.edges.indices, edges_to_add])
    new_crossing = np.concatenate([lattice.edges.crossing, np.array(crossing_to_add)])
    new_edges = np.delete(new_edges, edges_to_remove, axis=0)
    new_crossing = np.delete(new_crossing, edges_to_remove, axis=0)
    l_data = new_vertices, new_edges, new_crossing

    # make adjacency
    edge_vectors = (
        new_vertices[new_edges][:, 1] - new_vertices[new_edges][:, 0] + new_crossing
    )
    adjacent_edges = _sorted_vertex_adjacent_edges(
        new_vertices, new_edges, edge_vectors
    )

    l_data = gu.shift_vertex((*l_data, adjacent_edges), i1, p1)
    l_data = gu.shift_vertex((*l_data, adjacent_edges), i2, p2)

    l_out = Lattice(*l_data)

    if debug_plot:
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        pl.plot_edges(lattice, ax=ax[0], linewidth=1)
        pl.plot_edges(
            lattice,
            ax=ax[0],
            linewidth=3,
            subset=edges_around,
            directions=directions_around,
        )
 
        pl.plot_edges(l_out, ax=ax[1], linewidth=1)
        # pl.plot_vertex_indices(lattice, ax=ax[2])
        pl.plot_plaquette_indices(lattice, ax=ax[2])
        pl.plot_edges(lattice, ax=ax[2])

    return l_out


def bipartite_1_minus(lattice: Lattice, plaquette: int, central_vertex: int, debug_plot=False):

    #  make sure the lattice is bipartite
    sides = np.array([p.n_sides for p in lattice.plaquettes])
    assert np.all(sides%2 == 0)

    plaq = lattice.plaquettes[plaquette]
    assert plaq.n_sides == 4, "You must choose a square"
    assert central_vertex in plaq.vertices, "The vertex must be on the plaquette given"

    # find vertices about square
    _ = np.where(plaq.vertices == central_vertex)[0][0]
    vertices_around = plaq.vertices[np.arange(_ - 2, _ + 3) % plaq.n_sides]
    edges_around = plaq.edges[np.arange(_ - 2, _ + 2) % plaq.n_sides]
    directions_around = plaq.directions[np.arange(_ - 2, _ + 2) % plaq.n_sides]
    directions_around[:2] *= -1  # make sure they emanate from the central vertex
    vectors = lattice.edges.vectors[edges_around] * directions_around[:, None]
    crossing_around = lattice.edges.crossing[edges_around] * directions_around[:, None]
    
    # find bridge connected one
    # bridge_


    print(directions_around)