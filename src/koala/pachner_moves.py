import numpy as np
from koala.lattice import Lattice, _sorted_vertex_adjacent_edges
from koala import graph_utils as gu
from koala import plotting as pl
from matplotlib import pyplot as plt


def bipartite_1_plus(
    lattice: Lattice, plaquette: int, central_vertex: int, debug_plot=False
):
    """Performs a 1+ bipartite bistellar flip on a given trivalent bipartite lattice, adding (at least) a single square or plaquette

    Args:
        lattice (Lattice): the lattice to perform the flip on
        plaquette (int):  the plaquette that the flip will be performed on, must have more than 4 sides
        central_vertex (int): the vertex that will be the center of the flip
        debug_plot (bool, optional): whether to plot the flip for debugging purposes. Defaults to False.

    Returns:
        Lattice: the resulting lattice after the flip
    """

    # make sure the lattice is bipartite
    sides = np.array([p.n_sides for p in lattice.plaquettes])
    assert np.all(sides % 2 == 0)

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

    p1 = 0.5 * np.sum(vectors[1:-1], axis=0)
    p2 = 0.5 * np.sum(vectors, axis=0)

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


def bipartite_1_minus(lattice: Lattice, edge: int):
    """Performs a 1- bipartite bistellar flip on a given trivalent bipartite lattice, removing (at least) a single square or plaquette

    Args:
        lattice (Lattice): the lattice to perform the flip on
        edge (int): the edge to remove

    Returns:
        Lattice: new lattice with the flip performed
    """

    adjacent_plaquettes = lattice.edges.adjacent_plaquettes[edge]
    edge_vertices = lattice.edges.indices[edge]
    neighbour_vertices = np.concatenate(
        [lattice.vertices.adjacent_vertices[v] for v in edge_vertices]
    )
    starts_ends = np.array(list(set(neighbour_vertices) - set(edge_vertices)))

    edges_out = lattice.edges.indices.copy()
    crossing_out = lattice.edges.crossing.copy()
    for i in adjacent_plaquettes:
        plaq = lattice.plaquettes[i]
        assert plaq.n_sides > 4, "You cant contract a square"

        bookends = np.where(np.isin(plaq.vertices, starts_ends))[0]

        edges_to_sum = plaq.edges[bookends[0] : bookends[1]]
        directions_to_sum = plaq.directions[bookends[0] : bookends[1]]

        edge = np.array(
            [
                lattice.edges.indices[
                    edges_to_sum[0], 1 * (directions_to_sum[0] == -1)
                ],
                lattice.edges.indices[
                    edges_to_sum[-1], 1 * (directions_to_sum[-1] == 1)
                ],
            ]
        )

        crossing = np.sum(
            lattice.edges.crossing[edges_to_sum] * directions_to_sum[:, None], axis=0
        )

        edges_out = np.concatenate((edges_out, edge[None, :]), axis=0)
        crossing_out = np.concatenate((crossing_out, crossing[None, :]), axis=0)

    data = gu._remove_vertices_backend(
        lattice.vertices.positions, edges_out, crossing_out, edge_vertices
    )
    return Lattice(*data)


# def bipartite_2_plus(lattice: Lattice, edge: int):
#     """Performs a 2+ bipartite bistellar flip on a given trivalent bipartite lattice, adding two squares

#     Args:
#         lattice (Lattice): the lattice to perform the flip on
#         edge (int): the edge to flip

#     Returns:
#         Lattice: new lattice with the flip performed
#     """

    

