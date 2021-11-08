import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt

def plot_lattice(vertices, adjacency, adjacency_crossing, ax = None, edge_colors = None, vertex_colors = None, scatter_args = None):
    """"

    Args:

        adjacency_crossing: np.ndarray (n_edges, 2), 
    """
    if ax is None: ax = plt.gca()

    edge_vertices = vertices[adjacency]
    displacements = edge_vertices[:,0,:] - edge_vertices[:,1,:]

    mask = np.any(adjacency_crossing != 0, axis = -1)
    outside_idx = np.where(mask)[0]
    inside_idx = np.where(np.logical_not(mask))[0]

    inside_edges = adjacency[inside_idx]
    outside_edges = adjacency[outside_idx]

    if edge_colors is not None:
        inside_colors = edge_colors[inside_idx]
        outside_colors = edge_colors[outside_idx]
    else:
        inside_colors = 'k'
        outside_colors = 'r'

    inside_edge_vertices = vertices[inside_edges]
    outside_edge_vertices = vertices[outside_edges]

    lc_inside = LineCollection(inside_edge_vertices, colors = inside_colors)
    ax.add_collection(lc_inside)

    for i, sign in enumerate([-1, 1]):
        temp =  outside_edge_vertices.copy()
        temp[:, i, :] = temp[:, i, :] + sign * adjacency_crossing[outside_idx]
        lc_outside = LineCollection(temp, colors = outside_colors)
        ax.add_collection(lc_outside)

    ax.set(xlim = (0,1), ylim = (0,1))
    if scatter_args is not None: ax.scatter(
        vertices[:,0],
        vertices[:,1],
        zorder = 3,
        **scatter_args,
    )

    return ax