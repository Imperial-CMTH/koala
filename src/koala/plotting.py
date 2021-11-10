import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from .graph_utils import vertex_neighbours, clockwise_edges_about

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

def plot_degeneracy_breaking(vertex_i, g, ax = None):
    """
    Companion function to graph_utils.clockwise_edges_about, 
    plots the edges on an axis with labelled angles and the positive x axis as a dotted line
    """
    if ax is None: ax = plt.gca()
    #we choose the 0th vertex
    vertex = g.vertices[vertex_i]
    
    vertex_colors = np.array(['k' for _ in g.vertices])
    
    #label the main vertex red
    vertex_colors[vertex_i] = 'r'
    
    #label its neighbours green
    vertex_colors[vertex_neighbours(vertex_i, g.adjacency)[0]] = 'g'

    #color the edges in a clockwise fashion
    ordered_edge_indices = clockwise_edges_about(vertex_i, g)

    highlight_edge_neighbours = np.array(['k' for i in range(g.adjacency.shape[0])])
    highlight_edge_neighbours[ordered_edge_indices] = ['r', 'g', 'b']
 
    ax.hlines(y = vertex[1], xmin = vertex[0], xmax = ax.get_ylim()[1], linestyle = 'dotted', alpha = 0.5, color = 'k')

    plot_lattice(g.vertices, g.adjacency, g.adjacency_crossing, edge_colors = highlight_edge_neighbours, scatter_args = dict(color = vertex_colors), ax = ax)
    
