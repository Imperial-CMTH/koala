import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from .graph_utils import vertex_neighbours, clockwise_edges_about

def plot_lattice(lattice, ax = None, edge_labels = None, edge_color_scheme = ['r','g','b'], vertex_labels = None, vertex_color_scheme = ['r','b'], scatter_args = None):
    """Plots a 2d graph. Optionally with coloured edges or vertices.

    Args:
        lattice (Lattice): A koala lattice dataclass containing the vertices, adjacency and adjacency_crossing
        ax (matplotlib axis, optional): Axis to plot to. Defaults to plt.gca().
        edge_labels (np.ndarray, optional): A list of integer edge labels, length must be the same as adjacency, If None, then all edges are plotted in black. Defaults to None.
        edge_color_scheme (list, optional): List of matplotlib  colour strings for edge colouring. Defaults to ['r','g','b'].
        vertex_labels (np.ndarray, optional): A list of labels for colouring the vertices, if None, vertices are not plotted. Defaults to None.
        vertex_color_scheme (list, optional): List of matplotlib  colour strings for vertex colouring. Defaults to ['r','b'].
        scatter_args ([type], optional): Directly passes arguments to plt.scatter for the vertices. Use if you want to put in custom vertex attributes. Defaults to None.

    Returns:
        matplotlib axis: The axis that we have plotted to.
    """

    vertices, adjacency, adjacency_crossing = lattice.vertices, lattice.adjacency, lattice.adjacency_crossing

    if ax is None: ax = plt.gca()

    edge_vertices = vertices[adjacency]
    displacements = edge_vertices[:,0,:] - edge_vertices[:,1,:]

    mask = np.any(adjacency_crossing != 0, axis = -1)
    outside_idx = np.where(mask)[0]
    inside_idx = np.where(np.logical_not(mask))[0]

    inside_edges = adjacency[inside_idx]
    outside_edges = adjacency[outside_idx]

    edge_color_scheme = np.array(edge_color_scheme)
    if edge_labels is not None:
        inside_colors = edge_color_scheme[edge_labels[inside_idx]]
        outside_colors = edge_color_scheme[edge_labels[outside_idx]]
    else:
        inside_colors = 'k'
        outside_colors = 'k'

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
    vertex_color_scheme = np.array(vertex_color_scheme)

    if vertex_labels is not None:
        vertex_colors = vertex_color_scheme[vertex_labels]
        scatter_args = dict(c = vertex_colors)

    if (scatter_args is not None) or (vertex_labels is not None): ax.scatter(
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
    
def plot_vertex_indices(g, ax = None, offset = 0.01):
    """
    Plot the indices of the vertices on a graph
    """
    if ax is None: ax = plt.gca()
    for i, v in enumerate(g.vertices): ax.text(*(v+offset), f"{i}")
    
#TODO: Make this work with edges that cross the boundaries
def plot_edge_indices(g, ax = None, offset = 0.01):
    """
    Plot the indices of the edges on a graph
    """
    if ax is None: ax = plt.gca()
    for i, e in enumerate(g.adjacency): 
        midpoint = g.vertices[e].mean(axis = 0)
        if not np.any(g.adjacency_crossing[i]) != 0:
            ax.text(*(midpoint+offset), f"{i}", color = 'g')