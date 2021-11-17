import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
from .graph_utils import vertex_neighbours, clockwise_edges_about
from .voronization import generate_point_array

def _line_fully_in_unit_cell(lines : np.ndarray) -> np.ndarray:
    """Tells you which of a set of lines is entirely within the unit cell

    Args:
        lines (np.ndarray): lines[i, j, :] is the jth point of the ith line

    Returns:
        np.ndarray: out[i] == 1 if line[i] is fully inside the unit cell
    """
    coords_in_cell = (0 < lines) & (lines < 1)
    return np.all(coords_in_cell, axis = (-2,-1))

def _lines_cross_unit_cell(lines : np.ndarray) -> np.ndarray:
    """Tells you which of a set of lines crosses the boundaries of the unit cell
    
    Works by writing a point on each line as: point = start*t + (1-t)*end
    It then solves for t = (l - end) / (start - end) for l = (0,0), (1,1) which gives:
    t = ((tx0, ty0), (tx1, ty1))
    if 0 < t_x0 < 1 it means point_x = 0 at t = t_x0
    so then we check that 0 < point_y(t_x0) < 1 and if so then it works 

    Args:
        lines (np.ndarray): lines[i, j, :] is the jth point of the ith line

    Returns:
        np.ndarray: out[i] == 1 if line[i] is fully inside the unit cell
    """
    start, end = lines[:, 0, None, :], lines[:, 1, None, :] #start.shape = (n_lines, 1, 2)
    l = np.array([[0,0],[1,1]])[None, :, :] #shape (1, 2, 2)
    t = (l - end) / (start - end) #shape (n_lines, 2, 2)
    
    #flip the last axis of start and end so that we use the t that made the x coord cross to compute the y coord
    #t.shape (n_lines, 0/1, x/y) start[..., ::-1].shape (n_lines, 1, y,x)
    other_coord_value_at_t = start[..., ::-1] * t + (1 - t) * end[..., ::-1]
    cross = (0 < t) & (t <= 1) & (0 < other_coord_value_at_t) & (other_coord_value_at_t <= 1)
    cross = np.any(cross, axis = (1,2))
    return cross

def plot_lattice(lattice, ax = None, 
                 edge_labels = None, edge_color_scheme = ['r','g','b'],
                 vertex_labels = None, vertex_color_scheme = ['r','b'],
                 edge_arrows = False,
                 scatter_args = None):
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

    edge_color_scheme = np.array(edge_color_scheme)
    
    if edge_labels is not None:
        edge_colors = np.tile(edge_color_scheme[edge_labels], 9)
    else:
        edge_colors = np.full(fill_value = 'k', shape = adjacency.shape[0]*9)
    
    edge_vertices = vertices[adjacency] 
    edge_vertices[:, 0, :] -= adjacency_crossing
    
    unit_cell_vectors = generate_point_array(np.array([0,0]), padding = 1)[:, None, None, :] #shape (9, 2) -> (9, 1, 1, 2)
    replicated_edges = edge_vertices[None,...] + unit_cell_vectors #shape (n_edges, 2, 2) -> (9, n_edges, 2, 2)
    replicated_edges =  replicated_edges.reshape((-1,2,2)) #shape (9, n_edges, 2, 2) -> (9*n_edges, 2, 2)
    
    vis = _lines_cross_unit_cell(replicated_edges) | _line_fully_in_unit_cell(replicated_edges)
    lc = LineCollection(replicated_edges[vis, ...], colors = edge_colors[vis])
    ax.add_collection(lc)
    
    if edge_arrows:
        for color, (start, end) in zip(edge_colors[vis], replicated_edges[vis, ...]):
            center = 1/2 * (end + start)
            direction = 0.001 * (end - start) / np.linalg.norm((end - start))
            arrow_start = center - direction
            ax.arrow(x=arrow_start[0], y=arrow_start[1], dx=direction[0], dy=direction[1],
                     color = color,
                     head_width = 0.03, head_length = 0.03, width = 0, zorder = 4)



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