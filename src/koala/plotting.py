import numpy as np
import scipy
import matplotlib
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt

from .graph_utils import vertex_neighbours, clockwise_edges_about
from .voronization import generate_point_array, Lattice

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
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (l - end) / (start - end) #shape (n_lines, 2, 2)
    
    #values of the above where start - end has zeros mean the lines are axis aligned
    #so just check if l == end
    t[~np.isfinite(t)] = 0.5*(l == end)[~np.isfinite(t)] 

    #flip the last axis of start and end so that we use the t that made the x coord cross to compute the y coord
    #t.shape (n_lines, 0/1, x/y) start[..., ::-1].shape (n_lines, 1, y,x)
    other_coord_value_at_t = start[..., ::-1] * t + (1 - t) * end[..., ::-1]
    cross = (0 < t) & (t <= 1) & (0 < other_coord_value_at_t) & (other_coord_value_at_t <= 1)
    cross = np.any(cross, axis = (1,2))
    return cross

def plot_lattice(lattice, ax = None, 
                 edge_labels = None, edge_color_scheme = ['r','g','b','k'],
                 vertex_labels = None, vertex_color_scheme = ['r','b','k'],
                 edge_arrows = False, edge_index_labels = False,
                 scatter_args = None):
    """Plots a 2d graph. Optionally with coloured edges or vertices.

    Args:
        lattice (Lattice): A koala lattice dataclass containing the vertices, adjacency and adjacency_crossing
        ax (matplotlib axis, optional): Axis to plot to. Defaults to plt.gca().
        edge_labels (np.ndarray, optional): A list of integer edge labels, length must be the same as adjacency, If None, then all edges are plotted in black. Defaults to None.
        edge_color_scheme (list, optional): List of matplotlib  colour strings for edge colouring. Defaults to ['r','g','b'].
        vertex_labels (np.ndarray, optional): A list of labels for colouring the vertices, if None, vertices are not plotted. Defaults to None.
        vertex_color_scheme (list, optional): List of matplotlib  colour strings for vertex colouring. Defaults to ['r','b'].
        edge_arrows (bool | np.ndarray): If True or boolean array, arrows are drawn on the edges. Defaults to False.
        edge_index_labels (bool | np.ndarray): If True or boolean array, edge indices are plotted on the edges. Defaults to False.
        scatter_args (dict, optional): Directly passes arguments to plt.scatter for the vertices. Use if you want to put in custom vertex attributes. Defaults to None.

    Returns:
        matplotlib axis: The axis that we have plotted to.
    """

    vertices, adjacency, adjacency_crossing = lattice.vertices, lattice.edges.indices, lattice.edges.crossing

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
    
    if np.any(edge_arrows) or np.any(edge_index_labels):
        original_edge_indices = np.tile(np.arange(lattice.edges.indices.shape[0]), 9)
        if isinstance(edge_arrows, bool): edge_arrows = np.full(fill_value = edge_arrows, shape = lattice.edges.indices.shape[0])
        if isinstance(edge_index_labels, bool): edge_index_labels = np.full(fill_value = edge_index_labels, shape = lattice.edges.indices.shape[0])
        edge_arrows = np.tile(edge_arrows, 9)
        edge_index_labels = np.tile(edge_index_labels, 9)

        for i, color, (start, end) in zip(original_edge_indices[vis], edge_colors[vis], replicated_edges[vis, ...]):
            center = 1/2 * (end + start)
            length = np.linalg.norm(end - start)
            head_length = min(0.2 * length, 0.02)
            direction = head_length * (end - start) / length
            arrow_start = center - direction
            if edge_arrows[i]: 
                ax.arrow(x=arrow_start[0], y=arrow_start[1], dx=direction[0], dy=direction[1],
                     color = color,
                     head_width = head_length, head_length = head_length, width = 0, zorder = 4, head_starts_at_zero = True, length_includes_head = True)
            if edge_index_labels[i]:
                ax.text(*(center), f"{i}", color = 'w' if edge_arrows[i] else 'k', ha = 'center', va = 'center', zorder = 5)



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

cdict = {'red':   [(0.0,  0.0, 1.0),
                   (1.0,  0.0, 1.0)],

         'green': [(0.0,  0.0, 1.0),
                   (1.0,  1.0, 1.0)],

         'blue':  [(0.0,  0.0, 1.0),
                   (1.0,  0.0, 1.0)]}

white2green = matplotlib.colors.LinearSegmentedColormap('my_colormap2', cdict, 256)

def plot_scalar(g: Lattice, scalar: np.ndarray, ax = None, resolution : int = 100, method : str = 'cubic', cmap = white2green, vmin : float = None, vmax : float = None):
    """Plot a scalar field on a 2d amorphous lattice

    Args:
        g (Lattice): The lattice object to plot over
        scalar (np.ndarray) shape: (len(g.vertices),)): The scalar field to plot.
        ax (axis, optional): Optional axis to plot to.
        resolution (int, optional): Number of points to interpolate to in x and y. Defaults to 100.
        method (str, optional): method{‘linear’, ‘nearest’, ‘cubic’}, see docs for scipy.interpolate.griddata. Defaults to 'cubic'.
        cmap (colormap, optional): Color map to use, either string name or object. Defaults to a custom white to green fade.
        vmin (float, optional): arg to pcolormesh. Defaults to None.
        vmax (float, optional): arg to pcolormesh. Defaults to None.

    Returns:
        [type]: [description]
    """
    if ax is None: ax = plt.gca()
    if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        
    x = np.linspace(0, 1, resolution)
    y = np.linspace(0, 1, resolution)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    def to_points(xv, yv): return np.array([xv, yv]).transpose((1,2,0))
    s = scipy.interpolate.griddata(g.vertices, scalar, to_points(xv,yv), method = method, rescale = True)
    ax.pcolormesh(xv, yv, s, cmap = cmap, vmin = 0, vmax = 1)

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

    highlight_edge_neighbours = np.array([3 for i in range(g.adjacency.shape[0])])
    highlight_edge_neighbours[ordered_edge_indices] = [0, 1, 2]
 
    ax.hlines(y = vertex[1], xmin = vertex[0], xmax = ax.get_ylim()[1], linestyle = 'dotted', alpha = 0.5, color = 'k')

    plot_lattice(g, edge_labels = highlight_edge_neighbours, scatter_args = dict(color = vertex_colors), ax = ax)
    
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
    for i, e in enumerate(g.edges.indices): 
        midpoint = g.vertices[e].mean(axis = 0)
        if not np.any(g.edges.crossing[i]) != 0:
            ax.text(*(midpoint+offset), f"{i}", color = 'g')