import numpy as np
import scipy
import scipy.interpolate
import itertools
import matplotlib
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib import pyplot as plt

from . import graph_utils

from .lattice import LatticeException
from .graph_utils import vertex_neighbours, clockwise_edges_about
from .voronization import generate_point_array, Lattice


#### New plotting interface ####

def plot_vertices(lattice : Lattice,
                  labels : np.ndarray = 0, 
                  color_scheme : np.ndarray = 'black',
                  subset : np.ndarray = slice(None, None, None), 
                  ax = None,
                  **kwargs):
    """Plot the vertices of a lattice. 
    This uses matplotlib.pyplot.scatter under the hood and you may 
    pass in any keyword to be passed along to scatter.
    You can use this to, for instance, plot some of the vertices as
     triangles and some as squares.

    :param lattice: The lattice to use.
    :type lattice: Lattice
    :param labels: int or array of ints specifying the colors, defaults to 0. May be the same size as the vertices or of the subset.
    :type labels: np.ndarray, optional
    :param color_scheme: List or array of colors, defaults to ['black', ]
    :type color_scheme: np.ndarray, optional
    :param subset: An array of indices, boolean array or slice that selects which elements to plot, defaults to plotting all. 
    :type subset: np.ndarray, optional
    :param ax: The axis to plot on, defaults to plt.gca()
    :type subset: axis, optional
    """ 
    labels, colors, color_scheme, subset, ax, transform = _process_plot_args(lattice, ax, labels, color_scheme, subset, lattice.n_vertices)          

    args = dict(
        x = lattice.vertices.positions[subset,0],
        y = lattice.vertices.positions[subset,1],
        c = colors,
        zorder = 3
    )
    args.update(**kwargs) #doing this means the user can override zorder
    ax.scatter(**args, transform = transform)
    return ax

def plot_edges(lattice : Lattice,
                    labels : np.ndarray = 0,
                    color_scheme : np.ndarray = ['k','r','b'],
                    subset : np.ndarray = slice(None, None, None), 
                    directions : np.ndarray = None,
                    ax = None,
                    arrow_head_length = None,
                    **kwargs):
    """
    Plot the edges of a lattice with optional arrows.
    This uses matplotlib.collections.LineColection under the hood and you may 
    pass in any keyword to be passed along to it. 
    Note that arrays for alpha or linestyle don't currently work since they would have to be tiled correctly, and are not currently.

    If directions is not none, arrows are plotted from the first vertex to the second unless direction[i] == -1

    :param lattice: The lattice to use.
    :type lattice: Lattice
    :param labels: int or array of ints specifying the colors, defaults to 0. May be the same size as the vertices or of the subset.
    :type labels: np.ndarray, optional
    :param color_scheme: List or array of colors, defaults to ['black', ]
    :type color_scheme: np.ndarray, optional
    :param subset: An array of indices, boolean array or slice that selects which elements to plot, defaults to plotting all. 
    :type subset: np.ndarray, optional
    :param directions: An array of arrow directions +/-1, defaults to None.
    :type directions: np.ndarray, optional
    :param ax: The axis to plot on, defaults to plt.gca()
    :type subset: axis, optional
    """ 
    labels, colors, color_scheme, subset, ax, transform = _process_plot_args(lattice, ax, labels, color_scheme, subset, lattice.n_edges)  

    edge_colors = np.tile(colors, 9)
    edge_vertices = lattice.vertices.positions[lattice.edges.indices[subset]]
    edge_vertices[:, 0, :] -= lattice.edges.crossing[subset]
    
    unit_cell_vectors = generate_point_array(np.array([0,0]), padding = 1)[:, None, None, :] #shape (9, 2) -> (9, 1, 1, 2)
    replicated_edges = edge_vertices[None,...] + unit_cell_vectors #shape (n_edges, 2, 2) -> (9, n_edges, 2, 2)
    replicated_edges =  replicated_edges.reshape((-1,2,2)) #shape (9, n_edges, 2, 2) -> (9*n_edges, 2, 2)
    
    vis = _lines_cross_unit_cell(replicated_edges) | _line_fully_in_unit_cell(replicated_edges)
    # print(edge_colors.shape, replicated_edges.shape, vis.shape)
    lc = LineCollection(replicated_edges[vis, ...], colors = edge_colors[vis], transform = transform, **kwargs)
    ax.add_collection(lc)

    if directions is not None:
        directions = _broadcast_args(directions, subset, lattice.n_edges, dtype = int)
        directions = np.tile(directions, 9)
        _plot_edge_arrows(ax, edge_colors[vis],replicated_edges[vis, ...],directions[vis], lc, lattice.unit_cell)


    return ax

def plot_plaquettes(lattice : Lattice,
                    labels : np.ndarray = 0,
                    color_scheme : np.ndarray = ['r','b','k'],
                    subset : np.ndarray = slice(None, None, None), 
                    ax = None,
                    **kwargs):
    """
    Plot the plaquettes of a lattice. 
    This uses matplotlib.collections.PolyColection under the hood and you may 
    pass in any keyword to be passed along to it. 
    Note that currently the calls are done per plaquette so you can't for instance have multiple alpha values.
    Adding a color argument overides the color_scheme and labels.

    :param lattice: The lattice to use.
    :type lattice: Lattice
    :param labels: int or array of ints specifying the colors, defaults to 0. May be the same size as the vertices or of the subset.
    :type labels: np.ndarray, optional
    :param color_scheme: List or array of colors, defaults to ['black', ]
    :type color_scheme: np.ndarray, optional
    :param subset: An array of indices, boolean array or slice that selects which elements to plot, defaults to plotting all. 
    :type subset: np.ndarray, optional
    :param ax: The axis to plot on, defaults to plt.gca()
    :type subset: axis, optional
    """ 

    labels, colors, color_scheme, subset, ax, transform = _process_plot_args(lattice, ax, labels, color_scheme, subset, lattice.n_plaquettes)           

    indices = np.arange(lattice.n_plaquettes)[subset]
    plaquettes = lattice.plaquettes[subset]

    collections = []
    for i, color, p in zip(indices, colors, plaquettes):
        
        #get the edge vectors going anticlockwise around the plaquette
        vectors = lattice.edges.vectors[p.edges] * p.directions[:, None]

        #use those to construct the points around the plaquette ignoring PBC
        points = lattice.vertices.positions[p.vertices[0]] + np.cumsum(vectors, axis = 0)

        #represent the line segments of p with (start_point, end_point) tuples
        lines = np.array(list(zip(points, np.roll(points, -1, axis = 0))))

        #find out if the polygon crosses the x=0,1 or y=0,1 lines
        partially_inside = np.any(_lines_cross_any_cell_boundary(lines), axis = 0)
        
        #figure out how this polygon needs to be replicated
        #crossx[0] == True means one of the line segments of the polygon crossed x = 0 for example
        crossx, crossy = partially_inside.T
        padx_bool = np.array([crossx[1], 1, crossx[0]], dtype = bool)
        padx = np.array([-1, 0, 1])[padx_bool]

        pady_bool = np.array([crossy[1], 1, crossy[0]], dtype = bool)
        pady = np.array([-1, 0, 1])[pady_bool]
            
        replicated_polygons = _replicate_polygon(points, padx, pady)
        
        # allow the user to overide color args
        poly_args = dict(color = color, transform = transform)
        poly_args.update(kwargs)

        collection = PolyCollection(replicated_polygons, **poly_args)
        collections.append(collection)
        #one could add all these up into one huge polycollection but it doesn't seem to be any faster
        ax.add_collection(collection)
    return collections

def plot_dual(lattice, subset = slice(None,None), **kwargs):
    """Given a lattice, plot the edges in it's dual or a subset of them, other args are passed through to plot_edges.

    :param lattice: The lattice to use.
    :type lattice: Lattice
    :param subset: a subset of edges to plot, defaults to all.
    :type subset: slice, boolean array or indices, optional
    :param ax: the ax to plot to, defaults to None
    :type ax: axis, optional

    :return: The dual lattice represented as a second Lattice object.
    :rtype: Lattice
    """    
    st_as_lattice = graph_utils.make_dual(lattice, subset)
    plot_edges(st_as_lattice, **kwargs)
    return st_as_lattice

def _plot_edge_arrows(ax, colors, edges, directions, linecollection, unit_cell, head_length = None):
    n_edges = edges.shape[0]
    edges = unit_cell.transform(edges.reshape(-1,2)).reshape(n_edges, 2, 2)
    linewidth = linecollection.get_linewidths()[0] #currently don't support multiple linewidths
    for color, (end, start), dir in zip(colors, edges, directions):
        start, end = [start, end][::dir]
        center = 1/2 * (end + start)
        length = np.linalg.norm(end - start)
        head_length = head_length or min(0.2 * length, 0.02 * linewidth / 1.5)
        direction = head_length * (start - end) / length
        arrow_start = center - direction
        ax.arrow(x=arrow_start[0], y=arrow_start[1], dx=direction[0], dy=direction[1],
                color = color, head_width = head_length, head_length = head_length,
                width = 0, zorder = 4, head_starts_at_zero = True, length_includes_head = True)


def _broadcast_args(arg, subset, N, dtype = int):
    """Normalise an argument for plotting that can take three forms:
        1) a single thing of type [dtype]
        2) an array of size l.n_vertices, l.n_edges or l.n_plaquettes
        3) a smaller array that matches the subset
    Returns an array of type 3
    """
    # Fix 1) if it's just a single int, broadcast it to the size of the lattice.
    if isinstance(arg, dtype): arg = np.full(N, arg, dtype = dtype)

    # make sure it's a numpy array (of the right type) and not a list.
    arg = np.array(arg).astype(dtype)

    # if it refers to the entire lattice, subset it down
    subset_size = np.sum(np.ones(N)[subset], dtype = int) 
    if arg.shape[0] == N: arg = arg[subset]
    elif arg.shape[0] == subset_size: arg = arg
    else: raise ValueError(f"Argument shape {arg.shape} should be either lattice.n_* ({N}) or the size of the subset ({subset_size})")

    return arg

def _scale_nicely(lattice, ax):
    square = np.array([[(0,0), (1,0)] , [(1,0), (1,1)], [(1,1), (0,1)], [(0,1), (0,0)]])
    transformed_square = lattice.unit_cell.transform(square.reshape(-2,2)).reshape(4,2,2)
    lx, hx = np.min(transformed_square[...,0]), np.max(transformed_square[...,0])
    ly, hy = np.min(transformed_square[...,1]), np.max(transformed_square[...,1])
    extent = max((hx - lx)/2, (hy - ly)/2)
    mid_x, mid_y = (hx + lx)/2, (hy + ly)/2
    ax.set(xlim = (mid_x - extent,mid_x + extent), ylim = (mid_y - extent,mid_y + extent))

def _process_plot_args(lattice, ax, labels, color_scheme, subset, N):
    """
    Deals with housekeeping operations common to all plotting functions. 
    Specifically:
        Broadcast single values to be the size of the lattice.
        Allow labels to refer to either the whole lattice or the subset.
        Check if ax is none, and if so, create one.
    """
    # if color_scheme is a string, broadcast to the size of the lattice.
    if not hasattr(Lattice, 'unit_cell'): Lattice.unit_cell = matplotlib.transforms.IdentityTransform()
    if isinstance(color_scheme, str): color_scheme = [color_scheme, ]
    color_scheme = np.array(color_scheme)
    subset = np.arange(N)[subset]
    labels = _broadcast_args(labels, subset, N, dtype = int)
    
    colors = color_scheme[labels]

    if ax is None: ax = plt.gca()
    _scale_nicely(lattice, ax)

    transform = lattice.unit_cell + ax.transData

    return labels, colors, color_scheme, subset, ax, transform

#### Functions to plot index labels of the vertices ####

def plot_vertex_indices(lattice : Lattice):
    if ax is None: ax = plt.gca()
    ax.set(xlim = (0,1), ylim = (0,1))
    pass

#TODO: Make this work with edges that cross the boundaries
def plot_edge_indices(g, ax = None, offset = 0.01):
    """
    Plot the indices of the edges on a graph
    """
    if ax is None: ax = plt.gca()
    for i, e in enumerate(g.edges.indices): 
        midpoint = g.vertices.positions[e].mean(axis = 0)
        if not np.any(g.edges.crossing[i]) != 0:
            ax.text(*(midpoint+offset), f"{i}", color = 'g')
  

############### Old plotting interface + internal stuff ##############################

import warnings
import functools

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        return func(*args, **kwargs)
    return new_func

def peru_friendly_color_scheme(n_colors: int):
    if n_colors == 2:
        scheme = ['r', 'b']
    elif n_colors == 3:
        scheme = ['b', 'y', 'lightgrey' ]
    elif n_colors == 4:
        scheme = ['r','b', 'y', 'lightgrey']
    else:
        raise LatticeException('Honestly you cannot possibly expect someone as colorblind as me to be able to find FIVE colors I can tell apart, thats ridiculous')

    return scheme

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
    so then we check that 0 < point_y(t_x0) < 1 and if so then then we know the line segment crosses the x = 0 boundary
    Args:
        lines (np.ndarray): lines[i, j, :] is the jth point of the ith line

    Returns:
        np.ndarray: out[i] == 1 if line[i] is fully inside the unit cell

    :param lines: : Represents lines by pairs of points
    :type lines: np.ndarray shape (n_lines, 2, 2)
    :return: cross[i] is true is the corresponding line crossed any of the 4 boundaries of the unit cell
    :rtype: np.ndarray shape (n_lines,)
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
    return np.any(cross, axis = (1,2))

@deprecated
def plot_lattice(lattice, ax = None, 
                 edge_labels = None, edge_color_scheme = ['r','g','b','k'],
                 vertex_labels = None, vertex_color_scheme = ['r','b','k'],
                 edge_arrows = False, edge_index_labels = False, bond_signs = None,
                 scatter_args = None):
    """Plots a 2d graph. Optionally with colored edges or vertices.

    Args:
        lattice (Lattice): A koala lattice dataclass containing the vertices, adjacency and adjacency_crossing
        ax (matplotlib axis, optional): Axis to plot to. Defaults to plt.gca().
        edge_labels (np.ndarray, optional): A list of integer edge labels, length must be the same as adjacency, If None, then all edges are plotted in black. Defaults to None.
        edge_color_scheme (list, optional): List of matplotlib  color strings for edge coloring. Defaults to ['r','g','b'].
        vertex_labels (np.ndarray, optional): A list of labels for coloring the vertices, if None, vertices are not plotted. Defaults to None.
        vertex_color_scheme (list, optional): List of matplotlib  color strings for vertex coloring. Defaults to ['r','b'].
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
        edge_colors = np.full(fill_value = edge_color_scheme[-1], shape = adjacency.shape[0]*9)
    
    edge_vertices = vertices.positions[adjacency] 
    edge_vertices[:, 0, :] -= adjacency_crossing
    
    unit_cell_vectors = generate_point_array(np.array([0,0]), padding = 1)[:, None, None, :] #shape (9, 2) -> (9, 1, 1, 2)
    replicated_edges = edge_vertices[None,...] + unit_cell_vectors #shape (n_edges, 2, 2) -> (9, n_edges, 2, 2)
    replicated_edges =  replicated_edges.reshape((-1,2,2)) #shape (9, n_edges, 2, 2) -> (9*n_edges, 2, 2)
    
    vis = _lines_cross_unit_cell(replicated_edges) | _line_fully_in_unit_cell(replicated_edges)
    lc = LineCollection(replicated_edges[vis, ...], colors = edge_colors[vis])
    ax.add_collection(lc)
    
    if np.any(edge_arrows) or np.any(edge_index_labels):
        original_edge_indices = np.tile(np.arange(lattice.n_edges), 9)
        if isinstance(edge_arrows, bool): edge_arrows = np.full(fill_value = edge_arrows, shape = lattice.n_edges)
        if isinstance(edge_index_labels, bool): edge_index_labels = np.full(fill_value = edge_index_labels, shape = lattice.n_edges)
        if bond_signs is None: bond_signs = np.ones(lattice.n_edges, dtype = int)

        edge_arrows = np.tile(edge_arrows, 9)
        edge_index_labels = np.tile(edge_index_labels, 9)
        bond_signs = np.tile(bond_signs, 9)

        for i, color, (start, end), bond_sign in zip(original_edge_indices[vis], edge_colors[vis], replicated_edges[vis, ...], bond_signs[vis]):
            start, end = [start, end][::bond_sign]
            center = 1/2 * (end + start)
            length = np.linalg.norm(end - start)
            head_length = min(0.2 * length, 0.02)
            direction = head_length * (start - end) / length
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
        vertices.positions[:,0],
        vertices.positions[:,1],
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
        scalar (np.ndarray) shape: (len(g.vertices.positions),)): The scalar field to plot.
        ax (axis, optional): Optional axis to plot to.
        resolution (int, optional): Number of points to interpolate to in x and y. Defaults to 100.
        method (str, optional): method{'linear', 'nearest', 'cubic'}, see docs for scipy.interpolate.griddata. Defaults to 'cubic'.
        cmap (colormap, optional): Color map to use, either string name or object. Defaults to a custom white to green fade.
        vmin (float, optional): arg to pcolormesh. Defaults to None.
        vmax (float, optional): arg to pcolormesh. Defaults to None.

    Returns:
        [type]: [description]
    """
    if ax is None: ax = plt.gca()
    if isinstance(cmap, str): cmap = plt.get_cmap(cmap)
        
    x = np.linspace(0, 1, resolution, endpoint = True)
    y = np.linspace(0, 1, resolution, endpoint = True)

    xv, yv = np.meshgrid(x, y, sparse=False, indexing='ij')

    def to_points(xv, yv): return np.array([xv, yv]).transpose((1,2,0))
    s = scipy.interpolate.griddata(g.vertices.positions, scalar, to_points(xv,yv), method = method, rescale = True)
    ax.pcolormesh(xv, yv, s, cmap = cmap, vmin = vmin, vmax = vmax)
    return xv, yv, s

def plot_degeneracy_breaking(vertex_i, g, ax = None):
    """
    Companion function to graph_utils.clockwise_edges_about, 
    plots the edges on an axis with labelled angles and the positive x axis as a dotted line
    """
    if ax is None: ax = plt.gca()
    #we choose the 0th vertex
    vertex = g.vertices.positions[vertex_i]
    
    vertex_colors = np.array(['k' for _ in g.vertices.positions])
    
    #label the main vertex red
    vertex_colors[vertex_i] = 'r'
    
    #label its neighbours green
    vertex_colors[vertex_neighbours(g, vertex_i)[0]] = 'g'

    #color the edges in a clockwise fashion
    ordered_edge_indices = clockwise_edges_about(vertex_i, g)

    highlight_edge_neighbours = np.array([3 for i in range(g.edges.indices.shape[0])])
    highlight_edge_neighbours[ordered_edge_indices] = [0, 1, 2]
 
    ax.hlines(y = vertex[1], xmin = vertex[0], xmax = ax.get_ylim()[1], linestyle = 'dotted', alpha = 0.5, color = 'k')

    plot_lattice(g, edge_labels = highlight_edge_neighbours, scatter_args = dict(color = vertex_colors), ax = ax)
    
def plot_vertex_indices(g, ax = None, offset = 0.01):
    """
    Plot the indices of the vertices on a graph
    """
    if ax is None: ax = plt.gca()
    for i, v in enumerate(g.vertices.positions): ax.text(*(v+offset), f"{i}")

def _lines_cross_any_cell_boundary(lines : np.ndarray) -> np.ndarray:
    """
    Used in plot_plaquettes to determine which plaquettes need to be plotted multiple times.
    Similar to the above, but tells you if the line cross y=0,1 and x=0,1 at any point rather than just the unit cell.

    :param lines: : Represents lines by pairs of points
    :type lines: np.ndarray shape (n_lines, 2, 2)
    :param which: If True output information about which lines were crossed, defaults to False
    :type which: bool, optional
    :return: If which is false, cross[i] is true is the corresponding line crossed any of the 4 boundaries of the unit cell,
    if which is true, cross[i] = ((x=0, y=0), (x=1, y=1)) where each entry if true if line[i] crossed the corresponding boundary.
    :rtype: np.ndarray shape (n_lines,) or (n_lines, 2, 2) if which is True
    """
    start, end = lines[:, 0, None, :], lines[:, 1, None, :] #start.shape = (n_lines, 1, 2)
    l = np.array([[0,0],[1,1]])[None, :, :] #shape (1, 2, 2)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (l - end) / (start - end) #shape (n_lines, 2, 2)

    t[~np.isfinite(t)] = 0.5*(l == end)[~np.isfinite(t)] 

    other_coord_value_at_t = start[..., ::-1] * t + (1 - t) * end[..., ::-1]
    cross = (0 < t) & (t <= 1) 
    return cross

def _replicate_polygon(polygon, padx, pady):
    """Used in plot plaquettes to periodically tile a polygon in the x or y directions"""
    dxdy = np.array(list(itertools.product(padx, pady)))
    return polygon[None, ...] + dxdy[:, None, :]

def cross_product_2d(a,b): return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0] 

def line_intersection(lines, lines2, abs_tolerance = 1e-14, full_output = False):
    """
    Generic Line to line intersection function
    
    :param lines: The first set of lines, with shape [n_lines, 2, 2] where lines[i, 0, :] is the first point of the ith line
    :type line: np.ndarray
    
    :param lines2: The second set of lines to intersect with. Shape [n_lines2, 2, 2]
    :type line2: np.ndarray
    
    :param abs_tolerance: The absolute floating point tolerance to use for the parallel and colinear special cases.
    :type abs_tolerance: float
    
    :return: intersection, boolean array shape [n_lines, n_lines2]
    :rtype: np.ndarray
    """
    
    s1 = lines[:, 0, :][:, None]
    e1 = lines[:, 1, :][:, None]
    d1 = e1 - s1 #shape of s1, e1, d1 = (n_lines, 1, 2)
    
    s2 = lines2[:, 0, :][None, :]
    e2 = lines2[:, 1, :][None, :]
    d2 = e2 - s2 #shape of s2, e2, d2 = (1, n_lines2, 2)

    d2_cross_d1 = cross_product_2d(d2, d1) # 0 if lines are parallel
    displacement_cross_d1 = cross_product_2d((s1 - s2), d1) # when parallel this is 0 if lines are also colinear
    
    with np.errstate(divide='ignore', invalid='ignore'):

        #only works if not parallel, hence the use of np.errstate
        t1 = cross_product_2d((s1 - s2), d1) / cross_product_2d(d2, d1)
        t2 = cross_product_2d((s2 - s1), d2) / cross_product_2d(d1, d2)
        
        #used for the colinear case
        t_star1 = ((s1 - s2) / d1)[..., 0]
        t_star2 = ((s1 - s2) / d2)[..., 0]
    
    #add some floating point tolerance
    are_parallel = np.abs(d2_cross_d1) < abs_tolerance
    are_colinear = are_parallel & (np.abs(displacement_cross_d1) < abs_tolerance)
    
    t_in_range = (0 <= t1) & (t1 <= 1) & (0 <= t2) & (t2 <= 1)
    t_star_in_range = ((-1 <= t_star1) & (t_star1 <= 1)) | ((-1 <= t_star2) & (t_star2 <= 1))

    intersect = np.select(
        condlist = [~are_parallel, ~are_colinear, are_colinear], 
        choicelist = [t_in_range, False, t_star_in_range],
        default=False
    )
    if full_output: return intersect, are_parallel, are_colinear
    return intersect