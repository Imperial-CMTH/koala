############################################################################
# Routines for generating periodic Voronizations of arbitrary point sets   #
# defined on [0, 1]^n                                                      #
############################################################################

import numpy as np
import numpy.typing as npt
from scipy.spatial import Voronoi, KDTree
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt
import itertools

from .lattice import Lattice


def points_near_unit_image(vor, idx):
    """Find the vertex in unit cell near the image of the vertex with index
    idx. Assumes a voronoi diagram vor generated from a point set in a
    3x3 replicated geometry spanning [-1, 2].
    """
    point = vor.vertices[idx]
    point = point % 1
    return np.argmin(np.linalg.norm(vor.vertices - point, axis=-1))


def generate_point_array(points, padding=1):
    """"
    Repeats points into a 3x3, 5x5 etc grid.
    Args:
        points: np.ndarray (n_points, 2) - Array of points to be replicated.
        padding: int - how many layers to add, 1 -> 3x3, 2 -> 5x5 etc.
    Returns:
        points: np.ndarray (n_points * (2*padding+1)**2, 2) - Array of replicated points.
    """
    linear_pad = np.arange(-padding, padding + 1)  #(-1,0,1), (-2,-1,0,1,2) etc
    dxdy = np.array(list(itertools.product(
        linear_pad, repeat=2)))  #the above but running over all pairs in 2d
    return np.concatenate(points[None, ...] + dxdy[:, None, :])


square = np.array([
    [[0, 0], [0, 1]],
    [[0, 1], [1, 1]],
    [[1, 1], [1, 0]],
    [[1, 0], [0, 0]],
])


def plot_lines(ax, lines, **kwargs):
    lc = LineCollection(lines, **kwargs)
    ax.add_collection(lc)


#TODO - Factor out debugging plot code
def generate_lattice(original_points: npt.NDArray[np.floating],
                     debug_plot: bool = False,
                     shift_vertices=True) -> Lattice:
    """Generate a `Lattice` object with periodic boundary conditions from an initial set of points
        ``original_points`` whose voronoi diagram constitute the vertices and edges of the lattice.

    Args:
        original_points (npt.NDArray[np.floating]): Points used to
            generating lattice voronoi diagram
        debug_plot (bool, optional): Output intermediate debugging plots
            to check PBC logic, defaults to False
        shift_vertices (bool, optional): If True, shift all the vertices
            to the center of the delaunay triangle they sit in - stops
            any vertices getting too close together and makes things a
            bit neater

    Returns:
        Lattice object: Lattice object containing information about
        vertices, edges, and
    """
    padding = 1 if original_points.shape[
        0] > 10 else 2  #how many layers to add, 1 -> 3x3, 2 -> 5x5 etc.
    #Create periodic boundary conditions by replicating the point set across 3x3 unit cells
    points = generate_point_array(original_points, padding)

    # Generate Voronoi diagram w/ SciPy
    vor = Voronoi(points)
    ridge_indices = np.array(vor.ridge_vertices)

    # shift the positions of the vertices to the center of the delaunay triangle they are in
    if shift_vertices:
        ridge_points = np.array(vor.ridge_points)

        # find the edges that touch each vertex
        vertex_adjacent_edges = []
        for vertex_number in range(vor.vertices.shape[0]):
            v_edges = np.nonzero((ridge_indices[:, 0] == vertex_number) +
                                 (ridge_indices[:, 1] == vertex_number))[0]
            vertex_adjacent_edges.append(v_edges)

        # find the indices of the points that live in the three regions around the vertex
        delaunay_edges = ridge_points[vertex_adjacent_edges]
        points_flat = ridge_points[vertex_adjacent_edges].reshape(
            delaunay_edges.shape[0], 6)
        delaunay_point_indices = np.apply_along_axis(np.unique, 1, points_flat)
        delaunay_points = vor.points[delaunay_point_indices]

        # average these to get the delaunay centers
        delaunay_center = np.sum(delaunay_points, 1) / 3
        vor.vertices = delaunay_center

    ridge_vertices = vor.vertices[ridge_indices]

    #count how many of each ridges points fall in the unit cell
    ridges_vertices_in_unit_cell = np.sum(
        np.all((0 < ridge_vertices) & (ridge_vertices <= 1),
               axis=2),  #select points that are in the unit cell
        axis=1)

    #the indices of ridges either fully inside, or half inside the unit cell
    # the & np.all(ridge_indices != -1, axis = -1) deals with the case where (-1,-1) into vertices happens to lie within the unit cell
    #so we have to check for this in both crossing_ridges and outer_ridges
    finite = np.all(
        ridge_indices != -1, axis=-1
    )  #this ignores ridges at the boundary that have no second vertex (scipy puts (-1,-1) instead)
    inside_ridges = ridge_indices[(ridges_vertices_in_unit_cell == 2) & finite]
    crossing_ridges = ridge_indices[(ridges_vertices_in_unit_cell == 1) &
                                    finite]
    outer_ridges = ridge_indices[(ridges_vertices_in_unit_cell == 0) & finite]

    if debug_plot:
        _, axes = plt.subplots(ncols=2, figsize=(20, 10))
        axes[0].scatter(*points.T, s=0.1, color='k')
        plot_lines(axes[0], vor.vertices[inside_ridges])
        plot_lines(axes[0], vor.vertices[crossing_ridges], colors='r')
        plot_lines(axes[0],
                   vor.vertices[outer_ridges],
                   colors='grey',
                   alpha=0.5)
        for v in generate_point_array(np.array(0), padding=padding):
            plot_lines(axes[0],
                       square + v[..., :],
                       linestyle='--',
                       colors='k',
                       alpha=0.2)
        axes[0].set(xlim=(-padding, padding + 1), ylim=(-padding, padding + 1))

    #record if each edge crossed a cell boundary in the x or y direction
    crossing_ridges = np.sort(
        crossing_ridges, axis=-1
    )  #this sort has to happen because we need it to set the direction of the adjacency_crossing vector
    crossing_ridge_vertices = vor.vertices[crossing_ridges]
    adjacency_crossing = np.floor(crossing_ridge_vertices[:, 1, :]) - np.floor(
        crossing_ridge_vertices[:, 0, :])
    adjacency_crossing = adjacency_crossing.astype(int)

    # Replace the half inside ridges with their corresponding indices in the unit cell
    kdtree = KDTree(vor.vertices)
    _, crossing_ridges = kdtree.query(vor.vertices[crossing_ridges] % 1, k=1)

    # We make a temp sorted version of crossing_ridges and adjacency_crossing just
    #  to get the indices from np.unique
    # we need adjacency_crossing because technically we can have two edges between the same
    # vertices if they go different ways around the torus
    # and the sign of adjacency flips if swap the ordering of i,j in crossing_ridges
    idx = np.argsort(crossing_ridges, axis=-1)
    sorted_crossing_ridges = np.take_along_axis(crossing_ridges, idx, axis=-1)
    swapped = idx[:, 0]  #if we swapped the ordering of that ridge
    sorted_adjacency_crossing = adjacency_crossing * (
        2 * swapped - 1)[:, None]  #then we flip the sign of this
    edge_key = np.concatenate(
        [sorted_crossing_ridges, sorted_adjacency_crossing], axis=-1)

    #NB yes we are sorting twice, we do actually need to
    #deduplicate by first sorting the index order and then calling unique
    #use the returned indices to also dedup the adjacency_crossing array
    _, idx = np.unique(edge_key, axis=0, return_index=True)
    crossing_ridges = crossing_ridges[idx]
    adjacency_crossing = adjacency_crossing[idx]

    if debug_plot:
        plot_lines(axes[1], vor.vertices[inside_ridges])
        for i, s in [[0, -1], [1, +1]]:
            a = vor.vertices[crossing_ridges].copy()
            a[:, i, :] = a[:, i, :] + s * adjacency_crossing
            plot_lines(axes[1], a, colors='r')
        axes[1].set(xlim=(-1, 2), ylim=(-1, 2))

    pbc_ridges = np.concatenate([inside_ridges, crossing_ridges])
    adjacency_crossing = np.concatenate(
        [np.zeros_like(inside_ridges), adjacency_crossing])

    indices_to_copy = list(set(pbc_ridges.flatten()))
    # list of vertices in same order as index list
    new_vertices = vor.vertices[indices_to_copy]

    # map old indices to position in index list
    old_idx_to_new = {old: new for new, old in enumerate(indices_to_copy)}
    idx_mapper = np.vectorize(old_idx_to_new.get)

    new_pbc_ridges = idx_mapper(pbc_ridges)
    # assert np.all(np.bincount(new_pbc_ridges.flatten()) == 3), """You've hit an edge case where for low point densities where a 3x3 unit cell is not enough to prevent one of the ridges being infinite in length, try again."""

    return Lattice(vertices=new_vertices,
                   edge_indices=new_pbc_ridges,
                   edge_crossing=adjacency_crossing)
