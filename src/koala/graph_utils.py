############################################################################
#              Generic routines for doing things on graphs                 #
#                                                                          #
############################################################################
import numpy as np
from .voronization import Lattice

def vertex_neighbours(vertex_i, adjacency):
    """
    Return the neighbouring nodes of a point

    Args:
        vertex_i: int the index into vertices of the node we want the neighbours of
        adjacency: (M, 2) A list of pairs of vertices representing edges
    Returns:
        vertex_indices: (k), the indices into vertices of the neighbours
        edge_indices: (k), the indices into adjacency of the edges that link vertex_i to its neighbours 
        
    Note that previous version of this function broke the expectation that edge_indices[i] is the edge that links 
    vertex_i to vertex_indices[i], make sure to preserve this property.
    """
    edge_indices = np.where(np.any(vertex_i == adjacency, axis=-1))[0]
    edges = adjacency[edge_indices]
    vertex_indices = edges[edges != vertex_i]
    assert(vertex_indices.shape == edge_indices.shape)
    return vertex_indices, edge_indices

def edge_neighbours(edge_i, adjacency):
    """
    Return the neighbouring edges of an edge (the edges connected to the same nodes as this edge)

    Args:
        edge_i: int the index into vertices of the node we want the neighbours of
        adjacency: (M, 2) A list of pairs of vertices representing edges
    Returns:
        edge_indices: (k), the indices into adjacency of the edges that link vertex_i to its neighbours 
    """
    edge = adjacency[edge_i]
    v1 = edge[0]
    v2 = edge[1]
    mask = np.any(v1 == adjacency, axis = -1) | np.any(v2 == adjacency, axis=-1)
    mask[edge_i] = False #not a neighbour of itself
    return np.where(mask)[0]

def clockwise_edges_about(vertex_i : int, g : Lattice) -> np.ndarray:
    """
    Finds the edges that border vertex_i, orders them clockwise starting from the positive x axis
    and returns those indices in order. Use this to break the degeneracy of graph coloring.

    Args:
        vertex_i (int): int the index into g.vertices of the node we want to use. Generally use 0
        g (Lattice): a graph object with keys vertices, adjacency, adjacency_crossing
    Returns:
        ordered_edge_indices: np.ndarray (n_neighbours_of_vertex_i) ordered indices of the edges. 
    """
    #get the edges and vertices around vertex 0
    vertex_indices, edge_indices = vertex_neighbours(vertex_i, g.adjacency)
    edge_vectors = get_edge_vectors(vertex_i, edge_indices, g)

    #order them clockwise from the positive x axis
    angles = np.arctan2(edge_vectors[:, 1], edge_vectors[:,0])
    angles = np.where(angles > 0, angles, 2*np.pi + angles) #move from [-pi, pi] to [0, 2*pi]
    ordering = np.argsort(angles)
    ordered_edge_indices = edge_indices[ordering]
    return ordered_edge_indices

def get_edge_vectors(vertex_i : int, edge_indices : np.ndarray, l : Lattice) -> np.ndarray:
    """
    Get the vector starting from vertex_i along edge_i, taking into account boundary conditions
    Args:
        vertex_i (int): the index of the vertex
        edge_i (int): the index of the edge
        lattice (Lattice): the lattice to use

    Returns:
        np.ndarray (2,): 
    """
    #this is a bit nontrivial, g.adjacency_crossing tells us if the edge crossed into another unit cell but 
    #it is directional, hence we need to check for each edge if vertex_i was the first of second vertex stored
    #the next few lines do that so we can add g.adjacency_crossing with the right sign
    edges = l.adjacency[edge_indices]
    start_or_end = (edges != vertex_i)[:, 1] #this is true if vertex_i starts the edge and false if it ends it
    other_vertex_indices = np.take_along_axis(edges, start_or_end[:, None].astype(int), axis = 1).squeeze() #this gets the index of the other end of each edge
    offset_sign = (2*start_or_end - 1) #now it's +/-1
    
    #get the vectors along the edges
    return l.vertices[other_vertex_indices] - l.vertices[vertex_i][None, :] + offset_sign[:, None] * l.adjacency_crossing[edge_indices]
