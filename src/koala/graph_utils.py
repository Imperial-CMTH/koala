############################################################################
#              Generic routines for doing things on graphs                 #
#                                                                          #
############################################################################
import numpy as np

def vertex_neighbours(vertex_i, adjacency):
    """
    Return the neighbouring nodes of a point

    Args:
        vertex_i: int the index into vertices of the node we want the neighbours of
        adjacency: (M, 2) A list of pairs of vertices representing edges
    Returns:
        vertex_indices: (k), the indices into vertices of the neighbours
        edge_indices: (k), the indices into adjacency of the edges that link vertex_i to its neighbours 
    """
    edge_indices = np.where(np.any(vertex_i == adjacency, axis=-1))[0]
    vertex_indices = np.unique(adjacency[edge_indices])
    vertex_indices = vertex_indices[np.logical_not(vertex_indices == vertex_i)]
                          
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

def order_clockwise(vectors): 
    """
    return the indices that order vectors clockwise starting from (0,0)

    Args:
        vectors: shape (N, 2) 
    Returns:
        indices: shape (N), the indices that order vecs clockwise starting from vecs[0]
    """
    angles = np.arctan2(vectors[:, 1], vectors[:,0])
    return np.argsort(-angles)

