############################################################################
#              Generic routines for doing things on graphs                 #
#                                                                          #
############################################################################
import numpy as np
from .lattice import Lattice, INVALID
from typing import Tuple

def plaquette_spanning_tree(lattice: Lattice, shortest_edges_only = True):
    """Given a lattice this returns a list of edges that form a spanning tree over all the plaquettes (aka a spanning tree of the dual lattice!)
    The optional argument shortest_edge_only automatically sorts the edges to ensure that only the shortest connections are used 
    (which is kind of a fudgey way of stopping the algorithm from picking edges that connect over the periodic boundaries). If you're hungry for 
    speed you might want to turn it off. The algorith is basically prim's algorithm - so it should run in linear time.

    :param lattice: the lattice you want the tree on
    :type lattice: Lattice
    :param shortest_edges_only: do you want a minimum spanning tree - distance wise, defaults to True
    :type shortest_edges_only: bool, optional
    :return: a list of the edges that form the tree
    :rtype: np.ndarray
    """
    plaquettes_in = np.full(lattice.n_plaquettes, -1)
    edges_in = np.full(lattice.n_plaquettes-1, -1)

    plaquettes_in[0] = 0
    
    boundary_edges = np.copy(lattice.plaquettes[0].edges)

    for n in range(lattice.n_plaquettes-1):
        
        # if we want to keep the edges short - sort the available boundaries
        if shortest_edges_only:

            def find_plaq_distance(edge):
                p1,p2 = lattice.edges.adjacent_plaquettes[edge]
                c1 = 10 if p1 == INVALID else lattice.plaquettes[p1].center
                c2 = 10 if p2 == INVALID else lattice.plaquettes[p2].center
                distance = np.sum((c1 - c2)**2)
                return distance
                
            distances = np.vectorize(find_plaq_distance)( boundary_edges )
            order = np.argsort(distances)
        else:
            order = np.arange(len(boundary_edges))


        for edge_index in boundary_edges[order]:
            
            edge_plaq = lattice.edges.adjacent_plaquettes[edge_index]

            if INVALID in edge_plaq:
                continue
            
            lattice.plaquettes[edge_plaq[0]].center

            outisde_plaquette_present = [x not in plaquettes_in for x in  edge_plaq]
            inside_plaquette_present = [x in plaquettes_in for x in  edge_plaq]
            

            # if this edge links an inside and outside plaquette
            if np.any(outisde_plaquette_present) and np.any(inside_plaquette_present):

                # add the new plaquette to the list of inside ones
                position = np.where(outisde_plaquette_present)[0][0]
                new_plaquette = edge_plaq[position]
                plaquettes_in[n+1] = new_plaquette
                edges_in[n] = edge_index

                # add the new edges to the boundary edges
                boundary_edges = np.append(boundary_edges, lattice.plaquettes[new_plaquette].edges)

                # remove any doubled edges - these will be internal
                a, c = np.unique(boundary_edges, return_counts=True)
                boundary_edges = a[c == 1]
            

                break
            
    return edges_in 

# FIXME: change function signature to take lattice object instead of adjacency list
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

    #the next two lines replaces the simpler vertex_indices = edges[edges != vertex_i] because the allow points to neighbour themselves
    start_or_end = (edges != vertex_i)[:, 1] #this is true if vertex_i starts the edge and false if it ends it
    vertex_indices = np.take_along_axis(edges, start_or_end[:, None].astype(int), axis = 1).squeeze() #this gets the index of the other end of each edge
    #vertex_indices = edges[edges != vertex_i]
    assert(vertex_indices.shape == edge_indices.shape)
    return vertex_indices, edge_indices

def edge_neighbours(lattice, edge_i):
    """
    Return the neighbouring edges of an edge (the edges connected to the same nodes as this edge)

    :param lattice: The lattice
    :type lattice: Lattice
    :param edge_i: the index of the edge we want the neighbours of
    :type edge_i: integer
    :return: edge_indices: (k), the indices into adjacency of the edges that link vertex_i to its neighbours 
    :rtype: np.ndarray (k,)
    """
    edge = lattice.edges.indices[edge_i]
    v1 = edge[0]
    v2 = edge[1]
    mask = np.any(v1 == lattice.edges.indices, axis = -1) | np.any(v2 == lattice.edges.indices, axis=-1)
    mask[edge_i] = False #not a neighbour of itself
    return np.where(mask)[0]

def clockwise_about(vertex_i : int, g : Lattice) -> np.ndarray:
    """
    Finds the vertices/edges that border vertex_i, order them clockwise starting from the positive x axis
    and returns those indices in order.

    Args:
        vertex_i (int): int the index into g.vertices.positions of the node we want to use. Generally use 0
        g (Lattice): a graph object with keys vertices, adjacency, adjacency_crossing
    Returns:
        ordered_edge_indices: np.ndarray (n_neighbours_of_vertex_i) ordered indices of the edges. 
    """
    #get the edges and vertices around vertex 0
    vertex_indices, edge_indices = vertex_neighbours(vertex_i, g.edges.indices)
    edge_vectors = get_edge_vectors(vertex_i, edge_indices, g)

    #order them clockwise from the positive x axis
    angles = np.arctan2(edge_vectors[:, 1], edge_vectors[:,0])
    angles = np.where(angles > 0, angles, 2*np.pi + angles) #move from [-pi, pi] to [0, 2*pi]
    ordering = np.argsort(angles)
    ordered_edge_indices = edge_indices[ordering]
    ordered_vertex_indices = vertex_indices[ordering]
    return ordered_vertex_indices, ordered_edge_indices

def clockwise_edges_about(vertex_i : int, g : Lattice) -> np.ndarray:
    """
    Finds the edges that border vertex_i, orders them clockwise starting from the positive x axis
    and returns those indices in order. Use this to break the degeneracy of graph coloring.

    Args:
        vertex_i (int): int the index into g.vertices.positions of the node we want to use. Generally use 0
        g (Lattice): a graph object with keys vertices, adjacency, adjacency_crossing
    Returns:
        ordered_edge_indices: np.ndarray (n_neighbours_of_vertex_i) ordered indices of the edges. 
    """
    return clockwise_about(vertex_i, g)[1]

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
    #the next few lines do that so we can add g.edges.indices_crossing with the right sign
    edges = l.edges.indices[edge_indices]
    start_or_end = (edges != vertex_i)[:, 1] #this is true if vertex_i starts the edge and false if it ends it
    other_vertex_indices = np.take_along_axis(edges, start_or_end[:, None].astype(int), axis = 1).squeeze() #this gets the index of the other end of each edge
    offset_sign = (2*start_or_end - 1) #now it's +/-1
    
    #get the vectors along the edges
    return l.vertices.positions[other_vertex_indices] - l.vertices.positions[vertex_i][None, :] + offset_sign[:, None] * l.edges.crossing[edge_indices]

def adjacent_plaquettes(l : Lattice, p_index : int) -> Tuple[np.ndarray, np.ndarray]:
    """For a given lattice, compute the plaquettes that share an edge with lattice.plaquettes[p_index] and the shared edge.
    Returns a list of plaquettes indices and a matching list of edge indices.

    :param l: The lattice.
    :type l: Lattice
    :param p_index: The index of the plaquette to find the neighbours of.
    :type p_index: int
    :return: (plaque_indices, edge_indices)
    :rtype: Tuple[np.ndarray, np.ndarray]
    """    
    p = l.plaquettes[p_index]
    edges = p.edges
    neighbouring_plaquettes = l.edges.adjacent_plaquettes[edges]
    
    #remove edges that are only part of this plaquette
    valid = ~np.any(neighbouring_plaquettes == INVALID, axis = -1)
    edges, neighbouring_plaquettes = edges[valid], neighbouring_plaquettes[valid, :]
    
    # get just the other plaquette of each set
    p_index_location = neighbouring_plaquettes[:, 1] == p_index
    other_index = 1 - p_index_location.astype(int)[:, None] 
    neighbouring_plaquettes = np.take_along_axis(neighbouring_plaquettes, other_index, axis = 1).squeeze()
    
    return neighbouring_plaquettes, edges

def rotate(vector, angle):
    rm = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return rm @ vector

import itertools

def edge_crossing_that_minimises_length(start, end):
    """Given two points in the unit plane, return the edge crossing = [-1/0/+1, -1/0/+1,]
    that minimises the length of the edge between them.
    
    :param: start: The start point of the edge
    :type: np.ndarray shape (2,)
    :param: end: The start point of the edge
    :type: np.ndarray shape (2,)

    :return: The edge crossing that minimises the length of the edge.
    :rtype: np.ndarray shape (2,)
    """
    if np.linalg.norm(start - end, ord = 2) < 0.5: return np.array([0,0])
    crossings = np.array(list(itertools.product([-1,0,1], repeat = 2)))
    lengths = np.linalg.norm(start - end - crossings, ord = 2, axis = -1)
    return crossings[np.argmin(lengths)]

def make_dual(lattice, subset = slice(None, None)):
    """
    Given a lattice and a subset of its edge, contruct the dual lattice
    and return it as a new Lattice object.

    :param lattice: The lattice to make the dual of.
    :type lattice: Lattice
    :param subset: The edges to include in the dual.
    :type subset: slice, boolean array, or integer indices.

    :return: The dual lattice.
    :rtype: Lattice
    """
    subset = np.arange(lattice.n_edges, dtype = int)[subset]
    st_edges = np.array([lattice.edges.adjacent_plaquettes[i] for i in subset])
    def plaquette_index_to_center(i): return lattice.plaquettes[i].center
    st_verts = np.array([plaquette_index_to_center(i) for i in range(lattice.n_plaquettes)])
    st_crossing = np.array([edge_crossing_that_minimises_length(start, end) 
                for start, end in st_verts[st_edges]])
    return Lattice(st_verts, st_edges, st_crossing)

########### code that uses scipy.sparse.csgraph ################
from scipy import sparse
from scipy.sparse import csgraph

def sparse_adjacency(lattice: Lattice):
    """
    Create a sparse (dok_matrix) adjacency matrix from a Lattice object.
    Useful to use a Lattice object as input to a scipy.sparse.csgraph routine.
    """
    adj = sparse.dok_matrix((lattice.n_vertices, lattice.n_vertices))
    adj[lattice.edges.indices[:,1], lattice.edges.indices[:,0]] = 1
    adj[lattice.edges.indices[:,0], lattice.edges.indices[:,1]] = 1
    return adj

def edge_to_index_mapper(lattice: Lattice):
    """
    This returns a map where k = map[i,j] tells you the index k of an edge (i,j)
    The order of (i,j) or (j,i) doesn't matter.
    Useful to convert back from the output of a scipy.sparse.csgraph routine.
    """
    adj = np.zeros((lattice.n_vertices, lattice.n_vertices), dtype = int)
    indices = lattice.edges.indices
    adj[indices[:,1], indices[:,0]] = np.arange(lattice.n_edges)
    adj[indices[:,0], indices[:,1]] = np.arange(lattice.n_edges)
    return adj

def adjacency_to_edgelist(adj):
    "Given a sparse adjacency matrix, return the edges list of tuples."
    return np.array(list(adj.todok().keys()))

def minimum_spanning_tree(lattice : Lattice):
    """"
    Use scipy.sparse.csgraph.minimum_spanning_tree to find a 
    minimum spanning tree of the given lattice.
    """
    adjacency = sparse_adjacency(lattice)
    msp_adjacency = csgraph.minimum_spanning_tree(adjacency)
    msp_edge_tuples = adjacency_to_edgelist(msp_adjacency)
    edge_to_index = edge_to_index_mapper(lattice)
    edge_indices = edge_to_index[msp_edge_tuples[:, 0], msp_edge_tuples[:, 1]]
    return edge_indices