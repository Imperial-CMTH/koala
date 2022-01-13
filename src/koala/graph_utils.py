############################################################################
#              Generic routines for doing things on graphs                 #
#                                                                          #
############################################################################
import numpy as np
from .lattice import Lattice, INVALID
from typing import Tuple

def plaquette_spaning_tree(lattice: Lattice, shortest_edges_only = True):
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