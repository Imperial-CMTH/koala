import numpy as np
from numpy.lib.index_tricks import nd_grid
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class Plaquette:
    """Represents a single plaquette in a lattice. Not a list since plaquettes can have varying size.

    :param vertices: Indices correspondng to the vertices that border the plaquette. These are always organised to start from the lowest index and then go clockwise around the plaquette
    :type vertices: np.ndarray[int] (plaquettesize)
    :param edges: Indices correspondng to the edges that border the plaquette. These are arranged to start from the lowest indexed vertex and progress clockwise.
    :type edges: np.ndarray[int] (plaquettesize)
    :param directions: Valued +/- 1 depending on whether the i'th edge points clockwise/anticlockwise around the plaquette
    :type directions: np.ndarray[int] (plaquettesize)
    :param centers: Coordinates of the center of the plaquette
    :type centers: np.ndarray[float] (2)
    """
    vertices: np.ndarray
    edges: np.ndarray
    directions: np.ndarray
    centers: np.ndarray


@dataclass
class Edges:
    """
    Represents the list of edges in the lattice

    :param indices: Indices of points connected by each edge. 
    :type indices: np.ndarray[int] (nedges, 2)
    :param vectors: Vectors pointing along each edge
    :type vectors: np.ndarray[float] (nedges, 2)
    :param crossing: Tells you whether the edge crosses the boundary conditions, and if so, in ehich direction. One value for x-direction and one for y-direction
    :type crossing: np.ndarray[int] (nedges, 2)
    :param adjacent_plaquettes: Lists the indices of every plaquette that touches each edge
    :type adjacent_plaquettes: np.ndarray[int] (nedges, 2)
    """

    indices: np.ndarray
    vectors: np.ndarray
    crossing: np.ndarray
    adjacent_plaquettes: np.ndarray


@dataclass
class Vertices:
    """
    Represents a list of vertices in the lattice

    :param positions: List of the positions of every vertex
    :type positions: np.ndarray[float] (nvertices, 2)
    :param adjacent_edges: Lists the indices of every edge that connects to that vertex. Listed in clockwise order from the lowest index
    :type adjacent_edges: list[np.ndarray] (nvertices, n_edges_per_vertex)
    :param adjacent_plaquettes: Lists the indices of every plaquette that touches the vertex
    :type adjacent_plaquettes: np.ndarray[int] (nvertices, 3)
    """

    positions: np.ndarray
    adjacent_edges: np.ndarray
    adjacent_plaquettes: np.ndarray


class Lattice(object):
    """Data structure containing information about a lattice consisting of vertices in real space connected by undirected edges.

    :param vertices: Data structure containing vertex positions, and the edges/plaquettes touching each vertex
    :type vertices: Vertices
    :param edges: Data structure containing indices of vertices comprising each edge, the spatial displacement vectors
        corresponding to those edges, flags for edges which cross the system boundary in periodic Lattices, and the plaquettes
        touching each edge.
    :type edges: Edges
    :param plaquettes: All of the polygons (aka plaquettes) comprising the lattice, specifying their constituent vertices, edges,
        winding directions, and centers.
    :type list[Plaquette]
    """    
    def __init__(
            self,
            vertices: npt.NDArray[np.floating],
            edge_indices: npt.NDArray[np.integer],
            edge_crossing: npt.NDArray[np.integer]):
        """Constructor for Lattices

        :param vertices: Spatial locations of lattice vertices
        :type vertices: npt.NDArray[np.floating] Shape (nverts, 2)
        :param edge_indices: Indices corresponding to the vertices which each edge connects
        :type edge_indices: npt.NDArray[np.integer] Shape (nedges, 2)
        :param edge_crossing: Flags describing which boundaries of the system each edge crosses in periodic boundary conditions.
        Each entry in the final axis corresponds to a spatial dimension, 1(-1) denotes an edge crossing a boundary in the positive
        (negative) direction along that dimension. 0 corresponds to no boundary crossing.
        :type edge_crossing: npt.NDArray[np.integer] Shape (nedges, 2)
        """            

        # calculate the vector corresponding to each edge
        edge_vectors = (vertices[edge_indices][:, 1] -
                        vertices[edge_indices][:, 0] + edge_crossing)

        # calculate the list of edges adjacent to each vertex
        vertex_adjacent_edges = _sorted_vertex_adjacent_edges(
            vertices, edge_indices, edge_vectors)

        self.vertices = Vertices(
            positions=vertices,
            adjacent_edges=vertex_adjacent_edges,
            adjacent_plaquettes=None  # TODO: construct adjacent plaquette finder
        )

        self.edges = Edges(
            indices=edge_indices,
            vectors=edge_vectors,
            crossing=edge_crossing,
            adjacent_plaquettes=None  # TODO: construct adjacent plaquette finder
        )

        self.plaquettes = _find_all_plaquettes(self)
        # TODO: find plaquette centers


def _sorted_vertex_adjacent_edges(
        vertex_positions,
        edge_indices,
        edge_vectors):
    """Gives you an array where the i'th row contains the indices of the edges that connect to the i'th vertex. 
    The edges are always organised in clockwise order starting from 12:00, which will be handy later ;)

    :param vertex_positions: List of the positions of every vertex
    :type vertex_positions: np.ndarray[float] (nvertices, 2)
    :param edge_indices: Indices of points connected by each edge. 
    :type edge_indices: np.ndarray[int] (nedges, 2)
    :param edge_vectors: Vectors pointing along each edge
    :type edge_vectors: np.ndarray[float] (nedges, 2)
    :return: List containing the indices of the edges that connect to each point, ordered clockwise around the point.
    :rtype: list[int] (nvertices, nedges_per_vertex)
    """

   # sorts these lists to make sure that they always are ordered clockwise from 12:00 like on a clock face
    vertex_adjacent_edges = []
    for index in range(vertex_positions.shape[0]):
        v_edges = np.nonzero(
            (edge_indices[:, 0] == index) + (edge_indices[:, 1] == index))[0]

        # is the chosen index is first or second in the list
        v_parities = []
        for i, edge in enumerate(v_edges):
            par = 1 if (edge_indices[edge][0] == index) else -1
            v_parities.append(par)
        v_parities = np.array(v_parities)

        # find the angle that each vertex comes out at
        v_vectors = edge_vectors[v_edges]*v_parities[:, None]
        v_angles = np.arctan2(-v_vectors[:, 0], v_vectors[:, 1]) % (2*np.pi)

        # reorder the indices
        # [sorted(-v_angles).index(x) for x in -v_angles]
        order = np.argsort(-v_angles)
        edges_out = v_edges[order]

        vertex_adjacent_edges.append(edges_out)

    vertex_adjacent_edges = vertex_adjacent_edges

    return vertex_adjacent_edges


def _find_plaquette(
        starting_edge: int,
        starting_direction: int,
        l: Lattice):
    """Given a single edge, and a direction, this code finds theplaquette corresponding to starting in that 
    direction and only taking left turns. This means plaquettes are ordered anticlockwise - which amounts to going round each vertex clockwise.

    :param starting_edge: Index of the edge where you start
    :type starting_edge: int
    :param starting_direction: Direction to take the first step. 0 means the same direction as the edge, 1 means opposite
    :type starting_direction: int (0 or 1)
    :param l: Lattice to be searched for the plaquette
    :type l: Lattice
    :return: A plaquette object representing the found plaquette
    :rtype: Plaquette
    """
    # TODO - fix it so we also find the plaquette centers - left this out because PBC is a pain
    # TODO - Decide when you have an open boundaries exterior plaquette!
    edge_indices = l.edges.indices
    vertex_adjacent_edges = l.vertices.adjacent_edges
    start_vertex = edge_indices[starting_edge, starting_direction]
    current_edge = starting_edge
    current_vertex = start_vertex
    current_direction = starting_direction
    # print(current_vertex, current_edge ,edge_indices[current_edge], current_direction)

    plaquette_edges = [starting_edge]
    plaquette_vertices = [start_vertex]
    plaquette_directions = [starting_direction]

    while True:
        # one anticlockwise step around the plaquette
        current_vertex = edge_indices[current_edge][np.where(
            np.roll(edge_indices[current_edge], 1) == current_vertex)[0][0]]
        current_edge_choices = vertex_adjacent_edges[current_vertex]
        current_edge = current_edge_choices[(np.where(current_edge_choices == current_edge)[
                                             0][0] + 1) % current_edge_choices.shape[0]]
        current_direction = 0 if np.where(
            edge_indices[current_edge] == current_vertex)[0][0] == 0 else 1

        # stop when you get back to where you started
        if current_edge == starting_edge:
            break
        plaquette_edges.append(current_edge)
        plaquette_vertices.append(current_vertex)
        plaquette_directions.append(current_direction)

    plaquette_edges = np.array(plaquette_edges)
    plaquette_vertices = np.array(plaquette_vertices)
    plaquette_directions = np.array(plaquette_directions)

    found_plaquette = Plaquette(vertices=plaquette_vertices,
                                edges=plaquette_edges, directions=plaquette_directions, centers=None)

    return found_plaquette


def _find_all_plaquettes(l: Lattice):
    """Construct a list of Plaquette objects, representing all of the polygons in the lattice.

    :param l: Lattice whose plaquettes are to be found
    :type l: Lattice
    :return: List of all plaquettes in the lattice
    :rtype: list[Plaquette]
    """

    edge_indices = l.edges.indices
    vertex_adjacent_edges = l.vertices.adjacent_edges
    # have we already found a plaquette on that edge going in that direction
    edges_fwd_backwd_remaining = np.ones_like(edge_indices)

    plaquettes = []
    for i in range(edge_indices.shape[0]):

        # every edge touches at most two new plaquettes one going forward and one going backwards
        if edges_fwd_backwd_remaining[i, 0] == 1:
            plaq_obj = _find_plaquette(
                i, 0, l)
            edges_fwd_backwd_remaining[plaq_obj.edges, plaq_obj.directions] = 0
            plaquettes.append(plaq_obj)

        if edges_fwd_backwd_remaining[i, 1] == 1:
            plaq_obj = _find_plaquette(
                i, 1, l)
            edges_fwd_backwd_remaining[plaq_obj.edges, plaq_obj.directions] = 0
            plaquettes.append(plaq_obj)

    return plaquettes

def permute_vertices(l: Lattice, ordering: npt.NDArray[np.integer]) -> Lattice:
  """Create a new lattice with the vertex indices rearranged according to ordering,
  such that new_l.vertices[i] = l.vertices[ordering[i]].

  :param l: Original lattice to have vertices reordered
  :type l: Lattice
  :param ordering: Permutation of vertex ordering, i = ordering[i']
  :type ordering: npt.NDArray[np.integer]
  :return: New lattice object with permuted vertex indices
  :rtype: Lattice
  """
  original_verts = l.vertices
  original_edges = l.edges
  nverts = original_verts.shape[0]

  inverse_ordering = np.zeros((nverts,)).astype(int)
  inverse_ordering[ordering] = np.arange(nverts).astype(int) # inverse_ordering[i] = i'

  new_edges = Edges(
    indices = inverse_ordering[original_edges.indices],
    vectors = original_edges.vectors,
    crossing = original_edges.crossing
  )
  new_verts = original_verts[ordering]
  return Lattice(
    vertices=new_verts,
    edge_indices=new_edges.indices,
    edge_crossing=new_edges.crossing
  )
