import numpy as np
from numpy.lib.index_tricks import nd_grid
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class Plaquette:
    """Represents a single plaquette in a lattice

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
    :type adjacent_edges: np.ndarray[int] (nvertices, 3)
    :param adjacent_plaquettes: Lists the indices of every plaquette that touches the vertex
    :type adjacent_plaquettes: np.ndarray[int] (nvertices, 3)
    """
    
    positions: np.ndarray
    adjacent_edges: np.ndarray
    adjacent_plaquettes: np.ndarray



class Lattice(object):
    def __init__(
        self,
        vertices: npt.NDArray[np.floating], 
        edge_indices: npt.NDArray[np.integer],
        edge_crossing: npt.NDArray[np.integer],
        plaquettes=None):

        # calculate the vector corresponding to each edge
        edge_vectors = (vertices[edge_indices][:, 1] - vertices[edge_indices][:, 0] + edge_crossing)

        # calculate the list of edges adjacent to each vertex
        vertex_adjacent_edges = _sorted_vertex_adjacent_edges(vertices,edge_indices,edge_vectors)
        
        self.vertices = Vertices(
            positions= vertices, 
            adjacent_edges= vertex_adjacent_edges,
            adjacent_plaquettes= None # <---------------------------------------- missing
        )

        self.edges = Edges(
            indices= edge_indices,
            vectors = edge_vectors,
            crossing= edge_crossing,
            adjacent_plaquettes = None # <---------------------------------------- missing
        )

        # self.plaquettes  <---------------------------------------- missing


def _sorted_vertex_adjacent_edges(
    vertex_positions,
    edge_indices,
    edge_vectors):
    """Gives you an array where the i'th row contains the indicesof the edges that connect to the i'th vertex. 
    The edges are always organised in clockwise order, which will be handy later ;)

    :param vertex_positions: List of the positions of every vertex
    :type vertex_positions: np.ndarray[float] (nvertices, 2)
    :param edge_indices: Indices of points connected by each edge. 
    :type edge_indices: np.ndarray[int] (nedges, 2)
    :param edge_vectors: Vectors pointing along each edge
    :type edge_vectors: np.ndarray[float] (nedges, 2)
    :return: Array containing the indices of the three edges that connect to each point, ordered clockwise around the point.
    :rtype: list[int] (nvertices, 3)
    """

    vertex_adjacent_edges = []
    for index in range(vertex_positions.shape[0]):
        v_edges = list(np.nonzero((edge_indices[:,0] == index )+ (edge_indices[:,1] ==  index))[0])


        vertex_adjacent_edges = []
        for index in range(vertex_positions.shape[0]):
            v_edges = np.nonzero((edge_indices[:,0] == index )+ (edge_indices[:,1] ==  index))[0]

            # tells you whether the chosen index is first or second in the list
            v_parities = []
            for i, edge in enumerate(v_edges):
                par = 1 if (edge_indices[edge][0] == index) else -1
                v_parities.append(par)
            v_parities = np.array(v_parities)
            
            # find the angle that each vertex comes out at
            v_vectors = edge_vectors[v_edges]*v_parities[:,None]
            v_angles = np.arctan2(-v_vectors[:,0],v_vectors[:,1]) %(2*np.pi)

            # reorder the indices
            order =  np.argsort(-v_angles) #[sorted(-v_angles).index(x) for x in -v_angles]
            edges_out = v_edges[order]

            vertex_adjacent_edges.append(edges_out)

    vertex_adjacent_edges = vertex_adjacent_edges

    return vertex_adjacent_edges






'''
class Lattice(object):
    """Describes a lattice in 2D. Lattice is made up of vertices, edges and plaquettes.

    :param vertices: Spatial positions of vertices
    :type vertices: np.ndarray[float] (nvertices, 2)

    :param edges: A dataclass containing the edge indices, vectors corresppoinding to each edge and crossings, telling you if the edge crossed the periodic boundaries
    :type edges: Edges

    :param plaquettes: A list of Plaquette dataclasses, each containing the plaquette centers, the edges that border it, the vertices around it and the direction of each edge around the plaquette
    :type plaquettes: list[Plaquette]

    """
    def __init__(self, vertices, edge_indices, edge_crossing, plaquettes=None, directed=False):
        """Creates an instance of the lattice class. If plaquettes are not provided then they are calculated

        :param vertices: Spatial positions of vertices
        :type vertices: np.ndarray[float] (nvertices, 2)
        :param edge_indices: An array of the indices of points connected by this edge. Entries should always starts with the point with the lowest index.
        :type edge_indices: np.ndarray[int] (nedges, 2)
        :param edge_crossing: Tells you whether the edge crosses the boundary conditions, and if so, in ehich direction. One value for x-direction and one for y-direction
        :type edge_crossing: np.ndarray[int] (nedges, 2)
        :param plaquettes: A list of plaquette objects, describing each plaquette in the system, if None then it is calculated
        :type plaquettes: list[Plaquette], optional
        :param directed: If false, the graph edges are treated as undirected and kept in sorted order, otherwise the ordering is preserved.
        :type directed: bool, optional
        """
        self.vertices = vertices

        # fix the order of the edge indices to ensure that they always go from low index to high index
        if not directed: edge_indices, edge_crossing = self._reorder_edge_indices(edge_indices, edge_crossing)

        edge_vectors = self._compute_edge_vectors(vertices, edge_indices, edge_crossing)

        self.edges = Edges(
            indices=edge_indices, crossing=edge_crossing, vectors=edge_vectors
        )

        self.plaquettes = plaquettes or self._compute_plaquettes()

    def __repr__(self):
        return f"Lattice({self.vertices.shape[0]} vertices, {self.edges.indices.shape[0]} edges)"

    def _compute_plaquettes(self):
        return "plaquettes"

    def _compute_edge_vectors(
        self,
        vertices: npt.NDArray[np.floating],
        edge_indices: npt.NDArray[np.integer],
        edge_crossing: npt.NDArray[np.integer],
    ) -> npt.NDArray[np.floating]:
        """Computes displacement vectors corresponding to edges, respecting edge crossings.

        :param vertices: Spatial positions of vertices
        :type vertices: np.ndarray[float] (nvertices, 2)
        :param edge_indices: Indices corresponding to (start, end) vertices of edges
        :type edge_indices: np.ndarray[int] (nedges, 2)
        :param edge_crossing: Sign of system boundary crossings for each edge
        :type edge_crossing: np.ndarray[(-1,0,1)] (nedges, 2)
        :return: Spatial displacements vectors corresponding to edges
        :rtype: np.ndarray[float]
        """
        edge_vectors = (
            vertices[edge_indices][:, 0] - vertices[edge_indices][:, 1] + edge_crossing
        )
        return edge_vectors


    # TODO - is this the right way to define the outputs of the function????? AAAAAH TYPING :(  - peru
    def _reorder_edge_indices(
        self,
        edge_indices:npt.NDArray[np.integer],
        edge_crossing: npt.NDArray[np.integer]
    ):
        """Reorders the edge indices to make sure that they always label from low to high. Wherever we 
        swap the order of an edge we also have to flip the value of the adjacency crossing for that edge

        :param edge_indices: Indices corresponding to (start, end) vertices of edges
        :type edge_indices: np.ndarray[int] (nedges, 2)
        :param edge_crossing: Sign of system boundary crossings for each edge
        :type edge_crossing: np.ndarray[(-1,0,1)] (nedges, 2)
        :return: Reordered edge indices and fixed edge crossings
        :rtype: np.ndarray[int] (nedges, 2), np.ndarray[int] (nedges, 2)
        
        """
        sorted_indices = np.sort (edge_indices)
        sorted_mask = -1+2*(sorted_indices == edge_indices)[:,0]
        fixed_crossings = edge_crossing*sorted_mask[:,None]

        return sorted_indices, fixed_crossings
'''

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
