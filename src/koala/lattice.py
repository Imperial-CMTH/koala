import numpy as np
from numpy.lib.index_tricks import nd_grid
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class Plaquette:
    """Represents a single plaquette in a lattice

    :param vertices: Indices correspondng to the vertices that border the plaquette. These are always organised to start from the lowest index and then go clockwise around the plaquette
    :type vertices: np.ndarray[int] (nedges)
    :param edges: Indices correspondng to the edges that border the plaquette. These are arranged to start from the lowest indexed vertex and progress clockwise.
    :type edges: np.ndarray[int] (nedges)
    :param directions: Valued +/- 1 depending on whether the i'th edge points clockwise/anticlockwise around the plaquette
    :type directions: np.ndarray[int] (nedges)
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

    :param indices: An array of the indices of points connected by this edge. Entries should always starts with the point with the lowest index.
    :type indices: np.ndarray[int] (nvertices, 2)

    :param vectors: Indices correspondng to the edges that border the plaquette. These are arranged to start from the lowest indexed vertex and progress clockwise.
    :type vectors: np.ndarray[float] (nvertices, 2)

    :param crossing: Tells you whether the edge crosses the boundary conditions, and if so, in ehich direction. One value for x-direction and one for y-direction
    :type crossing: np.ndarray[int] (nvertices, 2)
    """
    indices: np.ndarray
    vectors: np.ndarray
    crossing: np.ndarray


class Lattice(object):
    """Describes a lattice in 2D. Lattice is made up of vertices, edges and plaquettes.

    :param vertices: Spatial positions of vertices
    :type vertices: np.ndarray[float] (nvertices, 2)

    :param edges: A dataclass containing the edge indices, vectors corresppoinding to each edge and crossings, telling you if the edge crossed the periodic boundaries
    :type edges: Edges

    :param plaquettes: A list of Plaquette dataclasses, each containing the plaquette centers, the edges that border it, the vertices around it and the direction of each edge around the plaquette
    :type plaquettes: list[Plaquette]

    """
    def __init__(self, vertices, edge_indices, edge_crossing, plaquettes=None):
        self.vertices = vertices

        # fix the order of the edge indices to ensure that they always go from low index to high index
        edge_indices, edge_crossing = self._reorder_edge_indices(edge_indices, edge_crossing)

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
        :return: A tuple containing reordered edge indices and fixed edge crossings
        :rtype: tuple[npt.NDArray[np.integer], npt.NDArray[np.np.integer]]
        
        """
        sorted_indices = np.sort (edge_indices)
        sorted_mask = -1+2*(sorted_indices == edge_indices)[:,0]
        fixed_crossings = edge_crossing*sorted_mask[:,None]

        return sorted_indices, fixed_crossings