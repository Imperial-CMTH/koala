import numpy as np
from numpy.lib.index_tricks import nd_grid
import numpy.typing as npt
from dataclasses import dataclass


@dataclass
class Plaquette:
    edges: np.ndarray
    vertices: np.ndarray
    directions: np.ndarray
    centers: np.ndarray


@dataclass
class Edges:
    indices: np.ndarray
    vectors: np.ndarray
    crossing: np.ndarray


class Lattice(object):
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