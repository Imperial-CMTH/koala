import numpy as np
import numpy.typing as npt
from dataclasses import dataclass
from .graph_utils import get_edge_vectors


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
    ) -> npt.NDArray:
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
