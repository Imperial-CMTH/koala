import numpy as np
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
    def __init__(self, vertices, edge_indices, edge_crossing, plaquettes = None):
        self.vertices = vertices
        edge_vectors = self._compute_edge_vectors(vertices, edge_indices, edge_crossing)
        
        self.edges = Edges(indices = edge_indices,
                           crossing = edge_crossing,
                           vectors = edge_vectors)
        
        self.plaquettes = plaquettes or self._compute_plaquettes()
        
    def __repr__(self): return f"Lattice({self.vertices.shape[0]} vertices, {self.edges.indices.shape[0]} edges)"
    
    def _compute_plaquettes(self):
        return "plaquettes"
        
    def _compute_edge_vectors(self, vertices, edge_indices, edge_crossing):
        return "edge_vectors"