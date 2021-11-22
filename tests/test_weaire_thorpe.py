import numpy as np
from matplotlib import pyplot as plt

from koala.weaire_thorpe import vertices_to_triangles
from koala.voronization import Lattice, generate_pbc_voronoi_adjacency
from koala.graph_utils import clockwise_edges_about
from koala.pointsets import generate_bluenoise
from koala.graph_color import edge_color
from koala.plotting import plot_lattice

def test_smoketest_weaire_thorpe():
    """This test doesn't do any assertions, it just checks that the normal usage doesn't crash"""
    fig, ax = plt.subplots()

    points =  generate_bluenoise(k = 100, nx = 5, ny = 5)
    g = generate_pbc_voronoi_adjacency(points)

    ordered_edge_indices = clockwise_edges_about(vertex_i = 0, g=g)
    solveable, edge_labels = edge_color(g.adjacency, n_colors = 3, fixed = enumerate(ordered_edge_indices))

    solveable, edge_labels = edge_color(g.adjacency, n_colors = 3, fixed = enumerate(ordered_edge_indices))
    WT_g = vertices_to_triangles(g, edge_labels);

    #ax.scatter(*g.vertices.T, color = 'k', alpha = 0.9)
    #ax.scatter(*WT_g.vertices.T, color = 'g')

    edge_labels = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)
    edge_arrows = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)

    plot_lattice(WT_g, edge_arrows = edge_arrows, ax = ax, edge_labels = edge_labels)


def test_multi_graphs():
    """The following graph has vertices with edges that link the same vertex together,
     this checks that clockwise edges about handles them correctly as part of the Weaire-Thorpe process"""

    g = Lattice(
        vertices = np.array([[0.5,0.7], [0.5,0.3]]),
        adjacency = np.array([[0,1],[0,0],[0,1],[1,1]]),
        adjacency_crossing = np.array([[0,0],[1,0],[1,0],[1,0]]),
        vor = None,
    )

    ordered_edge_indices = clockwise_edges_about(vertex_i = 0, g=g)
    solveable, edge_labels = edge_color(g.adjacency, n_colors = 3, fixed = enumerate(ordered_edge_indices))

    WT_g = vertices_to_triangles(g, edge_labels)
    
    edge_labels = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)
    internal_edges = np.where(np.arange(WT_g.adjacency.shape[0]) < g.adjacency.shape[0], 0, 1)

    fig, axes = plt.subplots(ncols = 2)
    plot_lattice(g, edge_arrows = True, ax = axes[0])#, edge_labels = edge_labels)
    plot_lattice(WT_g, edge_arrows = internal_edges, ax = axes[1], edge_labels = edge_labels)