import numpy as np

from koala.pointsets import generate_bluenoise
from koala.voronization import generate_lattice
from koala.graph_color import edge_color
from koala.plotting import plot_lattice, plot_vertex_indices, plot_degeneracy_breaking
from koala.voronization import Lattice

def test_plotting():
    n = 20
    points = generate_bluenoise(30,n,n)
    g = generate_lattice(points)
    solvable, solution = edge_color(g.edges.indices, n_colors = 3)
    point_coloring = np.random.randint(2,size=g.vertices.positions.shape[0])
    plot_lattice(g,edge_labels=solution)
    plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'])
    plot_lattice(g, vertex_labels=point_coloring)
    plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'],vertex_labels=point_coloring,vertex_color_scheme=['k','g'])
    plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'],scatter_args= {'c': 'r'} )
    plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'],scatter_args= {'c': 'r'}, edge_arrows=True)

g = Lattice(
    vertices = np.array([[0.5,0.5], [0.1,0.1], [0.5,0.9], [0.9,0.1]]),
    edge_indices = np.array([[0,1],[0,2],[0,3]]),
    edge_crossing = np.array([[0,0],[0,0],[0,0]]),
    )

def test_plot_vertex_indices():
    plot_lattice(g, edge_arrows = True, edge_index_labels = True, vertex_labels = 0)
    plot_vertex_indices(g)
    plot_degeneracy_breaking(0, g)