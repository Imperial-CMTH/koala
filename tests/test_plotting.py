import numpy as np

from koala.pointsets import generate_bluenoise
from koala.voronization import generate_lattice
from koala.graph_color import edge_color
from koala.plotting import plot_lattice, plot_vertex_indices, plot_degeneracy_breaking, plot_plaquettes
from koala.voronization import Lattice

h = Lattice(
    vertices = np.array([[0.5,0.5], [0.1,0.1], [0.5,0.9], [0.9,0.1]]),
    edge_indices = np.array([[0,1],[0,2],[0,3]]),
    edge_crossing = np.array([[0,0],[0,0],[0,0]]),
    )

n = 10
points = generate_bluenoise(30,n,n)
voronoi_lattice = generate_lattice(points)

test_lattices = [voronoi_lattice,h]

def test_plotting():
    for g in test_lattices:
        solvable, solution = edge_color(g.edges.indices, n_colors = 3)
        point_coloring = np.random.randint(2,size=g.vertices.positions.shape[0])
        plot_lattice(g,edge_labels=solution)
        plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'])
        plot_lattice(g, vertex_labels=point_coloring)
        plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'],vertex_labels=point_coloring,vertex_color_scheme=['k','g'])
        plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'],scatter_args= {'c': 'r'} )
        plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'],scatter_args= {'c': 'r'}, edge_arrows=True)



def test_plot_vertex_indices():
    for g in test_lattices:
        plot_lattice(h, edge_arrows = True, edge_index_labels = True, vertex_labels = 0)
        plot_vertex_indices(h)
        plot_degeneracy_breaking(0, h)

def test_plaquette_plotting():
    from matplotlib import pyplot as plt
    cmap = plt.get_cmap("tab10")

    for g in test_lattices:        
        plaq_labels = np.arange(g.n_plaquettes)
        color_scheme = cmap(plaq_labels % 10)
        plot_plaquettes(g, plaq_labels, color_scheme = color_scheme, polygon_args = dict(alpha = 0.3))