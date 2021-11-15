import numpy as np

from koala.pointsets import generate_bluenoise
from koala.voronization import generate_pbc_voronoi_adjacency
from koala.graph_color import edge_color
from koala.plotting import plot_lattice

def test_plotting():
    n = 20
    points = generate_bluenoise(30,n,n)
    g = generate_pbc_voronoi_adjacency(points)
    solvable, solution = edge_color( g.adjacency, n_colors = 3)
    point_coloring = np.random.randint(2,size=g.vertices.shape[0])
    plot_lattice(g,edge_labels=solution)
    plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'])
    plot_lattice(g, vertex_labels=point_coloring)
    plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'],vertex_labels=point_coloring,vertex_color_scheme=['k','g'])
    plot_lattice(g,edge_labels=solution,edge_color_scheme=['k','lightgrey','blue'],scatter_args= {'c': 'r'} )
