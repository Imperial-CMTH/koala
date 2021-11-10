import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from koala import plotting
from koala import SAT
def rotate(vector, angle):
    rm = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return rm @ vector


def create_peterson_graph():

    """
    Returns a peterson graph, one of the simplest (but non-planar) cubic graphs that is not three-colourable. 

    Returns:
        vertices: np.array shape (nvertices, ndim) - A list of the positions of all the vertices that make up the graph
        adjacency: np.array shape (nedges, 2) - A list of all edges in the graph, containing the indices of connected vertices
    """
    points_outer = np.tile(np.array([0, 0.45]), (5, 1))
    points_inner = np.tile(np.array([0, 0.27]), (5, 1))

    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    print(angles)
    for n, a in enumerate(angles):
        points_outer[n, :] = rotate(points_outer[n, :], a)
        points_inner[n, :] = rotate(points_inner[n, :], a)

    vertices = np.concatenate((points_outer, points_inner))
    vertices += 0.5

    outer_ring_adj = np.concatenate((np.arange(0, 5), np.roll(
        np.arange(0, 5), -1))).reshape((-1, 2), order='F')
    between_ring_adj = np.concatenate((np.arange(0, 5),
                                      np.arange(5, 10))).reshape((-1, 2), order='F')
    inner_ring_adj = np.concatenate((np.arange(5, 10), np.roll(
        np.arange(5, 10), -2))).reshape((-1, 2), order='F')

    adjacency = np.concatenate(
        (outer_ring_adj, between_ring_adj, inner_ring_adj))

    return vertices, adjacency


def create_tutte_graph():
    """
    Returns a tutte graph, a cubic graph with no Hamiltonian cycle, but is three-colourable. 

    Returns:
        vertices: np.array shape (nvertices, ndim) - A list of the positions of all the vertices that make up the graph
        adjacency: np.array shape (nedges, 2) - A list of all edges in the graph, containing the indices of connected vertices
    """
    vertices = np.array([
        [0.518, 0.586], 
        [0.294, 0.986], 
        [0.504, 0.99], 
        [0.69, 0.99], 
        [0.998, 0.616], 
        [0.872, 0.374], 
        [0.746, 0.152], 
        [0.024, 0.558], 
        [0.17, 0.382], 
        [0.334, 0.15], 
        [0.454, 0.54], 
        [0.518, 0.67], 
        [0.592, 0.53], 
        [0.35, 0.548], 
        [0.436, 0.484], 
        [0.342, 0.502], 
        [0.296, 0.478], 
        [0.336, 0.418], 
        [0.408, 0.404], 
        [0.332, 0.93], 
        [0.214, 0.502], 
        [0.138, 0.558], 
        [0.226, 0.43], 
        [0.282, 0.38], 
        [0.368, 0.272], 
        [0.394, 0.822], 
        [0.464, 0.732], 
        [0.638, 0.894], 
        [0.55, 0.734], 
        [0.696, 0.274], 
        [0.62, 0.482], 
        [0.658, 0.55], 
        [0.768, 0.568], 
        [0.906, 0.6], 
        [0.508, 0.774], 
        [0.674, 0.5], 
        [0.508, 0.83], 
        [0.728, 0.482], 
        [0.424, 0.864], 
        [0.556, 0.894], 
        [0.414, 0.922], 
        [0.506, 0.934], 
        [0.784, 0.506], 
        [0.842, 0.482], 
        [0.76, 0.376], 
        [0.824, 0.412]
    ])
    avg_pos = np.sum(vertices, axis = 0)/vertices.shape[0]
    vertices -= avg_pos - np.array([0.5,0.5])

    adjacency = np.array([
        [1 , 11], 
        [1 , 12], 
        [1 , 13], 
        [2 , 3], 
        [2 , 8], 
        [2 , 20], 
        [3 , 4], 
        [3 , 42], 
        [4 , 5], 
        [4 , 28], 
        [ 5 , 6], 
        [ 5 , 34], 
        [ 6 , 7], 
        [ 6 , 46], 
        [ 7 , 10], 
        [ 7 , 30], 
        [ 8 , 9], 
        [ 8 , 22], 
        [ 9 , 10], 
        [ 9 , 23], 
        [ 10 , 25], 
        [ 11 , 14], 
        [ 11 , 15], 
        [ 12 , 27], 
        [ 12 , 29], 
        [ 13 , 31], 
        [ 13 , 32], 
        [ 14 , 16], 
        [ 14 , 22], 
        [ 15 , 16], 
        [ 15 , 19], 
        [ 16 , 17], 
        [ 17 , 18], 
        [ 17 , 21], 
        [ 18 , 19], 
        [ 18 , 24], 
        [ 19 , 25], 
        [ 20 , 26], 
        [ 20 , 41], 
        [ 21 , 22], 
        [ 21 , 23], 
        [ 23 , 24], 
        [ 24 , 25], 
        [ 26 , 27], 
        [ 26 , 39], 
        [ 27 , 35], 
        [ 28 , 29], 
        [ 28 , 40], 
        [ 29 , 35], 
        [ 30 , 31], 
        [ 30 , 45], 
        [ 31 , 36], 
        [ 32 , 33], 
        [ 32 , 36], 
        [ 33 , 34], 
        [ 33 , 43], 
        [ 34 , 44], 
        [ 35 , 37], 
        [ 36 , 38], 
        [ 37 , 39], 
        [ 37 , 40], 
        [ 38 , 43], 
        [ 38 , 45], 
        [ 39 , 41], 
        [ 40 , 42], 
        [ 41 , 42], 
        [ 43 , 44], 
        [ 44 , 46], 
        [ 45 , 46]
    ])

    adjacency -= 1

    return vertices, adjacency


vertices, adjacency = create_tutte_graph()
#plotting.plot_lattice(vertices, adjacency, adjacency_crossing= np.zeros(adjacency.shape))
#plt.show()
out = SAT.edge_color(adjacency, 3)
color_dict = {0: 'r', 1: 'y', 2: 'b'}
edgecolors = np.array([color_dict[c] for c in out])
plotting.plot_lattice(vertices, adjacency, adjacency_crossing=np.zeros(adjacency.shape), edge_colors=edgecolors)
plt.show()