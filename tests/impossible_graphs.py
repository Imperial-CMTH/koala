import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def plot_lattice(vertices, adjacency):

    edge_vertices = vertices[adjacency]
    displacements = edge_vertices[:, 0, :] - edge_vertices[:, 1, :]

    mask = np.round(displacements)
    inside_bool = np.logical_not(mask.any(axis=1))

    inside_edges = adjacency[np.where(inside_bool)]
    outside_edges = adjacency[np.where(np.logical_not(inside_bool))]

    inside_edge_vertices = vertices[inside_edges]
    outside_edge_vertices = vertices[outside_edges]

    outside_displacements = outside_edge_vertices[:,
                                                  0, :] - outside_edge_vertices[:, 1, :]
    outside_mask = np.round(outside_displacements)

    first_changed = outside_edge_vertices.copy()
    second_changed = outside_edge_vertices.copy()

    first_changed[:, 0, :] -= outside_mask
    second_changed[:, 1, :] += outside_mask

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    lc_inside = LineCollection(inside_edge_vertices, colors='k')
    lc_first = LineCollection(first_changed, colors='k')
    lc_second = LineCollection(second_changed, colors='k')
    ax.add_collection(lc_inside)
    ax.add_collection(lc_first)
    ax.add_collection(lc_second)
    # ax.scatter(
    #     vertices[:,0],
    #     vertices[:,1]
    # )

    # TODO - look at what this should return


def rotate(vector, angle):
    rm = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return rm @ vector


def create_peterson_graph():
    points_outer = np.tile(np.array([0, 0.2]), (5, 1))
    points_inner = np.tile(np.array([0, 0.1]), (5, 1))

    angles = np.linspace(0, 2*np.pi, 5, endpoint=False)
    print(angles)
    for n, a in enumerate(angles):
        points_outer[n, :] = rotate(points_outer[n, :], a)
        points_inner[n, :] = rotate(points_inner[n, :], a)

    points = np.concatenate((points_outer, points_inner))
    points += 0.5

    outer_ring_adj = np.concatenate((np.arange(0, 5), np.roll(
        np.arange(0, 5), -1))).reshape((-1, 2), order='F')
    between_ring_adj = np.concatenate((np.arange(0, 5),
                                      np.arange(5, 10))).reshape((-1, 2), order='F')
    inner_ring_adj = np.concatenate((np.arange(5, 10), np.roll(
        np.arange(5, 10), -2))).reshape((-1, 2), order='F')

    adjacency = np.concatenate(
        (outer_ring_adj, between_ring_adj, inner_ring_adj))

    return points, adjacency


points, adj = create_peterson_graph()

plot_lattice(points, adj)
plt.show()
