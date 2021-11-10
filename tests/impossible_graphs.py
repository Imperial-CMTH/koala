import numpy as np
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection


def rotate(vector, angle):
    rm = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    return rm @ vector


def create_peterson_graph():
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
