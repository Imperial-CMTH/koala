import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt


def plot_lattice(vertices, adjacency):

    edge_vertices = vertices[adjacency]
    displacements = edge_vertices[:,0,:] - edge_vertices[:,1,:]

    mask = np.round(displacements)
    inside_bool =  np.logical_not(mask.any(axis=1))

    inside_edges = adjacency[np.where(inside_bool)]
    outside_edges = adjacency[np.where(np.logical_not(inside_bool))]

    inside_edge_vertices = vertices[inside_edges]
    outside_edge_vertices = vertices[outside_edges]

    outside_displacements = outside_edge_vertices[:,0,:] - outside_edge_vertices[:,1,:]
    outside_mask = np.round(outside_displacements)

    first_changed = outside_edge_vertices.copy()
    second_changed = outside_edge_vertices.copy()

    first_changed[:,0,:] -= outside_mask
    second_changed[:,1,:] += outside_mask

    fig = plt.figure()
    ax = plt.gca()
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))
    lc_inside = LineCollection(inside_edge_vertices,colors= 'k')
    lc_first = LineCollection(first_changed,colors= 'k')
    lc_second = LineCollection(second_changed,colors= 'k')
    ax.add_collection(lc_inside)
    ax.add_collection(lc_first)
    ax.add_collection(lc_second)
    # ax.scatter(
    #     vertices[:,0],
    #     vertices[:,1]
    # )

    #TODO - look at what this should return
