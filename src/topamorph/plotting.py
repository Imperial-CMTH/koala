import numpy as np
from matplotlib.collections import LineCollection
from matplotlib import pyplot as plt

def plot_lattice(vertices, adjacency, edge_colors = None):

    edge_vertices = vertices[adjacency]
    displacements = edge_vertices[:,0,:] - edge_vertices[:,1,:]

    mask = np.round(displacements)
    inside_bool =  np.logical_not(mask.any(axis=1))
    inside_idx = np.where(inside_bool)
    outside_idx = np.where(np.logical_not(inside_bool))

    inside_edges = adjacency[inside_idx]
    outside_edges = adjacency[outside_idx]

    if edge_colors is not None:
        inside_colors = edge_colors[inside_idx]
        outside_colors = edge_colors[outside_idx]
    else:
        inside_colors = 'k'
        outside_colors = 'k'

    inside_edge_vertices = vertices[inside_edges]
    outside_edge_vertices = vertices[outside_edges]

    outside_displacements = outside_edge_vertices[:,0,:] - outside_edge_vertices[:,1,:]
    outside_mask = np.round(outside_displacements)

    first_changed = outside_edge_vertices.copy()
    second_changed = outside_edge_vertices.copy()

    first_changed[:,0,:] -= outside_mask
    second_changed[:,1,:] += outside_mask

    fig, ax = plt.subplots()
    ax.set(xlim = (0,1), ylim = (0,1))
    lc_inside = LineCollection(inside_edge_vertices, colors = inside_colors)
    lc_first = LineCollection(first_changed, colors = outside_colors)
    lc_second = LineCollection(second_changed, colors = outside_colors)
    ax.add_collection(lc_inside)
    ax.add_collection(lc_first)
    ax.add_collection(lc_second)
    # ax.scatter(
    #     vertices[:,0],
    #     vertices[:,1]
    # )

    #TODO - look at what this should return