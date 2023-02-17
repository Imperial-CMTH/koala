import numpy as np
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from time import time
from mpire import WorkerPool

# symmetric = True

# if symmetric: sampling_points, triangulation = get_triangular_sampling_points(samples = 20)
# else: sampling_points, triangulation = get_non_symmetric_triangular_sampling_points(samples = 20)

# lattice, coloring = eg.honeycomb_lattice(15, True)
# ujk = np.ones(lattice.n_edges)

# def function(Js): return some data

# extra_args = dict(lattice = lattice,
#                   coloring = coloring,
#                   ujk = ujk,
#                   e_threshold = 0.1,
#                   e_range = 0.5)

# number_in_range, gaps = compute_phase_diagram(sampling_points, function, extra_args, n_jobs = 10)

# if symmetric: plot_phase_diagram_symmetric(lattice, ujk, triangulation, number_in_range, gaps)
# else: plot_phase_diagram(lattice, ujk, triangulation, number_in_range, gaps)


def get_non_symmetric_triangular_sampling_points(samples=10):
    """Usage:
    sampling_points, triangulation = pd.get_non_symmetric_triangular_sampling_points(samples = 20)
    data = f(sampling_points)

    ax.triplot(triangulation, "ko-", markersize = 1) #to plots the points where samples were taken
    ax.tricontourf(triangulation, data) #to plot the data
    """
    # Create triangulation.
    # Start with the unit square
    Jx = np.linspace(0, 0.5, samples)
    Jy = np.linspace(0, 0.5, samples)

    points = np.array([(x, y) for y in Jy for x in Jx])

    # Get rid of the uneccessary half of the square giving a right triangle
    xs, ys = points.T
    w = np.where(xs + ys <= 1)
    points = points[w]
    xs, ys = points.T
    zs = 1 - xs - ys
    triple_points = np.array([xs, ys, zs]).T

    # Transform the right triangle into an equilateral one for plotting
    theta = np.pi / 3
    T = np.array([[1, np.cos(theta)], [0, np.sin(theta)]])

    tpoints = np.einsum("ij,kj -> ki", T, points)
    triangulation = mtri.Triangulation(*tpoints.T)

    return triple_points, triangulation


def get_triangular_sampling_points(samples=10):
    # Create triangulation.
    # Start with the unit square
    Jx = np.linspace(0, 0.5, samples)
    Jy = np.linspace(0, 0.5, samples)
    grid_spacing = 1 / samples

    points = np.array([(x, y) for y in Jy for x in Jx])

    xs, ys = points.T
    zs = 1 - xs - ys

    shape = (zs - ys >= -grid_spacing / 2) & (ys - xs >= -grid_spacing / 2)
    points = points[shape]
    points = np.concatenate([points, [
        [1 / 3, 1 / 3],
    ]])  #add the center point because it sometimes gets cut off

    xs, ys = points.T
    zs = 1 - xs - ys
    triple_points = np.array([xs, ys, zs]).T

    def rotation(t):
        return np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])

    # reflection = np.array([[0, 1], [1, 0]])
    centerp = np.array([0.5, np.tan(np.pi / 6) / 2])
    skew = np.array([[1, np.cos(np.pi / 3)], [0, np.sin(np.pi / 3)]])

    triangulations = []
    for reflect in [False, True]:
        for i in range(3):
            #reflect in the diagonal
            lpoints = points[:, ::-1].copy() if reflect else points.copy()

            # Transform the right triangle into an equilateral one for plotting
            lpoints = np.einsum("ij,kj -> ki", skew, lpoints)

            # Rotate about the center point
            lpoints -= centerp
            t = -2 * np.pi / 3 * i
            lpoints = np.einsum("ij,kj -> ki", rotation(t), lpoints)
            lpoints += centerp

            #save this triangulation
            triangulations.append(mtri.Triangulation(*lpoints.T))

    return triple_points, triangulations


def plot_tri(ax, data, triangulations):
    vmin, vmax = np.min(data), np.max(data)

    for t in triangulations:
        ax.tricontourf(t, data, vmin=vmin, vmax=vmax)
        # ax.triplot(t, "ko-", markersize = 1)


def plot_triangle(ax, inner=True):
    points = np.array([[0, 0], [np.cos(np.pi / 3), np.sin(np.pi / 3)], [1, 0]])
    lines = [[points[i], points[(i + 1) % 3]] for i in range(3)]
    sub_triangle_points = [
        (points[i] + points[(i + 1) % 3]) / 2 for i in range(3)
    ]
    sub_triangle_lines = [[
        sub_triangle_points[i], sub_triangle_points[(i + 1) % 3]
    ] for i in range(3)]

    if inner:
        ax.add_collection(
            LineCollection(sub_triangle_lines, color="k", linestyle="dotted"))
    ax.add_collection(LineCollection(lines, color="k"))


def compute_phase_diagram(sampling_points, function, extra_args, n_jobs=1):
    print(
        f"Starting computation over {len(sampling_points)} points with {n_jobs} parallel procesess."
    )

    def computation(extra_args, Js):
        return np.array([function(J, **extra_args) for J in Js])

    t0 = time()
    with WorkerPool(n_jobs=n_jobs, shared_objects=extra_args) as pool:
        data = pool.map(computation, sampling_points, progress_bar=True).T

    dt = time() - t0
    print(f"That tooks {dt:.2f} seconds")

    return data
