############################################################################
# Routines for generating amorphous pointsets                              #
############################################################################

import numpy as np
from matplotlib import pyplot as plt

def grid(nx: int , ny: int ) -> np.ndarray:
    """Generates a uniformly spaced grid of points

    Args:
        nx (int): Number of points in x direction
        ny (int): Number of points in y direction

    Returns:
        np.ndarray: List of all the positions.
    """

    pos_x = np.linspace(0,1,nx, endpoint=False)
    pos_y = np.linspace(0,1,ny, endpoint=False)
    pos_x += 0.5*pos_x[1]
    pos_y += 0.5*pos_y[1]
    g_out = np.reshape(np.meshgrid(pos_x, pos_y), [2,-1]).T
    return g_out


def bluenoise(k, nx, ny, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    r = 1
    coords = [(x, y) for x in range(nx) for y in range(ny)]
    cells = {coord: None for coord in coords}

    def point_to_coord(point):
        return tuple(point.astype(np.uint32))

    def disk_uniform(r1, r2):
        # Generate a random point inside 2-disc [r1, r2]
        rho, theta = rng.uniform(r1, r2), rng.uniform(0, 2 * np.pi)
        return np.array([rho * np.cos(theta), rho * np.sin(theta)])

    x0 = rng.uniform(size=(2,)) * np.array([nx, ny])
    samples = [x0]
    cells[point_to_coord(x0)] = 0
    active_cells = [0]

    nsamples = 1
    while active_cells:
        idx = rng.choice(active_cells)
        x0 = samples[idx]
        # Generate new point within r and 2r of x0
        for i in range(k):
            x1 = x0 + disk_uniform(r, 2 * r)
            if np.any(x1 < 0) or np.any(x1 > nx):
                continue
            # TODO: make this more efficient by only checking points in
            # neighbouring grid cells
            if np.min(np.linalg.norm(x1 - np.array(samples), axis=-1)) > r:
                samples.append(x1)
                nsamples += 1
                active_cells.append(len(samples) - 1)
                cells[point_to_coord(x1)] = len(samples) - 1
                break
            elif i == k - 1:
                active_cells.remove(idx)
            else:
                continue
        # print(active_cells)

    return np.array(samples) / np.array([nx, ny])


def hyperuniform(nx, ny, kickstrength=1e-3, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    x = np.linspace(0, 1, nx)
    y = np.linspace(0, 1, ny)
    X, Y = np.meshgrid(x, y)
    cell_origins = np.array([X.flatten(), Y.flatten()]).T
    # Create points by displacing cell origins by uniform
    cell_offsets = np.random.uniform(0, 1, size=(nx * ny, 2)) * np.array(
        [1 / nx, 1 / ny]
    )
    initial_points = cell_origins + cell_offsets
    # Generate power law kicks from Pareto distribution, a=0.45
    # as mentioned in supp info
    mags = rng.pareto(a=0.45, size=(nx * ny, 2))
    dirs = rng.uniform(size=(nx * ny,)) * 2 * np.pi
    kicks = (
        kickstrength
        * np.array([mags[:, 0] * np.cos(dirs), mags[:, 1] * np.sin(dirs)]).T
    )
    final_points = initial_points + kicks
    # Crop to interval [0, 1]
    return final_points[
        np.where(np.all(final_points > 0, axis=-1) & np.all(final_points < 1, axis=-1))
    ]


def uniform(n, dim=2, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    return rng.uniform(size=(n, dim))


def _debug_move_plot(points, start_pos, positions_unlooped, prob_dist):
    _, ax = plt.subplots(figsize=(5, 5), dpi=200)
    ax.scatter(*points.T, s=2)
    ax.scatter(*start_pos, c="b", s=3)
    ax.pcolormesh(positions_unlooped[0], positions_unlooped[1], prob_dist, zorder=-10)
    ax.pcolormesh(
        positions_unlooped[0] + 1, positions_unlooped[1], prob_dist, zorder=-10
    )
    ax.pcolormesh(
        positions_unlooped[0], positions_unlooped[1] + 1, prob_dist, zorder=-10
    )
    ax.pcolormesh(
        positions_unlooped[0] - 1, positions_unlooped[1], prob_dist, zorder=-10
    )
    ax.pcolormesh(
        positions_unlooped[0], positions_unlooped[1] - 1, prob_dist, zorder=-10
    )
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.show()


# TODO - optimise this
def move_point(
    points: np.ndarray,
    index_to_move: int,
    sigma: float,
    kappa: float,
    beta: float = 1,
    move_limit: float = 4,
    resolution: int = 40,
    rng: np.random.Generator = None,
    debug_plot: bool = False,
) -> np.ndarray:
    """Given a set of points, and an index for the chosen point. Move the chosen point to a random new position subject to two constriants.
    1: The new point is in a gaussian circle around the original position, with radius specufied by sigma
    2: The new position is not within kappa or any other existing points

    Args:
        points (np.ndarray): Positions of all vertices
        index_to_move (int): Index of the vertex to be shifted
        sigma (float): Step size of the gaussian distribution around the original position
        kappa (float): Lengthscale of the repulsion around each other point
        beta (float, optional): Temperature, controls the degree of randomness. Defaults to 1.
        move_limit (float, optional): Points are only considered within a range of move_limit*sigma of the original point. Defaults to 4.
        resolution (int, optional): How finely to sample the probability distribution for new points to move to. Slows down the code if large. Defaults to 40.
        rng (np.random.Generator, optional): rng for all the random stuff. Defaults to None.
        debug_plot (bool, optional): If True, outputs a plot of all the points and the probability distribution - only in 2D. Defaults to False.

    Returns:
        np.ndarray: The new points with one moved
    """

    if rng is None:
        rng = np.random.default_rng()

    if np.allclose(sigma, 0):
        return points

    # make a grid of options for where to move it
    max_distance = sigma * move_limit
    pos_grid = np.linspace(-max_distance, max_distance, resolution)
    box_size = pos_grid[1] - pos_grid[0]
    dimensionality = points.shape[1]
    a = [pos_grid for j in range(dimensionality)]
    a = np.array(np.meshgrid(*a))

    # gaussian hamiltonian
    h_gauss = np.sum(a**2 / (2 * sigma**2), axis=0)

    # find the points inside the move limit from the initial vertex
    start_pos = points[index_to_move]
    vectors = points - start_pos
    vectors = (vectors + 0.5) % 1 - 0.5
    distances_squared = np.sum(vectors**2, axis=1)
    c1 = distances_squared < (move_limit * sigma * np.sqrt(2))  # within region
    c2 = np.arange(len(points)) != index_to_move
    nearest_points = np.where(c1 & c2)[0]

    # make the repulsion
    h_int = h_gauss * 0
    positions_unlooped = start_pos[:, *[None] * dimensionality] + a
    positions = positions_unlooped % 1
    for neighbour in nearest_points:
        n_pos = points[neighbour]
        r = positions - n_pos[:, *[None] * dimensionality]
        r = (r + 0.5) % 1 - 0.5
        r = np.sum(r**2, axis=0)
        r += 1e-16 # stop it breaking when r vanishes
        h_int += (kappa) / (r**0.5)  # np.sum(r**2, axis = 0)

    # find prob dist
    total_energy = h_int + h_gauss
    total_energy = total_energy - np.min(
        total_energy
    )  # remove any constant shift -- just makes numbers blow up
    prob_dist = np.exp(-beta * (total_energy))

    # plot for debugging
    if dimensionality == 2 and debug_plot:
        _debug_move_plot(points, start_pos, positions_unlooped, prob_dist)

    if np.isclose(np.sum(prob_dist), 0):
        raise ValueError(
            f"Beta is too high, the probability distribution vanishes ({np.sum(prob_dist)})"
        )

    prob_dist = prob_dist / np.sum(prob_dist)
    positions_flattened = np.reshape(positions, [dimensionality, -1])
    prob_flat = np.reshape(prob_dist, [-1])

    # sample it
    chosen_position = rng.choice(range(len(prob_flat)), p=prob_flat)
    chosen_position = positions_flattened[:, chosen_position]

    # randomise within the resolution of the distribution
    chosen_position += rng.uniform(-box_size / 2, box_size / 2, chosen_position.shape)

    out = points.copy()
    out[index_to_move] = chosen_position
    return out


def move_all_points(
    points: np.ndarray,
    sigma: float,
    kappa: float = None,
    beta: float = 1,
    rng: np.random.Generator = None,
    **kwargs,
) -> np.ndarray:
    """A wrapper for move_point. Chooses a random ordeing of all the points in the list, and then moves each one according to the parameters provided.

    Args:
        points (np.ndarray): Positions of all vertices
        sigma (float): Step size of the gaussian distribution around each original position
        kappa (float): Lengthscale of the repulsion between points
        beta (float, optional): Temperature, controls the degree of randomness. Defaults to 1.
        rng (np.random.Generator, optional): rng for all the random stuff. Defaults to None.

    Returns:
        np.ndarray: The new shifted points.
    """

    if rng is None:
        rng = np.random.default_rng()

    if kappa is None:
        kappa = 1 / np.sqrt(np.pi * len(points))

    p = points.copy()

    random_order = rng.permutation(range(len(p)))
    for v in random_order:

        if kappa == 0:
            p = gaussian_move_point(p, v, sigma, beta, rng)
        else:
            p = move_point(p, v, sigma, kappa, rng=rng, beta=beta, **kwargs)

    return p


def gaussian_move_point(
    points: np.ndarray,
    index_to_move: int,
    sigma: float,
    beta: float = 1,
    rng: np.random.Generator = None,
) -> np.ndarray:
    """Moves a point to an arbitrary position in a gaissian of sigma,   

    Args:
        points (np.ndarray): Positions of all vertices
        index_to_move (int): Index of the vertex to be shifted
        sigma (float): Step size of the gaussian distribution around the original position
        beta (float, optional): Temperature, controls the degree of randomness. Defaults to 1.
        rng (np.random.Generator, optional): rng for all the random stuff. Defaults to None.

    Returns:
        np.ndarray: The new points with one moved
    """
    
    dim = points.shape[1]
    rescaled_sigma = sigma/np.sqrt(beta)
    shift = rng.normal(0, rescaled_sigma, dim)
    points[index_to_move] = points[index_to_move]+shift

    return points
