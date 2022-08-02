############################################################################
# Routines for generating amorphous pointsets                              #
############################################################################

import numpy as np


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
        [1 / nx, 1 / ny])
    initial_points = cell_origins + cell_offsets
    # Generate power law kicks from Pareto distribution, a=0.45
    # as mentioned in supp info
    mags = rng.pareto(a=0.45, size=(nx * ny, 2))
    dirs = rng.uniform(size=(nx * ny,)) * 2 * np.pi
    kicks = kickstrength * \
        np.array([mags[:, 0]*np.cos(dirs), mags[:, 1]*np.sin(dirs)]).T
    final_points = initial_points + kicks
    # Crop to interval [0, 1]
    return final_points[np.where(
        np.all(final_points > 0, axis=-1) & np.all(final_points < 1, axis=-1))]


def uniform(n, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    points = rng.uniform(size=(n, 2))
    return points
