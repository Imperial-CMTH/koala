import numpy as np
from koala.pointsets import uniform, move_all_points, move_point, grid

def test_grid():
    x = grid(3,5)
    assert x.shape == (3*5, 2)
    x = grid(3,3)
    assert x.shape == (3*3, 2)
    x = grid(3,2)
    assert x.shape == (3*2, 2)


def test_uniform():
    rng = np.random.default_rng(1234)
    uniform(100, 2, rng)
    uniform(100, 2, rng)
    uniform(100, 4, rng)
    uniform(200, 2, rng)

def test_move_point():
    rng = np.random.default_rng(1234)

    points2 = uniform(100, 2, rng)
    points3 = uniform(100, 3, rng)

    move_point(points2, 1, 0.01, 0.01, resolution=10)
    move_point(points3, 1, 0.01, 0.01, resolution=10)


def test_move_all_points():
    rng = np.random.default_rng(1234)

    points2 = uniform(100, 2, rng)
    points3 = uniform(100, 3, rng)

    move_all_points(points2, 0.01, 0.01, resolution=10)
    move_all_points(points3, 0.01, 0.01, resolution=10)

    move_all_points(points2, 0.01, 0, resolution=10)
    move_all_points(points3, 0.01, 0, resolution=10)

