from koala.quasicrystals import de_brujin_grid, penrose_tiling
import numpy as np

def test_penrose_tiling():

    n_lines = [5,6,7]

    for n in n_lines:
        _ = penrose_tiling(n)

def test_de_brujin_grid():

    n_lines = [5,6,7]

    for n in n_lines:

        rands = np.random.random(7)
        _ = de_brujin_grid(n, 7,rands,0.1)
