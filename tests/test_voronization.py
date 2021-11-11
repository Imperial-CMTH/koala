
import numpy as np
import pytest
from koala.pointsets import generate_random
from koala.voronization import generate_pbc_voronoi_adjacency

def test_pbc_generation():
    for n in range(3, 100, 10):
        points = generate_random(n)
        g = generate_pbc_voronoi_adjacency(points)
        assert(np.all(np.bincount(g.adjacency.flatten()) == 3)) #all vertices have coordination number 3
