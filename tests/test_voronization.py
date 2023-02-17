
import numpy as np
from koala.pointsets import uniform
from koala.voronization import generate_lattice

def is_trivalent(lattice):
    return np.all(np.bincount(lattice.edges.indices.flatten()) == 3)

def test_pbc_generation():
    for n in range(3, 100, 10):
        points = uniform(n)
        g = generate_lattice(points)
        assert(is_trivalent(g)) #all vertices have coordination number 3


