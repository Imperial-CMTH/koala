import numpy as np
from koala import lattice
from koala.lattice import Lattice
from koala.voronization import generate_lattice
from koala.pointsets import generate_random
from koala.example_graphs import *


def test_lattice_class():
    lattice1 = tutte_graph()
    lattice2 = tri_square_pent()
    points = generate_random(30)
    lattice3 = generate_lattice(points)

    assert (len(lattice3.plaquettes) == 30)