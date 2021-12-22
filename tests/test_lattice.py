import numpy as np
from koala import pointsets
from matplotlib import pyplot as plt
from koala import voronization
from koala import plotting
from koala.lattice import cut_boundaries
from koala.example_graphs import *

def test_lattice_class():

    # generate a ton of weird graphs
    
    tri_square_pent()
    two_tri()
    tutte_graph()
    n_ladder(6,True)
    points1 = pointsets.generate_random(50)
    voronization.generate_lattice(points1)
    points2 = pointsets.generate_bluenoise(30,3,3)
    voronization.generate_lattice(points2)
    cut_boundaries(voronization.generate_lattice(points2), [False,True])
    cut_boundaries(voronization.generate_lattice(points2), [True,True])

