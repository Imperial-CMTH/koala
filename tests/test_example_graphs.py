import numpy as np
from koala.example_graphs import generate_honeycomb


def is_trivalent(lattice):
    return np.all(np.bincount(lattice.edges.indices.flatten()) == 3)

def color_honeycomb_lattice(lattice):
    "Return an edge coloring of the honeycomb lattice using bond angles" 
    v = lattice.edges.vectors
    angles = np.floor(np.arctan2(v[:, 0], v[:, 1]))
    
    labels = 3*np.ones(angles.shape, dtype = int) #use 3 as an invalid value
    labels[angles == 0.0] = 0 #red bonds pointing up
    labels[angles == 2.0] = 1 #green bonds point to the right
    labels[angles == -3.0] = 2 #blue bonds point to the left
    
    return labels
 
def test_honeycomb():
    N = 10
    honeycomb = generate_honeycomb(N, return_colouring=False)
    honeycomb, coloring = generate_honeycomb(N, return_colouring=True)

    #all vertices have coordination number 3
    assert(is_trivalent(honeycomb)) 

    #all plaquettes have 6 sides
    sides = np.array([p.n_sides for p in honeycomb.plaquettes])
    assert(np.all(sides == 6))

    # check the colors make sense by checking the angles
    assert np.all(coloring == color_honeycomb_lattice(honeycomb))

    # check that two sublattices are a bi-partition of the lattice
    honeycomb_sublattice_labels = np.arange(honeycomb.n_vertices) % 2
    red_edges = honeycomb.edges.indices[coloring == 0]
    sublattice_edges = honeycomb_sublattice_labels[red_edges]
    assert(not np.any(sublattice_edges[:, 0] == sublattice_edges[:, 1]))



