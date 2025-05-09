import numpy as np

from koala.flux_finder import find_flux_sector, fluxes_from_bonds, ujk_from_fluxes, fluxes_from_ujk, loop_flux
from koala import voronization, pointsets, example_graphs
from koala.flux_finder.pathfinding import path_between_plaquettes, path_between_vertices
from koala.lattice import Lattice

# make some lattices to test
n = 15
honeycomb, _ = example_graphs.honeycomb_lattice(n, return_coloring=True)
random_points = pointsets.uniform(n*n)
amorphous = voronization.generate_lattice(random_points, shift_vertices=True)

def find_gnd_state(lattice:Lattice):
    gnd_flux = np.array([-1]*lattice.n_plaquettes)
    ujk = ujk_from_fluxes(lattice, gnd_flux)
    return ujk, gnd_flux

def test_ujk_from_fluxes():
    """find_flux_sector checks for itself if it finds an output within one flux of the input"""
    for lattice in [honeycomb, amorphous]:
        ujk, flux = find_gnd_state(lattice)
        flux_tester = fluxes_from_ujk(lattice, ujk)

        assert(np.all(flux == flux_tester))

def test_pathfinding():
    for lattice in [honeycomb, amorphous]:
        a,b = np.random.randint(lattice.n_vertices, size = (2,))
        path_between_vertices(lattice, a, b, early_stopping = False)
        path_between_vertices(lattice, a, b, early_stopping = True)

        a,b = np.random.randint(lattice.n_plaquettes, size = (2,))
        path_between_plaquettes(lattice, a, b, early_stopping = False)
        path_between_plaquettes(lattice, a, b, early_stopping = True)

# test old flux code
def find_random_flux_sector(l):
    """Tries to find a random flux sector."""
    random_fluxes = np.random.choice([-1, 1], size = l.n_plaquettes)
    bonds = find_flux_sector(l, random_fluxes)
    fluxes = fluxes_from_bonds(l, bonds)
    return bonds, fluxes

def test_flux_finder():
    """find_flux_sector checks for itself if it finds an output within one flux of the input"""
    find_random_flux_sector(honeycomb)
    find_random_flux_sector(amorphous)


def test_loop_flux():


    def _test_loop_flux(lattice):
        ujk = 1-2*np.random.choice(2, lattice.n_edges)
        fluxes = fluxes_from_ujk(lattice, ujk)

        for j, plaq in enumerate(lattice.plaquettes):
            x = loop_flux(lattice, plaq.edges, ujk)
            if len(plaq.edges)%2==0:
                assert x == fluxes[j]

    _test_loop_flux(honeycomb)
    _test_loop_flux(amorphous)
