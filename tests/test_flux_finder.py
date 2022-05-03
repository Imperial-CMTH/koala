import numpy as np

from koala.flux_finder import find_flux_sector, fluxes_from_bonds
from koala import voronization, pointsets, example_graphs
from koala import graph_color

def find_random_flux_sector(l):
    """Tries to find a random flux sector."""
    random_fluxes = np.random.choice([-1, 1], size = l.n_plaquettes)
    bonds = find_flux_sector(l, random_fluxes)
    fluxes = fluxes_from_bonds(l, bonds)
    return bonds, fluxes

def test_flux_finder():
    """find_flux_sector checks for itself if it finds an output within one flux of the input"""
    n = 15
    honeycomb, _ = example_graphs.generate_honeycomb(n, return_coloring=True)
    honeycomb_bonds, _ = find_random_flux_sector(honeycomb)
    
    random_points = pointsets.generate_random(n*n)
    amorphous = voronization.generate_lattice(random_points, shift_vertices=True)
    amorphous_bonds, _ = find_random_flux_sector(amorphous)