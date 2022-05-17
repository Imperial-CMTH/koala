<h1 align="center">🐨  Koala  🎋</h1>
<p align="center">
    <em>Putting Condensed Matter Hamiltonians on Amorphous lattice models since 2021!</em>
</p>

<p align="center">
<a href="https://zenodo.org/badge/latestdoi/422218038">
    <img src="https://zenodo.org/badge/422218038.svg"/>
</a>
<a href="https://wfxr.mit-license.org/2017">
        <img src="https://img.shields.io/badge/License-MIT-brightgreen.svg"/>
</a>
<img src="https://github.com/Imperial-CMTH/koala/actions/workflows/ci.yml/badge.svg"/>
</p>

## Changelog
- Implement a way to find all the colorings of a lattice
- Add an axes argument to plotting function
- Changed pbc_voronisation to emit a third array adjacency_crossing with size (n_edges, 2) that tells you which of the 3x3 unit cell grid this edge goes into, alternatively you can think of it as saying if this edge crosses each of the two cuts that unrwrap the torus into the plane.
- renamed SAT to graph_color 
- renamed PBV_Voronoi datastructure to Lattice
- replaced the first three arguments of plot_lattice with a Lattice object


## Todo
- For an n-gon with fixed colored edges how many valid colorings are there? Is this the origin of the degeneracy of colorings?


Flux_finder
    - currently code uses 0/1 for edge directions and +/- 1 U_ij values, do we like this?
    - do docstrings in flux_finder (except _functions)
    - Move _functions in flux_finder to bottom or top depending on what works
    - Possibly move pathfinding to a folder or something

Long Term:
    Refactor plotting
        - make functions like plot_edges, plot_vertices, plot_plaquettes...

    Add a little explanation to the beginning of each file explaining what it's for

## Examples
```
n = 10

from koala.pointsets import generate_random
points = generate_random(n)

from koala.voronization import generate_pbc_voronoi_adjacency, cut_boundaries
vertices, adjacency = generate_pbc_voronoi_adjacency(points)

from koala.plotting import plot_lattice
plot_lattice(vertices, adjacency)
```
