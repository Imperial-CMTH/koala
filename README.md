# koala
A package to do topological marker calculations on amorphous lattices.

Package structure copied from [here](https://blog.ionelmc.ro/2014/05/25/python-packaging).

## Changelog
- Implement a way to find all the colourings of a lattice
- Add an axes argument to plotting function
- Changed pbc_voronisation to emit a third array adjacency_crossing with size (n_edges, 2) that tells you which of the 3x3 unit cell grid this edge goes into, alternatively you can think of it as saying if this edge crosses each of the two cuts that unrwrap the torus into the plane.

## Todo
- Implement enumerating plaquettes (should be easy to start from vor.regions)
- Make a way to do a regular honeycomb lattice
- For an n-gon with fixed colored edges how many valid colourings are there? Is this the origin of the degeneracy of colorings?
- Return the kdtree of the lattice vertices because it's useful for other stuff
- make function convert from colorings to lists of color strings and put it in plotting.py
- Compute a histogram of the nummber of edsges in each plaquette
- Make adjacency_crossing an optional argument to the plot_lattice function.

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