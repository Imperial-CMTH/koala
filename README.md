# topamorph
A package to do topological marker calculations on amorphous lattices.

Package structure copied from [here](https://blog.ionelmc.ro/2014/05/25/python-packaging).

## Examples

```
n = 10

from topamorph.pointsets import generate_random
points = generate_random(n)

from topamorph.voronization import generate_pbc_voronoi_adjacency, cut_boundaries
vertices, adjacency = generate_pbc_voronoi_adjacency(points)

from topamorph.plotting import plot_lattice
plot_lattice(vertices, adjacency)
```