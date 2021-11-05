# koala
A package to do topological marker calculations on amorphous lattices.

Package structure copied from [here](https://blog.ionelmc.ro/2014/05/25/python-packaging).

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