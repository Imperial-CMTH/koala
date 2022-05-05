import numpy as np
from numpy.testing import assert_allclose
import pytest

from koala.pointsets import generate_random
from koala.voronization import generate_lattice
from koala.graph_color import edge_color
from koala.hamiltonian import *

n = 10
points = generate_random(n*n)
g = generate_lattice(points)
solvable, solution = edge_color(g.edges.indices, n_colors = 3)

def test_sublattice_labelling():
  # bisect lattice, dimerizing along 0 bonds
  l = bisect_lattice(g, solution, 0)
  nverts = l.vertices.positions.shape[0]
  # loop over points in the lattice
  for v_idx in range(l.vertices.positions.shape[0]):
    # find the edge containing v with color 0
    e_idx = np.any(v_idx == l.edges.indices,axis=-1) & (solution == 0)
    e_verts = l.edges.indices[e_idx][0]
    # ensure that the two ends of this edge correspond to distinct
    # sublattices
    assert(
      ((e_verts[0] >= (nverts // 2)) & (e_verts[1] < (nverts // 2)))
        | ((e_verts[0] < (nverts // 2)) & (e_verts[1] >= (nverts // 2)))
    )

def test_hamiltonians():
  # bisect lattice, dimerizing along 0 bonds
  l = bisect_lattice(g, solution, 0)

  ujk = np.full(g.n_edges, 1)
  ham = generate_majorana_hamiltonian(g, solution, ujk, np.array([1, 1, 1]))
  np.testing.assert_allclose(ham, np.conj(ham.T))
  ujk = np.full(l.n_edges, 1)
  ham_bisected = generate_majorana_hamiltonian(l, solution, ujk, np.array([1, 1, 1]))
  ham_fermionic = majorana_to_fermion_ham(ham_bisected)
  np.testing.assert_allclose(ham_fermionic, np.conj(ham_fermionic.T))
  