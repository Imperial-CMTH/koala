import numpy as np
from numpy.testing import assert_allclose
import pytest

from koala.pointsets import generate_random
from koala.voronization import generate_lattice
from koala.graph_color import color_lattice
from koala.hamiltonian import *

n = 10
points = generate_random(n*n)
lattice = generate_lattice(points)
coloring = color_lattice(lattice)

def test_sublattice_labelling():
  # bisect lattice, dimerizing along 0 bonds
  l = bisect_lattice(lattice, coloring, 0)
  nverts = l.vertices.positions.shape[0]
  # loop over points in the lattice
  for v_idx in range(l.vertices.positions.shape[0]):
    # find the edge containing v with color 0
    e_idx = np.any(v_idx == l.edges.indices,axis=-1) & (coloring == 0)
    e_verts = l.edges.indices[e_idx][0]
    # ensure that the two ends of this edge correspond to distinct
    # sublattices
    assert(
      ((e_verts[0] >= (nverts // 2)) & (e_verts[1] < (nverts // 2)))
        | ((e_verts[0] < (nverts // 2)) & (e_verts[1] >= (nverts // 2)))
    )

def test_hamiltonians():
  # bisect lattice, dimerizing along 0 bonds
  l = bisect_lattice(lattice, coloring, 0)

  ujk = np.full(lattice.n_edges, 1)
  ham = majorana_hamiltonian(lattice, coloring, ujk, np.array([1, 1, 1]))
  np.testing.assert_allclose(ham, np.conj(ham.T))
  ujk = np.full(l.n_edges, 1)
  ham_bisected = majorana_hamiltonian(l, coloring, ujk, np.array([1, 1, 1]))
  ham_fermionic = majorana_to_fermion_ham(ham_bisected)
  np.testing.assert_allclose(ham_fermionic, np.conj(ham_fermionic.T))
  