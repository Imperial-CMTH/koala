import numpy as np
from numpy.testing import assert_allclose
import pytest

from koala.pointsets import generate_bluenoise
from koala.voronization import generate_lattice
from koala.graph_color import edge_color
from koala.hamiltonian import *

def test_sublattice_labelling():
    n = 20
    points = generate_bluenoise(30,n,n)
    g = generate_lattice(points)
    solvable, solution = edge_color(g.edges.indices, n_colors = 3)
    # bisect lattice, dimerizing along 0 bonds
    l = bisect_lattice(g, solution, 0)
    nverts = l.vertices.shape[0]
    # loop over points in the lattice
    for v_idx in range(l.vertices.shape[0]):
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
  n = 20
  points = generate_bluenoise(30,n,n)
  g = generate_lattice(points)
  solvable, solution = edge_color(g.edges.indices, n_colors = 3)
  # bisect lattice, dimerizing along 0 bonds
  l = bisect_lattice(g, solution, 0)

  ham = generate_majorana_hamiltonian(g)
  np.testing.assert_allclose(ham, np.conj(ham.T))
  ham_bisected = generate_majorana_hamiltonian(l)
  ham_fermionic = majorana_to_fermion_ham(ham_bisected)
  np.testing.assert_allclose(ham_fermionic, np.conj(ham_fermionic.T))
  