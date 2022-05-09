import numpy as np
import numpy.typing as npt
from koala.lattice import Lattice, permute_vertices
from koala import graph_utils

def find_plaquette(lattice, j, k):
  """Find the plaquette shared by the edge spanning second nearest neighbour vertices j and k,

  :param lattice: Lattice containing vertices j, k
  :type lattice: Lattice
  :param j: Index of first site
  :type j: int
  :param k: Index of second site
  :type k: int
  :raises ValueError: If sites j and k are not second nearest neighbours
  :return: Index of the plaquette in lattice shared by the edges connecting vertices j, k
  :rtype: int
  """
  # first find the mutual neighbouring point
  vert_neighbours_j, edge_neighbours_j = graph_utils.vertex_neighbours(lattice, j)
  vert_neighbours_k, edge_neighbours_k = graph_utils.vertex_neighbours(lattice, k)
  vert_neighbour_shared = np.intersect1d(vert_neighbours_j, vert_neighbours_k)
  
  if not len(vert_neighbour_shared) == 1:
      raise ValueError("Not second nearest neighbours")
  else:
      vert_neighbour_shared = vert_neighbour_shared[0]
  
  edge_jl = edge_neighbours_j[np.where(vert_neighbours_j == vert_neighbour_shared)]
  edge_kl = edge_neighbours_k[np.where(vert_neighbours_k == vert_neighbour_shared)]
  
  plaq_j = lattice.edges.adjacent_plaquettes[edge_jl]
  plaq_k = lattice.edges.adjacent_plaquettes[edge_kl]
  
  return np.intersect1d(plaq_j, plaq_k)[0]

def winding_sign_2nn(lattice, j, k):
  """Find winding sign of edges matching 2nd nearest neighbours j and k around their shared plaquette
  
  :param lattice: Lattice containing vertices j, k
  :type lattice: Lattice
  :param j: Index of first site
  :type j: int
  :param k: Index of second site
  :type k: int
  """
  try:
      shared_p = find_plaquette(lattice, j, k)
  except ValueError:
      # not second nearest neighbours
      return 0.0

  if shared_p > lattice.n_plaquettes:
    return 0.0
  # it suffices to find the winding direction of
  # a single edge
  a = lattice.vertices.positions[j]
  b = lattice.vertices.positions[k]
  c = lattice.plaquettes[shared_p].center
  
  return np.sign((a[0]-c[0])*(b[1]-c[1])-(a[1]-c[1])*(b[0]-c[0]))
    

def bisect_lattice(l: Lattice, solution: npt.NDArray[np.integer], along: int = 0) -> npt.NDArray[np.integer]:
  """Generate a new lattice with vertex indices permuted such that the first nvert/2 entries are in sublattice A
  and the rest are in sublattice B, according to the given coloring `solution`
  
  :param l: Non-bisected lattice
  :type l: Lattice
  :param solution: Edge coloring of `l`
  :type solution: npt.NDArray[np.integer] Shape (nedges,)
  :param along: Bond type along which to dimerize the system
  :type along: int
  :return: Bisected lattice
  :rtype: Lattice
  """    
  # Label each vertex in lattice g as belonging to A or B sublattice, according to the coloring
  # given by solution. By default, the first entry in solution will define the flavor of bond along
  # which the lattice is 'dimerized' to form the sublattices.

  nverts = l.vertices.positions.shape[0]
  # Pick all edges of color `along`
  dimer_mask = (solution == along)
  dimer_vertices = l.edges.indices[dimer_mask]
  # Create empty array with vertex sublattice labels and populate
  sublattice_labels = np.zeros(shape=(nverts,))
  sublattice_labels[dimer_vertices[:,0]] = 0
  sublattice_labels[dimer_vertices[:,1]] = 1

  vertex_reordering = np.argsort(sublattice_labels)
  bisected_lattice = permute_vertices(l, vertex_reordering)
  return bisected_lattice

def generate_majorana_hamiltonian(lattice: Lattice, coloring: npt.NDArray, ujk: npt.NDArray, J: npt.NDArray[np.floating] = (1.0,1.0,1.0), K: np.floating = 0.0) -> npt.NDArray[np.complexfloating]:
  """Assign couplings ($A_{jk} \in \pm 2J$) to each bond in lattice `l` and construct the matrix. Indices refer
  to the vertex indices in the Lattice object. This is the quadratic Majorana Hamiltonian of eqn (13)
  in Kitaev's paper. Optionally includes second nearest neighbour couplings i.e. lowest order field
  perturbation.

  :param l: Lattice to construct Hamiltonian on, from which nearest bond-sharing vertices are determined
  :type l: Lattice
  :param coloring: Edge coloring of `l`
  :type coloring: npt.NDArray[np.integer] Shape (lattice.n_edges,)
  :param J: Coupling parameter for Kitaev model, defaults to 1.0
  :type J: npt.NDArray[np.floating] or float
  :param ujk: Link variables, with value +1 or -1
  :param K: second nearest neighbour (field) coupling strength, defaults to 0.0
  :type K: np.floating

  :return: Quadratic Majorana Hamiltonian matrix representation in Majorana basis
  :rtype: npt.NDArray
  """
  edge_type_list = J[coloring]
  bond_values = 2*edge_type_list*ujk

  ham = np.zeros((lattice.n_vertices, lattice.n_vertices))
  ham[lattice.edges.indices[:,1], lattice.edges.indices[:,0]] = bond_values
  ham[lattice.edges.indices[:,0], lattice.edges.indices[:,1]] = -bond_values

  K_ham = np.zeros((lattice.n_vertices, lattice.n_vertices))

  # CHECKME: this might not be valid for any coloring other than the canonical kitaev
  # honeycomb coloring!
  if K > 0.0:
    K_ham[lattice.edges.indices[:,1], lattice.edges.indices[:,0]] = ujk
    K_ham[lattice.edges.indices[:,0], lattice.edges.indices[:,1]] = -ujk
    K_ham = K_ham @ K_ham
    K_ham *= 2*K

    indices = np.array([[(j,i) for i in np.arange(lattice.n_vertices)] for j in np.arange(lattice.n_vertices)])
    indices = indices.reshape([-1, 2])

    # winding_signs = np.array([[winding_sign(lattice,k,j) for j in range(k,lattice.n_vertices)] for k in range(0,lattice.n_vertices)])
    winding_signs = np.apply_along_axis(lambda i: winding_sign_2nn(lattice, *i), 1, indices)
    winding_signs = winding_signs.reshape([lattice.n_vertices, lattice.n_vertices])
    K_ham *= winding_signs

  return (1.0j)*(ham - K_ham)

def majorana_to_fermion_ham(majorana_ham: npt.NDArray[np.complexfloating]) -> npt.NDArray[np.complexfloating]:
  """Transforms a Hamiltonian in the Majorana basis to a Fermionic basis, whose pairing
  will be dictated by the sublattice layout of the Majoranas. Elements of the A(B) sublattice
  correspond to the first(second) half of the Majorana indices.

  :param majorana_ham: (nvert,nvert) matrix representation of Hamiltonian in Majorana basis
  :type majorana_ham: npt.NDArray[np.complexfloating]
  :return: (nvert,nvert) matrix representation of Hamiltonian in Fermion basis
  :rtype: npt.NDArray[np.complexfloating]
  """  
  sublat_size = majorana_ham.shape[0] // 2
  F = -1.0j*majorana_ham[:sublat_size, :sublat_size]
  D = -1*-1.0j*majorana_ham[sublat_size:, sublat_size:]
  M = -1.0j*majorana_ham[:sublat_size, sublat_size:]


  h = (M + M.T) + 1.0j*(F - D)
  d = (M.T - M) + 1.0j*(F + D)
  top_block = np.concatenate(
    (h, d), axis=1
  )
  bottom_block = np.concatenate(
    (np.conj(d.T), -h.T), axis=1
  )
  return np.concatenate(
    (top_block, bottom_block), axis=0
  )