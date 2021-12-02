import numpy as np
import numpy.typing as npt
from koala.lattice import Lattice, permute_vertices

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

  nverts = l.vertices.shape[0]
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

# TODO: generalize this to anisotropic couplings
def generate_majorana_hamiltonian(l: Lattice, J: float = 1.0, method: str = 'random') -> npt.NDArray[np.complexfloating]:
  """Assign couplings ($A_{jk} \in \pm 2J$) to each bond in lattice `l` and construct the matrix. Indices refer
  to the vertex indices in the Lattice object. This is the quadratic Majorana Hamiltonian of eqn (13)
  in Kitaev's paper.

  :param l: Lattice to construct Hamiltonian on, from which nearest bond-sharing vertices are determined
  :type l: Lattice
  :param J: Coupling parameter for Kitaev model, defaults to 1.0
  :type J: float, optional
  :param method: Assignment rule for couplings, either 'random' or 'uniform, defaults to 'random'
  :type method: str, optional
  :return: Quadratic Majorana Hamiltonian matrix representation in Majorana basis
  :rtype: npt.NDArray
  """  
  nverts = l.vertices.shape[0]
  nedges = l.edges.indices.shape[0]
  ham = np.zeros((nverts, nverts))
  
  if method == 'random':
    hoppings = 2*J*np.random.choice([1,-1], size=(nedges,))
  elif method == 'uniform':
    hoppings = 2*J*np.ones((nedges,))
  
  ham[l.edges.indices[:,0], l.edges.indices[:,1]] = hoppings
  ham[l.edges.indices[:,1], l.edges.indices[:,0]] = -1*hoppings

  ham = ham * 1.0j / 4.0

  return ham

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