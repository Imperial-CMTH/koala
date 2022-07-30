import numpy as np
import numpy.typing as npt
from koala.lattice import Lattice, permute_vertices


def bisect_lattice(lattice: Lattice,
                   solution: npt.NDArray[np.integer],
                   along: int = 0) -> npt.NDArray[np.integer]:
    """Generate a new lattice with vertex indices permuted such that the first nvert/2 entries are in sublattice A
  and the rest are in sublattice B, according to the given coloring `solution`

  Args:
      l (Lattice): Non-bisected lattice
      solution (npt.NDArray[np.integer] Shape (nedges,)): Edge coloring
          of `l`
      along (int): Bond type along which to dimerize the system

  Returns:
      Lattice: Bisected lattice
  """
    # Label each vertex in lattice g as belonging to A or B sublattice, according to the coloring
    # given by solution. By default, the first entry in solution will define the flavor of bond along
    # which the lattice is 'dimerized' to form the sublattices.

    nverts = lattice.vertices.positions.shape[0]
    # Pick all edges of color `along`
    dimer_mask = (solution == along)
    dimer_vertices = lattice.edges.indices[dimer_mask]
    # Create empty array with vertex sublattice labels and populate
    sublattice_labels = np.zeros(shape=(nverts,))
    sublattice_labels[dimer_vertices[:, 0]] = 0
    sublattice_labels[dimer_vertices[:, 1]] = 1

    vertex_reordering = np.argsort(sublattice_labels)
    bisected_lattice = permute_vertices(lattice, vertex_reordering)
    return bisected_lattice


def majorana_hamiltonian(
    lattice: Lattice,
    coloring: npt.NDArray,
    ujk: npt.NDArray,
    J: npt.NDArray[np.floating] = np.array([1.0, 1.0, 1.0])
) -> npt.NDArray[np.complexfloating]:
    """Assign couplings ($A_{jk} in pm 2J$) to each bond in lattice `l` and construct the matrix. Indices refer
  to the vertex indices in the Lattice object. This is the quadratic Majorana Hamiltonian of eqn (13)
  in Kitaev's paper.

  Args:
      l (Lattice): Lattice to construct Hamiltonian on, from which
          nearest bond-sharing vertices are determined
      coloring (npt.NDArray[np.integer] Shape (lattice.n_edges,)): Edge
          coloring of `l`
      J (npt.NDArray[np.floating] or float): Coupling parameter for
          Kitaev model, defaults to 1.0
      ujk: Link variables, with value +1 or -1

  Returns:
      npt.NDArray: Quadratic Majorana Hamiltonian matrix representation
      in Majorana basis
  """
    ham = np.zeros((lattice.n_vertices, lattice.n_vertices))
    Js = J[coloring] if coloring is not None else J[0]
    hoppings = 2 * Js * ujk

    ham[lattice.edges.indices[:, 1], lattice.edges.indices[:, 0]] = hoppings
    ham[lattice.edges.indices[:, 0], lattice.edges.indices[:,
                                                           1]] = -1 * hoppings

    ham = ham * 1.0j / 4.0

    return ham


def majorana_to_fermion_ham(
    majorana_ham: npt.NDArray[np.complexfloating]
) -> npt.NDArray[np.complexfloating]:
    """Transforms a Hamiltonian in the Majorana basis to a Fermionic basis, whose pairing
  will be dictated by the sublattice layout of the Majoranas. Elements of the A(B) sublattice
  correspond to the first(second) half of the Majorana indices.

  Args:
      majorana_ham (npt.NDArray[np.complexfloating]): (nvert,nvert)
          matrix representation of Hamiltonian in Majorana basis

  Returns:
      npt.NDArray[np.complexfloating]: (nvert,nvert) matrix
      representation of Hamiltonian in Fermion basis
  """
    sublat_size = majorana_ham.shape[0] // 2
    F = -1.0j * majorana_ham[:sublat_size, :sublat_size]
    D = -1 * -1.0j * majorana_ham[sublat_size:, sublat_size:]
    M = -1.0j * majorana_ham[:sublat_size, sublat_size:]

    h = (M + M.T) + 1.0j * (F - D)
    d = (M.T - M) + 1.0j * (F + D)
    top_block = np.concatenate((h, d), axis=1)
    bottom_block = np.concatenate((np.conj(d.T), -h.T), axis=1)
    return np.concatenate((top_block, bottom_block), axis=0)
