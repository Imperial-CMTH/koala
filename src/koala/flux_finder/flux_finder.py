import numpy as np
from ..lattice import INVALID, Lattice
from .pathfinding import straight_line_length, path_between_plaquettes
import functools
from collections import Counter
import warnings


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.warn(
            "Call to deprecated function {}.".format(func.__name__),
            category=DeprecationWarning,
            stacklevel=2,
        )
        return func(*args, **kwargs)

    return new_func



def _loop_sign(lattice: Lattice, loop_edges: np.ndarray) -> int:

    loop_indices = lattice.edges.indices[loop_edges].copy()
    sign = 1

    position = [0, 0]
    for _ in loop_edges[:-1]:
        ar_head = tuple([position[0], 1 - position[1]])
        next_site = loop_indices[ar_head]
        loop_indices[position[0]] = -1
        position = np.where(loop_indices == next_site)
        sign *= (-1) ** position[1]

    return int(sign)

def loop_flux(lattice: Lattice, loop_edges: np.ndarray, ujk: np.ndarray) -> int:
    """Given a loop, calculate the flux through the loop

    Args:
        lattice (Lattice): The lattice to work on
        loop_edges (np.ndarray): List of edges that comprise the loop.
        ujk (np.ndarray): Value of the flux on all edges in the lattice.

    Returns:
        int: The flux through the loop
    """
    return int(_loop_sign(lattice, loop_edges) * np.prod(ujk[loop_edges]))

def fluxes_from_ujk(lattice: Lattice, ujk: np.ndarray, real=True) -> np.ndarray:
    """Given a lattice l and a set of bonds = +/-1 defined on the edges of the lattice, calculate the fluxes.
    The fluxes are defined on each plaquette as the the product of each bond in the direction,
    with the sign fliped otherwise.

    Args:
        lattice (Lattice): The lattice
        ujk (np.ndarray): The bond variables +1 or -1
        real (bool, optional): If true, gives the real flux according to f = f_real*(i^n) . Defaults to True.

    Returns:
        np.ndarray: The fluxes
    """
    fluxes = np.zeros(lattice.n_plaquettes, dtype='int') if real else np.zeros(
        lattice.n_plaquettes, dtype='complex')

    for i, p in enumerate(lattice.plaquettes):
        bond_signs = ujk[p.edges]
        if not real:
            bond_signs = bond_signs * (1j)
        fluxes[i] = np.prod(-bond_signs * p.directions)
    return fluxes


def ujk_from_fluxes(lattice: Lattice,
                    target_flux_sector: np.ndarray = None,
                    initial_ujk_guess: np.ndarray = None):
    """given a target (real) flux sector, find a set of u_jk bond operators that correspond to that flux sector

    Args:
        lattice (Lattice): the lattice
        target_flux_sector (np.ndarray, optional): the target value for flux through every plaquette. Must be real and correspond to the product of ujks around the plaquette. Defaults to the ground state (all fluxes = -1).
        initial_bond_guess (np.ndarray, optional): a guess for the starting values of the bonds, if none then we assume all ujk = 1. Defaults to None.

    Raises:
        ValueError: find_flux_sector failed to get rid of all the -1 fluxes, this is a bug.
        ValueError: find_flux_sector thought that it worked but somehow still didn't, a bug.

    Returns:
        _type_: _description_ 
    """

    if target_flux_sector is None:
        target_flux_sector = np.full(lattice.n_plaquettes, -1, dtype=np.int8)
    if initial_ujk_guess is None:
        initial_ujk_guess = np.ones(lattice.n_edges, dtype=np.int8)
    initial_ujk_guess, target_flux_sector = initial_ujk_guess.copy(
    ), target_flux_sector.copy()

    initial_flux_sector = fluxes_from_ujk(lattice, initial_ujk_guess)
    fluxes_to_flip = target_flux_sector // initial_flux_sector

    # figure out which fluxes need to be flipped
    
    # check if the system has an odd number of plaqs that need flipped
    # if this is the case then the open boundaries also require flipping!
    if np.sum(fluxes_to_flip == -1) % 2 == 1:
        if lattice.boundary_conditions.sum() == 2:
            raise ValueError(
                "a system with periodic boundaries has an odd number to flip, this is a bug."
            )
        
        # find an edge that connects top the boundary
        e = lattice.edges.adjacent_plaquettes
        chosen_edge = np.where(lattice.edges.adjacent_plaquettes == INVALID)[0][0]
        initial_ujk_guess[chosen_edge] = -1

    initial_flux_sector = fluxes_from_ujk(lattice, initial_ujk_guess)
    fluxes_to_flip = target_flux_sector // initial_flux_sector
    if np.sum(fluxes_to_flip == -1) % 2 == 1:
        raise ValueError(
            "not managing to fix the exact flux sector, this is a bug."
        )

    # step 2) flip edges that join adjacent -1 fluxes
    bonds, fluxes_to_flip = _flip_adjacent_fluxes(lattice, initial_ujk_guess,
                                                  fluxes_to_flip)

    # steps 3) 4) and 5) Use A* to flip the remainging isolated fluxes
    bonds, fluxes_to_flip = _flip_isolated_fluxes(lattice, bonds,
                                                  fluxes_to_flip)

    if np.count_nonzero(fluxes_to_flip == -1) > 0:
        raise ValueError(
            "find_flux_sector failed to get rid of all the -1 fluxes, this is a bug."
        )

    found_fluxes = fluxes_from_ujk(lattice, bonds)

    if np.count_nonzero(found_fluxes - target_flux_sector) > 0:
        raise ValueError(
            "find_flux_sector thought that it worked but somehow still didn't, a bug."
        )

    return bonds.astype(np.int8)


def n_to_ujk_flipped(n: int, ujk: np.ndarray, min_spanning_set: np.ndarray):
    """given an integer n in the 0 - 2^(number of edges in spanning tree), this code flips the edges of ujk in the spanning tree in the combination given by the binary representation of n. Useful for exhaustively searching over the entire flux space of a lattice.

    Args:
        n (int): number determining the flip configuration
        ujk (np.ndarray( ±1 )): the edge signs of the whole lattice
        min_spanning_set (np.ndarray( int )): an array containing a
            minimum set of edges that form a plaquete-spanning tree on
            the system

    Returns:
        np.ndarray( ±1 ): a flipped set of ujk
    """
    n_in_tree = len(min_spanning_set)
    str_format = '0' + str(n_in_tree) + 'b'
    flips = np.array([int(x) for x in format(n, str_format)])
    ujk_flipped = ujk.copy()
    new_f_values = 1 - 2 * flips
    ujk_flipped[min_spanning_set] = new_f_values
    return ujk_flipped.astype(np.int8)


def fluxes_to_labels(fluxes: np.ndarray) -> np.ndarray:
    """Remaps fluxes from the set {1,-1} to labels in the form {0,1} for plotting.

    Args:
        fluxes (np.ndarray): Fluxes in the format +1 or -1

    Returns:
        np.ndarray: labels in [0,1]
    """
    return ((1 - fluxes) // 2).astype(np.int8)


def _random_bonds(l: Lattice) -> np.ndarray:
    return 1 - 2 * np.random.choice(2, size=l.n_edges)


def _greedy_plaquette_pairing(plaquettes, distance_func):
    if plaquettes.shape[0] % 2 == 1:
        plaquettes = plaquettes[:-1]

    to_pair = set(plaquettes)
    pairs = []
    while to_pair:
        cur = to_pair.pop()
        _, closest = min(
            ((distance_func(cur, other), other) for other in to_pair))
        pairs.append((cur, closest))
        to_pair.remove(closest)

    return np.array(pairs)


def _flip_adjacent_fluxes(l: Lattice, bonds: np.ndarray, fluxes: np.ndarray):
    """See docs for find_flux_sector, this helper method implements step 2)
    which is that it looks for adjacent -1 fluxes and flips their edges
    """
    for edge_index, (p_a, p_b) in enumerate(l.edges.adjacent_plaquettes):
        if (p_a == INVALID) or (p_b == INVALID):
            break
        if (fluxes[p_a] == -1) and (fluxes[p_b] == -1):
            bonds[edge_index] *= -1
            fluxes[p_a] *= -1
            fluxes[p_b] *= -1

    #attempt at vectorising, check this at somepoint
    #adj_fluxes = fluxes[l.edges.adjacent_plaquettes]
    #to_flip = np.where((adj_fluxes[:, 0] == -1) & (adj_fluxes[:, 1] == -1))
    #bonds[to_flip] *= -1
    #fluxes_to_flip = l.edges.adjacent_plaquettes[to_flip].flatten()
    #fluxes[fluxes_to_flip] *= -1

    return bonds, fluxes


def _flip_isolated_fluxes(l: Lattice, bonds: np.ndarray, fluxes: np.ndarray):
    """See docs for find_flux_sector, this helper method implements step 3-5)"""
    indices_to_flip = np.where(fluxes == -1)[0]

    def pos(p):
        return l.plaquettes[p].center

    def distance_func(a, b):
        return straight_line_length(pos(a), pos(b))

    close_pairs = _greedy_plaquette_pairing(indices_to_flip, distance_func)

    for a, b in close_pairs:
        plaquettes, edges_to_flip = path_between_plaquettes(l,
                                                            a,
                                                            b,
                                                            maxits=l.n_edges)
        bonds[edges_to_flip] *= -1
        fluxes[a] *= -1
        fluxes[b] *= -1

    return bonds, fluxes


######################################################################################################
######################################## deprecated functions ########################################
######################################################################################################

# these functions use an old definition of the real flux - found by taking the real or imaginary part of the imag flux
# really it makes more sense to just use the clockwise product of ujk, thich is what the new functions use


@deprecated
def fluxes_from_bonds(lattice: Lattice,
                      ujk: np.ndarray,
                      real=True) -> np.ndarray:
    """Given a lattice l and a set of bonds = +/-1 defined on the edges of the lattice, calculate the fluxes.
    The fluxes are defined on each plaquette as the the product of each bond in the direction,
    with the sign fliped otherwise.

    Args:
        lattice (Lattice): The lattice
        ujk (np.ndarray): The bond variables +1 or -1
        real (bool, optional): If true, gives the real coefficient of the real or imaginary fluxes . Defaults to True.

    Returns:
        np.ndarray: The fluxes
    """
    fluxes = np.zeros(lattice.n_plaquettes, dtype='int') if real else np.zeros(
        lattice.n_plaquettes, dtype='complex')

    for i, p in enumerate(lattice.plaquettes):
        bond_signs = ujk[p.edges]  #the signs of the bonds around this plaquette

        sign_real = np.array([1, -1, -1, 1])
        sign = sign_real[p.n_sides % 4]

        if not real and p.n_sides % 2 == 1:
            sign = sign * 1j

        fluxes[i] = sign * np.prod(bond_signs * p.directions)

    return fluxes


@deprecated
def find_flux_sector(
    lattice: Lattice,
    target_flux_sector: np.ndarray = None,
    initial_bond_guess: np.ndarray = None,
):
    """Find a bond configuration that produces the given flux sector for the lattice l.

    The high level method is:
        1) Figure out which fluxes need to flip using fluxes_to_flip = target_flux_sector / initial_flux_sector
        2) Look for any adjacent -1 fluxes in fluxes_to_flip that can be anhilated by just flipping 1 edge.
        3) Pair the remaining fluxes off into close-ish pairs
        4) Use A* to find a path between each pair.
        5) Flip all the edges along each path.

    Where:
    A bond configuration is an assignment of +/-1 to each bond.
    A flux sector is an assignment of +/-1 to even plaquette and +/-i to odd plaquettes.
    """
    # TODO - this is changing the intial guess as well as outputting the answer - not good if you want to keep the initial guess intact - scope issue
    if target_flux_sector is None:
        target_flux_sector = np.ones(lattice.n_plaquettes, dtype=np.int8)
    if initial_bond_guess is None:
        initial_bond_guess = np.ones(lattice.n_edges, dtype=np.int8)
    initial_bond_guess, target_flux_sector = initial_bond_guess.copy(
    ), target_flux_sector.copy()

    initial_flux_sector = fluxes_from_bonds(lattice, initial_bond_guess)

    # figure out which fluxes need to be flipped
    fluxes_to_flip = target_flux_sector // initial_flux_sector

    # step 2) flip edges that join adjacent -1 fluxes
    bonds, fluxes_to_flip = _flip_adjacent_fluxes(lattice, initial_bond_guess,
                                                  fluxes_to_flip)

    # steps 3) 4) and 5) Use A* to flip the remainging isolated fluxes
    bonds, fluxes_to_flip = _flip_isolated_fluxes(lattice, bonds,
                                                  fluxes_to_flip)

    if np.count_nonzero(fluxes_to_flip == -1) > 1:
        raise ValueError(
            "find_flux_sector failed to get rid of all the -1 fluxes, this is a bug."
        )

    found_fluxes = fluxes_from_bonds(lattice, bonds)

    if np.count_nonzero(found_fluxes - target_flux_sector) > 1:
        raise ValueError(
            "find_flux_sector thought that it worked but somehow still didn't, a bug."
        )

    return bonds.astype(np.int8)
