############################################################################
#              Generic routines for doing things on graphs                 #
#                                                                          #
############################################################################
import numpy as np
from .lattice import Lattice, INVALID
from typing import Tuple
from koala.lattice import Lattice
from pysat.solvers import Solver
from pysat.card import IDPool, CardEnc, EncType
import itertools as it
from .voronization import generate_lattice


def make_dual(lattice: Lattice) -> Lattice:
    """Given a lattice, generate the dual lattice

    Args:
        lattice (Lattice): Input lattice

    Returns:
        Lattice: The dual lattice
    """
    # set vertices of dual lattice at centers of initial lattice
    dual_verts = np.array([p.center for p in lattice.plaquettes])

    # make edges, taking care to remove any edges that connect to the outer boundary of the lattice
    dual_edges = lattice.edges.adjacent_plaquettes
    rows_to_remove = np.where(np.any(dual_edges == INVALID, axis=1))[0]
    cleaned_edges = np.delete(dual_edges, rows_to_remove, axis=0)

    # crossing is sorted out by checking whether the edges cross over half the system
    # TODO - this breaks for small system size - there must be a better solution
    dual_crossing = np.round(dual_verts[cleaned_edges[:, 0]] -
                             dual_verts[cleaned_edges[:, 1]])

    # TODO - currently we just check if it broke and raise an error -- this isnt
    # perfect and something will have to be fixed in the future
    duplicate_edge_crossing = np.concatenate([cleaned_edges, dual_crossing],
                                             axis=1)
    if len(np.unique(duplicate_edge_crossing, axis=0)) != len(cleaned_edges):
        raise Exception(
            "Dual is not currently designed to deal with lattices so small,\
                put more points in the lattice or cut boundaries")

    # make the lattice
    dual = Lattice(dual_verts, cleaned_edges, dual_crossing)
    return dual


# TODO - This doesnt work for very small system sizes, would be worth trying to understand why
def plaquette_spanning_tree(lattice: Lattice, shortest_edges_only=True):
    """Given a lattice this returns a list of edges that form a spanning tree over all the plaquettes (aka a spanning tree of the dual lattice!)
    The optional argument shortest_edge_only automatically sorts the edges to ensure that only the shortest connections are used
    (which is kind of a fudgey way of stopping the algorithm from picking edges that connect over the periodic boundaries). If you're hungry for
    speed you might want to turn it off. The algorith is basically prim's algorithm - so it should run in linear time.

    Args:
        lattice (Lattice): the lattice you want the tree on
        shortest_edges_only (bool, optional): do you want a minimum
            spanning tree - distance wise, defaults to True

    Returns:
        np.ndarray: a list of the edges that form the tree
    """
    plaquettes_in = np.full(lattice.n_plaquettes, -1)
    edges_in = np.full(lattice.n_plaquettes - 1, -1)

    plaquettes_in[0] = 0

    boundary_edges = np.copy(lattice.plaquettes[0].edges)

    for n in range(lattice.n_plaquettes - 1):

        # if we want to keep the edges short - sort the available boundaries
        if shortest_edges_only:

            def find_plaq_distance(edge):
                p1, p2 = lattice.edges.adjacent_plaquettes[edge]
                c1 = 10 if p1 == INVALID else lattice.plaquettes[p1].center
                c2 = 10 if p2 == INVALID else lattice.plaquettes[p2].center
                distance = np.sum((c1 - c2)**2)
                return distance

            distances = np.vectorize(find_plaq_distance)(boundary_edges)
            order = np.argsort(distances)
        else:
            order = np.arange(len(boundary_edges))

        for edge_index in boundary_edges[order]:

            edge_plaq = lattice.edges.adjacent_plaquettes[edge_index]

            if INVALID in edge_plaq:
                continue

            outisde_plaquette_present = [
                x not in plaquettes_in for x in edge_plaq
            ]
            inside_plaquette_present = [x in plaquettes_in for x in edge_plaq]

            # if this edge links an inside and outside plaquette
            if np.any(outisde_plaquette_present) and np.any(
                    inside_plaquette_present):

                # add the new plaquette to the list of inside ones
                position = np.where(outisde_plaquette_present)[0][0]
                new_plaquette = edge_plaq[position]
                plaquettes_in[n + 1] = new_plaquette
                edges_in[n] = edge_index

                # add the new edges to the boundary edges
                boundary_edges = np.append(
                    boundary_edges, lattice.plaquettes[new_plaquette].edges)

                # remove any doubled edges - these will be internal
                a, c = np.unique(boundary_edges, return_counts=True)
                boundary_edges = a[c == 1]

                break

    return edges_in


def remove_vertices(lattice: Lattice,
                    indices: np.ndarray,
                    return_edge_removal=False):
    """Generates a new lattice formed by deleting a subset of vertices from the input lattice

    Args:
        lattice (Lattice): The input latice
        indices (np.ndarray): N array of indices of vertices you want to remove
        return_edge_removal (Bool): If true, returns a list of indices of all the edges that were removed

    Returns:
        Lattice: A new lattice formed from the remaining parts of the input lattice after deletion
    """

    # figure out which edges are set for removal
    set_for_removal = np.full(lattice.n_vertices, False)
    set_for_removal[indices] = True

    # we have to relabel the edge index values after the vertices are removed
    subtraction = np.cumsum(set_for_removal)
    new_index = np.arange(lattice.n_vertices) - subtraction
    new_index[indices] = -1
    new_adjacency = new_index[lattice.edges.indices]

    # remove edges that connect to a deleted vertex
    edges_to_remove = np.where(new_adjacency == -1)[0]
    new_vertices = lattice.vertices.positions[~set_for_removal]
    new_adjacency = np.delete(new_adjacency, edges_to_remove, axis=0)
    new_crossing = np.delete(lattice.edges.crossing, edges_to_remove, axis=0)

    # finally make and return the new lattice
    if return_edge_removal:
        return Lattice(new_vertices, new_adjacency,
                       new_crossing), edges_to_remove
    else:
        return Lattice(new_vertices, new_adjacency, new_crossing)


def remove_trailing_edges(lattice: Lattice) -> Lattice:
    """Trims the trailing edges off a lattice, trailing edges 
    are defined as any edge that does not form part of a plaquette 
    - usually on the boundary of the system after we have used
    cut boundaries to enforce closed boundaries

    Args:
        lattice (Lattice): The lattice to be trimmed

    Returns:
        Lattice: A new lattice with no trailing edges
    """

    # lattice = lattice
    number_of_connections = np.array(
        [len(a) for a in lattice.vertices.adjacent_edges])
    dangling_vertices = np.argwhere(number_of_connections == 1).flatten()

    while (True):
        if len(dangling_vertices) == 0:
            break

        lattice = remove_vertices(lattice, dangling_vertices)
        number_of_connections = np.array(
            [len(a) for a in lattice.vertices.adjacent_edges])
        dangling_vertices = np.argwhere(number_of_connections == 1).flatten()
    return (lattice)


def vertex_neighbours(lattice, vertex_index):
    """Return the neighbouring nodes of a point

    Args:
        lattice (Lattice): a lattice object specifying the system
        adjacency: (M, 2) A list of pairs of vertices representing edges
    Returns:
        vertex_indices: (k), the indices into vertices of the neighbours
        edge_indices: (k), the indices into adjacency of the edges that link vertex_index to its neighbours

    Note that previous version of this function broke the expectation that edge_indices[i] is the edge that links
    vertex_index to vertex_indices[i], make sure to preserve this property.
    """
    adjacency = lattice.edges.indices
    edge_indices = np.where(np.any(vertex_index == adjacency, axis=-1))[0]
    edges = adjacency[edge_indices]
    # print(f"{edges = }, {edge_indices = }")

    #the next two lines replaces the simpler vertex_indices = edges[edges != vertex_i] because the allow points to neighbour themselves
    start_or_end = (
        edges != vertex_index
    )[:,
      1]  #this is true if vertex_index starts the edge and false if it ends it
    vertex_indices = np.take_along_axis(
        edges, start_or_end[:, None].astype(int),
        axis=1).flatten()  #this gets the index of the other end of each edge
    # print(f"{start_or_end = }, {vertex_indices = }")
    #vertex_indices = edges[edges != vertex_i]
    # print(vertex_indices.shape, edge_indices.shape)
    assert vertex_indices.shape == edge_indices.shape
    return vertex_indices, edge_indices


def edge_neighbours(lattice, edge_index):
    """Return the neighbouring edges of an edge (the edges connected to the same nodes as this edge)

    Args:
        lattice (Lattice): The lattice
        edge_index: the index of the edge we want the neighbours of
        edge_i (integer)

    Returns:
        np.ndarray (k,): edge_indices: (k), the indices into adjacency
        of the edges that link vertex_index to its neighbours
    """
    edge = lattice.edges.indices[edge_index]
    v1 = edge[0]
    v2 = edge[1]
    mask = np.any(v1 == lattice.edges.indices, axis=-1) | np.any(
        v2 == lattice.edges.indices, axis=-1)
    mask[edge_index] = False  #not a neighbour of itself
    return np.where(mask)[0]


def clockwise_about(vertex_index: int, g: Lattice) -> np.ndarray:
    """Finds the vertices/edges that border vertex_i, order them clockwise starting from the positive x axis
    and returns those indices in order.

    Args:
        vertex_index (int): int the index into g.vertices.positions of the node we want to use. Generally use 0
        g (Lattice): a graph object with keys vertices, adjacency, adjacency_crossing
    Returns:
        ordered_edge_indices: np.ndarray (n_neighbours_of_vertex_i) ordered indices of the edges.
    """
    #get the edges and vertices around vertex 0
    vertex_indices, edge_indices = vertex_neighbours(g, vertex_index)
    edge_vectors = get_edge_vectors(vertex_index, edge_indices, g)

    #order them clockwise from the positive x axis
    angles = np.arctan2(edge_vectors[:, 1], edge_vectors[:, 0])
    angles = np.where(angles > 0, angles,
                      2 * np.pi + angles)  #move from [-pi, pi] to [0, 2*pi]
    ordering = np.argsort(angles)
    ordered_edge_indices = edge_indices[ordering]
    ordered_vertex_indices = vertex_indices[ordering]
    return ordered_vertex_indices, ordered_edge_indices


def clockwise_edges_about(vertex_index: int, g: Lattice) -> np.ndarray:
    """Finds the edges that border vertex_i, orders them clockwise starting from the positive x axis
    and returns those indices in order. Use this to break the degeneracy of graph coloring.

    Args:
        vertex_index (int): int the index into g.vertices.positions of the node we want to use. Generally use 0
        g (Lattice): a graph object with keys vertices, adjacency, adjacency_crossing
    Returns:
        ordered_edge_indices: np.ndarray (n_neighbours_of_vertex_i) ordered indices of the edges.
    """
    return clockwise_about(vertex_index, g)[1]


def get_edge_vectors(vertex_index: int, edge_indices: np.ndarray,
                     l: Lattice) -> np.ndarray:
    """Get the vector starting from vertex_index along edge_i, taking into account boundary conditions
    Args:
        vertex_index(int): the index of the vertex
        edge_index (int): the index of the edge
        lattice (Lattice): the lattice to use

    Returns:
        np.ndarray (2,):
    """
    #this is a bit nontrivial, g.adjacency_crossing tells us if the edge crossed into another unit cell but
    #it is directional, hence we need to check for each edge if vertex_index was the first of second vertex stored
    #the next few lines do that so we can add g.edges.indices_crossing with the right sign
    edges = l.edges.indices[edge_indices]
    start_or_end = (
        edges != vertex_index
    )[:,
      1]  #this is true if vertex_index starts the edge and false if it ends it
    other_vertex_indices = np.take_along_axis(
        edges, start_or_end[:, None].astype(int),
        axis=1).squeeze()  #this gets the index of the other end of each edge
    offset_sign = (2 * start_or_end - 1)  #now it's +/-1

    #get the vectors along the edges
    return l.vertices.positions[other_vertex_indices] - l.vertices.positions[
        vertex_index][
            None, :] + offset_sign[:, None] * l.edges.crossing[edge_indices]


def adjacent_plaquettes(lattice: Lattice,
                        p_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """For a given lattice, compute the plaquettes that share an edge with lattice.plaquettes[p_index] and the shared edge.
    Returns a list of plaquettes indices and a matching list of edge indices.

    Args:
        l (Lattice): The lattice.
        p_index (int): The index of the plaquette to find the neighbours
            of.

    Returns:
        Tuple[np.ndarray, np.ndarray]: (plaque_indices, edge_indices)
    """
    p = lattice.plaquettes[p_index]
    edges = p.edges
    neighbouring_plaquettes = lattice.edges.adjacent_plaquettes[edges]

    #remove edges that are only part of this plaquette
    valid = ~np.any(neighbouring_plaquettes == INVALID, axis=-1)
    edges, neighbouring_plaquettes = edges[valid], neighbouring_plaquettes[
        valid, :]

    # get just the other plaquette of each set
    p_index_location = neighbouring_plaquettes[:, 1] == p_index
    other_index = 1 - p_index_location.astype(int)[:, None]
    neighbouring_plaquettes = np.take_along_axis(neighbouring_plaquettes,
                                                 other_index,
                                                 axis=1).squeeze(axis=-1)

    return neighbouring_plaquettes, edges


def rotate(vector, angle):
    rm = np.array([[np.cos(angle), -np.sin(angle)],
                   [np.sin(angle), np.cos(angle)]])
    return rm @ vector


def vertices_to_polygon(lattice: Lattice,
                        vertices: np.ndarray = None) -> Lattice:
    """Takes a lattice and turns the vertices in the list vertices into polygons. If vertices is None, all vertices with more than 2 edges will be turned into polygons.

    Args:
        lattice (Lattice): input lattice
        indices (np.ndarray): either an index for the point to be replaced, or a list of indices of points to be replaced

    Returns:
        Lattice: the new lattice with replaced points
    """

    if vertices is None:
        vertices = np.arange(lattice.n_vertices)
    # check if vertices is not iterable
    if not hasattr(vertices, '__iter__'):
        vertices = [vertices]

    new_positions = []
    vertices_shifted = []

    # these will be the remapped edges and crossings of the new lattice
    original_edges = np.full([lattice.n_edges, 2], None)
    original_crossing = lattice.edges.crossing.copy()

    # these will be the new edges and crossings of the new lattice
    added_edges = []
    added_crossing = []

    running_total = 0
    # find the new positions of the vertices
    for n in range(lattice.n_vertices):

        # if the vertex is in the list of vertices to be turned into polygons
        if n in vertices and len(lattice.vertices.adjacent_edges[n]) > 2:

            # find the edges+vertices that are adjacent to this vertex
            edges_from = lattice.vertices.adjacent_edges[n]
            other_vertices = lattice.edges.indices[edges_from]
            first_or_second = np.where(other_vertices != n)[1]

            # create new vertex positions
            vectors = -(1 - 2 * first_or_second[:, np.newaxis]
                       ) * lattice.edges.vectors[edges_from]
            new_set = lattice.vertices.positions[n] + vectors / 3

            # check if the new vertex positions are in the unit cell, if not, shift them
            shifted = (new_set // 1).astype("int")
            new_set = new_set % 1

            # now add the polygon edges
            around_polygon = np.array([
                [x, (x + 1) % len(new_set)] for x in range(len(new_set))
            ]) + running_total
            crossing_around = np.zeros_like(around_polygon)

            #  iterate over the new vertices
            for u in range(len(new_set)):

                current_index = running_total + u

                # shift the crossings if the vertices were shifted
                if np.any(shifted[u]):
                    original_crossing[edges_from[u]] += shifted[u] * (
                        1 - 2 * first_or_second[u])

                    crossing_around[u] -= shifted[u]
                    crossing_around[(u - 1) % len(new_set)] += shifted[u]

                new_positions.append(new_set[u])
                vertices_shifted.append(shifted[u])
                added_edges.append(around_polygon[u])
                added_crossing.append(crossing_around[u])

                original_edges[edges_from[u],
                               1 - first_or_second[u]] = current_index

            running_total += len(new_set)

        # else just keep on as normal
        else:
            # trivially add the vertex
            new_positions.append(lattice.vertices.positions[n])
            vertices_shifted.append(np.zeros(2))

            # add the edges that touch this vertex
            edges_from = lattice.vertices.adjacent_edges[n]
            other_vertices = lattice.edges.indices[edges_from]
            first_or_second = np.where(other_vertices != n)[1]
            original_edges[edges_from, 1 - first_or_second] = running_total

            running_total += 1

    added_edges = np.array(added_edges)
    added_crossing = np.array(added_crossing)

    vertices = np.array(new_positions)
    edges = np.concatenate([original_edges, added_edges])
    crossing = np.concatenate([original_crossing, added_crossing])

    return Lattice(vertices, edges.astype("int"), crossing.astype("int"))

def dimerise(lattice: Lattice, n_solutions=1):
    """Given a lattice, this finds one or many valid dimerisations of this lattice, 
    using SAT solvers. Output is given by a list of 0 or 1 for each edge, indicating 
    whether that edge forms a dimer or not.

    Args:
        lattice (Lattice): input lattice
        n_solutions (int, optional): How many solutions you want to output. Defaults to 1.

    Raises:
        ValueError: If the lattice has no valid dimers

    Returns:
        np.ndarray : An array of size (n_solutions, n_edges) indicating which bonds are in the dimerisation. 
    """

    n_reserved_literals = lattice.n_edges
    vpool = IDPool(start_from=n_reserved_literals)

    with Solver(name="g3") as s:

        for i in range(lattice.n_vertices):
            # the only constraint is that every vertex touches exactly one dimer
            lits = [int(u) for u in lattice.vertices.adjacent_edges[i] + 1]
            cnf = CardEnc.equals(lits=lits,
                                 bound=1,
                                 vpool=vpool,
                                 encoding=EncType.pairwise)
            s.append_formula(cnf)

        # solve the SAT problem and return the solutions
        solveable = s.solve()

        if solveable:
            if n_solutions == 1:
                model = it.islice(s.enum_models(), 1)
                solutions = np.sign(np.array(list(model))[0])
                return (solutions + 1) // 2

            if n_solutions is not None:
                models = it.islice(s.enum_models(), n_solutions)
                solutions = np.sign(np.array(list(models)))
                return (solutions + 1) // 2

        else:
            raise ValueError("No dimerisation exists for this lattice.")


def lloyd_relaxation(lattice: Lattice, n_steps: int = 5):
    """Performs n_steps iterations of Lloyd's algorithm, which serves to make an amorphous lattice somewhat more regular,
    with more normalised bond lengths. Generally, around 5 iterations is optimal for a decent convergence.

    Args:
        lattice (Lattice): The lattice we start with
        n_steps (int): How many steps the algorithm should take to relax the lattice

    Returns:
        Lattice: The relaxed lattice
    """
    for x in range(n_steps):
        new_centers = np.array([p.center for p in lattice.plaquettes])
        lattice = generate_lattice(new_centers)
    return lattice
