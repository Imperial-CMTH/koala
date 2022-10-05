import numpy as np
import numpy.typing as npt
from dataclasses import dataclass, field
from functools import cached_property
import matplotlib.transforms

INVALID = np.iinfo(int).max


class LatticeException(Exception):
    pass


@dataclass
class Plaquette:
    """Represents a single plaquette in a lattice. Not a list since plaquettes can have varying size.

    Args:
        vertices (np.ndarray[int] (n_sides)): Indices correspondng to
            the vertices that border the plaquette. These are always
            organised to start from the lowest index and then go
            clockwise around the plaquette
        edges (np.ndarray[int] (n_sides)): Indices correspondng to the
            edges that border the plaquette. These are arranged to start
            from the lowest indexed vertex and progress clockwise.
        directions (np.ndarray[int] (n_sides)): Valued +1,-1 depending
            on whether the i'th edge points clockwise/anticlockwise
            around the plaquette
        centers (np.ndarray[float] (2)): Coordinates of the center of
            the plaquette
        n_sides (int): Number of sides to the plaquette
        adjacent_plaquettes (np.ndarray[int] (n_sides)): Indices of all
            the plaquettes that share an edge with this one, ordered in
            the same order as the plaquette edges
    """

    vertices: np.ndarray
    edges: np.ndarray
    directions: np.ndarray
    center: np.ndarray
    n_sides: int
    adjacent_plaquettes: np.ndarray


@dataclass(frozen=True)
class Edges:
    """Represents the list of edges in the lattice

    Args:
        indices (np.ndarray[int] (nedges, 2)): Indices of points
            connected by each edge.
        vectors (np.ndarray[float] (nedges, 2)): Vectors pointing along
            each edge
        crossing (np.ndarray[int] (nedges, 2)): Tells you whether the
            edge crosses the boundary conditions, and if so, in which
            direction. One value for x-direction and one for y-direction
        adjacent_plaquettes (np.ndarray[int] (nedges, 2)): Lists the
            indices of every plaquette that touches each edge
    """

    indices: np.ndarray
    vectors: np.ndarray
    crossing: np.ndarray
    # adjacent_edges: np.ndarray    TODO - add this feature

    # a reference to the parent lattice, has no type because Lattice isn't defined yet
    _parent: ...= field(default=None, repr=False)

    @cached_property
    def adjacent_plaquettes(self) -> np.ndarray:
        self._parent.plaquettes  # access lattice.plaquettes to make them generate
        return self._parent._edges_adjacent_plaquettes


@dataclass(frozen=True)
class Vertices:
    """Represents a list of vertices in the lattice

    Args:
        positions (np.ndarray[float] (nvertices, 2)): List of the
            positions of every vertex
        adjacent_edges (list[np.ndarray] (nvertices, n_edges_per_vertex)):
            Lists the indices of every edge that connects to that
            vertex. Listed in clockwise order from the lowest index
        adjacent_plaquettes (np.ndarray[int] (nvertices, 3)): Lists the
            indices of every plaquette that touches the vertex
    """

    positions: np.ndarray
    adjacent_edges: np.ndarray
    coordination_numbers: np.ndarray
    # adjacent_vertices: np.ndarray    TODO - add this feature

    # a reference to the parent lattice, has no type because the Lattice class isn't defined yet
    _parent: ...= field(default=None, repr=False)

    @cached_property
    def adjacent_plaquettes(self) -> np.ndarray:
        self._parent.plaquettes  # access lattice.plaquettes to make them generate
        return self._parent._vertices_adjacent_plaquettes


class Lattice(object):
    """Data structure containing information about a lattice consisting of vertices in real space connected by undirected edges.

    Args:
        vertices (Vertices): Data structure containing vertex positions,
            and the edges/plaquettes touching each vertex
        edges (Edges): Data structure containing indices of vertices
            comprising each edge, the spatial displacement vectors
            corresponding to those edges, flags for edges which cross
            the system boundary in periodic Lattices, and the plaquettes
            touching each edge.
        plaquettes (list[Plaquette]): All of the polygons (aka
            plaquettes) comprising the lattice, specifying their
            constituent vertices, edges, winding directions, and
            centers.
    """

    def __init__(
            self,
            vertices: npt.NDArray[np.floating],
            edge_indices: npt.NDArray[np.integer],
            edge_crossing: npt.NDArray[np.integer],
            unit_cell=matplotlib.transforms.IdentityTransform(),
    ):
        """Constructor for Lattices

        Args:
            vertices (npt.NDArray[np.floating] Shape (nverts, 2)):
                Spatial locations of lattice vertices
            edge_indices (npt.NDArray[np.integer] Shape (nedges, 2)):
                Indices corresponding to the vertices which each edge
                connects
            edge_crossing (npt.NDArray[np.integer] Shape (nedges, 2)):
                Flags describing which boundaries of the system each
                edge crosses in periodic boundary conditions. Each entry 
                in the final axis corresponds to a spatial dimension, 1(-1) 
                denotes an edge crossing a boundary in the positive (negative) 
                direction along that dimension. 0 corresponds to no boundary 
                crossing.
        """

        # calculate the vector corresponding to each edge
        edge_vectors = (vertices[edge_indices][:, 1] -
                        vertices[edge_indices][:, 0] + edge_crossing)

        # calculate the list of edges adjacent to each vertex
        vertex_adjacent_edges = _sorted_vertex_adjacent_edges(
            vertices, edge_indices, edge_vectors)

        self.vertices = Vertices(
            positions=vertices,
            adjacent_edges=vertex_adjacent_edges,
            coordination_numbers=np.bincount(np.sort(edge_indices.flatten())),
            _parent=self,
        )

        self.edges = Edges(
            indices=edge_indices,
            vectors=edge_vectors,
            crossing=edge_crossing,
            _parent=self,
        )

        # some properties that count edges and vertices etc...
        self.n_vertices = self.vertices.positions.shape[0]
        self.n_edges = self.edges.indices.shape[0]

        self.unit_cell = unit_cell

    def __repr__(self):
        return f"Lattice({self.n_vertices} vertices, {self.n_edges} edges)"

    # find all the plaquettes
    @cached_property
    def plaquettes(self):
        _plaquettes = _find_all_plaquettes(self)

        # now add edge adjacency and point adjacency for plaquettes
        def set_first_invalid(row, value):
            index = np.where(row == INVALID)[0][0]
            row[index] = value

        # arrays that hold neighbouring plaquettes for edges and vertices
        max_coord = np.max(self.vertices.coordination_numbers)
        edges_plaquettes = np.full((self.n_edges, 2), INVALID)
        vertices_plaquettes = np.full((self.n_vertices, max_coord), INVALID)

        # set the values
        for n, plaquette in enumerate(_plaquettes):
            plaq_dir_index = (0.5 * (1 - plaquette.directions)).astype(int)
            edges_plaquettes[plaquette.edges, plaq_dir_index] = n

            x = vertices_plaquettes[plaquette.vertices]
            np.apply_along_axis(set_first_invalid, 1, x, n)
            vertices_plaquettes[plaquette.vertices] = x

        # Later when lattice.edges.adjacent_plaquettes or lattice.vertices.adjacent_plaquettes
        # are accessed, they are copied from __vertices_adjacent_plaquettes and __edges_adjacent_plaquettes
        self._vertices_adjacent_plaquettes = vertices_plaquettes
        self._edges_adjacent_plaquettes = edges_plaquettes

        # set the neighbouring plaquettes for every plaquette - stored in same order as plaquette edges
        for n, plaquette in enumerate(_plaquettes):
            edge_plaquettes = edges_plaquettes[plaquette.edges]
            roll_vals = np.where(edge_plaquettes != n)[1]
            other_plaquettes = edge_plaquettes[np.arange(len(roll_vals)),
                                               roll_vals]
            _plaquettes[n].adjacent_plaquettes = other_plaquettes

        return _plaquettes

    @cached_property
    def n_plaquettes(self):
        return len(self.plaquettes)

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        average_separation = 1 / np.sqrt(self.n_vertices)
        return all([
            np.allclose(
                self.vertices.positions,
                other.vertices.positions,
                atol=average_separation / 100,
            ),
            np.all(self.edges.indices == other.edges.indices),
            np.all(self.edges.crossing == other.edges.crossing),
        ])

    def __ne__(self, other):
        return not self.__eq__(other)

    def __getstate__(self):
        # define a minimal representation for the lattice
        for dtype in [np.uint8, np.uint16, np.uint32, np.uint64]:
            if self.n_vertices <= np.iinfo(dtype).max:
                edges = self.edges.indices.astype(dtype)
                break
        else:
            raise ValueError("A lattice with > 2**64 vertices is just too much")

        vertices = self.vertices.positions.astype(np.float32)

        def check_fits(array, dtype):
            assert (np.iinfo(dtype).min <= np.min(array) and
                    np.max(array) <= np.iinfo(dtype).max)

        check_fits(self.edges.crossing, np.int8)
        crossing = self.edges.crossing.astype(np.int8)

        return vertices, edges, crossing

    def __setstate__(self, state):
        if isinstance(state, dict):
            self.__dict__.update(state)  # For backwards compatibility
        else:  # The new way to pickle just saves vertex positions, edge indices and edge crossing
            self.__init__(*state)


def _sorted_vertex_adjacent_edges(vertex_positions, edge_indices, edge_vectors):
    """Gives you an array where the i'th row contains the indices of the edges that connect to the i'th vertex.
    The edges are always organised in clockwise order starting from 12:00, which will be handy later ;)

    Args:
        vertex_positions (np.ndarray[float] (nvertices, 2)): List of the
            positions of every vertex
        edge_indices (np.ndarray[int] (nedges, 2)): Indices of points
            connected by each edge.
        edge_vectors (np.ndarray[float] (nedges, 2)): Vectors pointing
            along each edge

    Returns:
        list[int] (nvertices, nedges_per_vertex): List containing the
        indices of the edges that connect to each point, ordered
        clockwise around the point.
    """

    # sorts these lists to make sure that they always are ordered clockwise from 12:00 like on a clock face
    vertex_adjacent_edges = []
    for index in range(vertex_positions.shape[0]):
        v_edges = np.nonzero((edge_indices[:, 0] == index) +
                             (edge_indices[:, 1] == index))[0]

        # is the chosen index is first or second in the list
        v_parities = []
        for edge in v_edges:
            par = 1 if (edge_indices[edge][0] == index) else -1
            v_parities.append(par)
        v_parities = np.array(v_parities)

        # find the angle that each vertex comes out at
        v_vectors = edge_vectors[v_edges] * v_parities[:, None]
        v_angles = np.arctan2(-v_vectors[:, 0], v_vectors[:, 1]) % (2 * np.pi)

        # reorder the indices
        # [sorted(-v_angles).index(x) for x in -v_angles]
        order = np.argsort(-v_angles)
        edges_out = v_edges[order]

        vertex_adjacent_edges.append(edges_out)

    return vertex_adjacent_edges


def _find_plaquette(starting_edge: int, starting_direction: int, l: Lattice):
    """Given a single edge, and a direction, this code finds the plaquette corresponding to starting in that
    direction and only taking left turns. This means plaquettes are ordered anticlockwise - which amounts to going round each vertex clockwise.

    Args:
        starting_edge (int): Index of the edge where you start
        starting_direction (int (+1 or -1)): Direction to take the first
            step. +1 means the same direction as the edge, -1 means
            opposite
        l (Lattice): Lattice to be searched for the plaquette

    Returns:
        Plaquette: A plaquette object representing the found plaquette
    """

    edge_indices = l.edges.indices
    vertex_adjacent_edges = l.vertices.adjacent_edges

    s_dir_index = int(0.5 * (1 - starting_direction))
    start_vertex = edge_indices[starting_edge, s_dir_index]
    current_edge = starting_edge
    current_vertex = start_vertex
    current_direction = starting_direction

    plaquette_edges = [starting_edge]
    plaquette_vertices = [start_vertex]
    plaquette_directions = [starting_direction]

    valid_plaquette = True

    while True:

        current_vertex = edge_indices[current_edge][np.where(
            np.roll(edge_indices[current_edge], 1) == current_vertex)[0][0]]
        current_edge_choices = vertex_adjacent_edges[current_vertex]
        current_edge = current_edge_choices[
            (np.where(current_edge_choices == current_edge)[0][0] + 1) %
            current_edge_choices.shape[0]]
        current_direction = (1 if np.where(
            edge_indices[current_edge] == current_vertex)[0][0] == 0 else -1)

        # stop when you get back to where you started
        if current_edge == starting_edge and current_direction == starting_direction:
            break

        # if you get trapped in a loop that doesn't include the start point - stop and return an exception
        edge_dir_bundle = [
            [e, d] for e, d in zip(plaquette_edges, plaquette_directions)
        ]
        cond = [current_edge, current_direction] in edge_dir_bundle[1:]
        if cond:
            # print(current_edge, current_direction, edge_dir_bundle)
            raise LatticeException(
                "plaquette finder is getting stuck. This usually happens if the lattice has self edges or other unexpected properties"
            )

        plaquette_edges.append(current_edge)
        plaquette_vertices.append(current_vertex)
        plaquette_directions.append(current_direction)

    plaquette_edges = np.array(plaquette_edges)
    plaquette_vertices = np.array(plaquette_vertices)
    plaquette_directions = np.array(plaquette_directions)

    # TODO check --- not sure if this is necessary --- check
    # check if the plaquette contains the same edge twice - if this is true then that edge is a bridge
    # this means the plaquette is not legit!
    if len(np.unique(plaquette_edges)) != len(plaquette_edges):
        valid_plaquette = False

    # this bit checks if the loop crosses a PBC boundary once only - if so then it is one of the two edges of a system crossing strip plaquette
    # which means that the system is in strip geometry. We discard the plaquette.
    plaquette_crossings = (plaquette_directions[:, None] *
                           l.edges.crossing[plaquette_edges])
    overall_crossings = np.sum(plaquette_crossings, axis=0)
    if np.sum(overall_crossings != [0, 0]):
        # then this plaquette is invalid
        valid_plaquette = False

    # form the points by adding the edge vectors to the first point - ignores boundary problems
    plaquette_vectors = l.edges.vectors[
        plaquette_edges] * plaquette_directions[:, None]
    plaquette_sums = np.cumsum(plaquette_vectors, 0)
    points = l.vertices.positions[plaquette_vertices[0]] + plaquette_sums
    plaquette_center = np.sum(points, 0) / (points.shape[0]) % 1

    # now we check if the plaquette is acually the boundary of the lattice - this happens when
    # we are in open boundaries, do this by checking the winding number using the outer angles
    # if they go the wrong way round we have an exterior plaquette

    angs = np.arctan2(plaquette_vectors[:, 0], plaquette_vectors[:, 1])
    rel_angs = angs - np.roll(angs, 1)
    ang = np.sum((rel_angs + np.pi) % (2 * np.pi) - np.pi)
    w_number = np.round(ang / (2 * np.pi)).astype("int")

    if w_number != -1:
        valid_plaquette = False

    n_sides = plaquette_edges.shape[0]

    found_plaquette = Plaquette(
        vertices=plaquette_vertices,
        edges=plaquette_edges,
        directions=plaquette_directions,
        center=plaquette_center,
        n_sides=n_sides,
        adjacent_plaquettes=None,
    )

    return found_plaquette, valid_plaquette


def _find_all_plaquettes(l: Lattice):
    """Construct a list of Plaquette objects, representing all of the polygons in the lattice.

    Args:
        l (Lattice): Lattice whose plaquettes are to be found

    Returns:
        list[Plaquette]: List of all plaquettes in the lattice
    """

    edge_indices = l.edges.indices

    # have we already found a plaquette on that edge going in that direction
    edges_fwd_backwd_remaining = np.ones_like(edge_indices)

    plaquettes = []
    for i in range(edge_indices.shape[0]):

        # every edge touches at most two new plaquettes one going forward and one going backwards
        if edges_fwd_backwd_remaining[i, 0] == 1:
            plaq_obj, valid = _find_plaquette(i, 1, l)
            direction_index = (0.5 * (1 - plaq_obj.directions)).astype(int)
            edges_fwd_backwd_remaining[plaq_obj.edges, direction_index] = 0
            if valid:
                plaquettes.append(plaq_obj)

        if edges_fwd_backwd_remaining[i, 1] == 1:
            plaq_obj, valid = _find_plaquette(i, -1, l)
            direction_index = (0.5 * (1 - plaq_obj.directions)).astype(int)
            edges_fwd_backwd_remaining[plaq_obj.edges, direction_index] = 0
            if valid:
                plaquettes.append(plaq_obj)

    return np.array(plaquettes, dtype=object)


def permute_vertices(lattice: Lattice,
                     ordering: npt.NDArray[np.integer]) -> Lattice:
    """Create a new lattice with the vertex indices rearranged according to ordering,
    such that new_l.vertices[i] = l.vertices[ordering[i]].

    Args:
        l (Lattice): Original lattice to have vertices reordered
        ordering (npt.NDArray[np.integer]): Permutation of vertex
            ordering, i = ordering[i']

    Returns:
        Lattice: New lattice object with permuted vertex indices
    """
    original_verts = lattice.vertices
    original_edges = lattice.edges
    nverts = original_verts.positions.shape[0]

    inverse_ordering = np.zeros((nverts,)).astype(int)
    inverse_ordering[ordering] = np.arange(nverts).astype(
        int)  # inverse_ordering[i] = i'

    new_edges = Edges(
        indices=inverse_ordering[original_edges.indices],
        vectors=original_edges.vectors,
        crossing=original_edges.crossing,
        _parent=None,
    )
    new_verts = original_verts.positions[ordering]
    return Lattice(
        vertices=new_verts,
        edge_indices=new_edges.indices,
        edge_crossing=new_edges.crossing,
    )


def cut_boundaries(
    lattice: Lattice, boundary_to_cut: list = (True, True)) -> Lattice:
    """Removes the x and/or y boundary edges of the lattice.

    Args:
        l (Lattice): The lattice to cut.
        boundary_to_cut (list[Bool], optional): whether to cut the x or
            y boundaries, defaults to [True,True]

    Returns:
        Lattice: A new lattice with boundaries cut.
    """
    vertices = lattice.vertices.positions
    edges = lattice.edges.indices
    crossing = lattice.edges.crossing

    x_external = crossing[:, 0] != 0
    y_external = crossing[:, 1] != 0

    condx = 1 - x_external * boundary_to_cut[0]
    condy = 1 - y_external * boundary_to_cut[1]

    cond = condx * condy

    internal_edge_ind = np.nonzero(cond)[0]
    new_edges = edges[internal_edge_ind]
    new_crossing = crossing[internal_edge_ind]

    lattice_out = Lattice(vertices, new_edges, new_crossing)

    return lattice_out
