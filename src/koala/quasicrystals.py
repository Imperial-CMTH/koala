import numpy as np
from scipy import linalg as la
from koala.lattice import Lattice
from koala.graph_utils import make_dual, remove_vertices, remove_trailing_edges


def penrose_tiling(number_of_db_lines: int) -> Lattice:
    """generates a random slice of penrose tiling, size determined by n_db_lines

    Args:
        n_db_lines (int): how many de brujin lines are used, acts to increase size of the tiling

    Returns:
        Lattice: a penrose tiling lattice object
    """

    number_of_bundles = 5
    grid_offsets = np.random.random(number_of_bundles) - 0.5
    grid_offsets = grid_offsets - (np.sum(grid_offsets) - 1) / number_of_bundles
    grid_offsets = (grid_offsets + 0.5) % 1 - 0.5

    lattice = de_brujin_grid(number_of_db_lines, number_of_bundles,
                             grid_offsets)

    return lattice

def random_offsets(number_of_bundles:int):
    """Generate random offsets for the de brujin grid

    Args:
        number_of_bundles (int): how many bundles are used, 5 generates Penrose tiling

    Returns:
        np.array: the offsets for each bundle
    """
    grid_offsets = np.random.random(number_of_bundles) - 0.5
    grid_offsets = grid_offsets - (np.sum(grid_offsets) - 1) / number_of_bundles
    grid_offsets = (grid_offsets + 0.5) % 1 - 0.5
    return grid_offsets

def de_brujin_grid(number_of_lines: int,
                   number_of_bundles=5,
                   grid_offsets=None,
                   angle_disorder=0) -> Lattice:
    """Generate a quasiperiodic lattice 

    Args:
        number_of_lines (_type_): how many de brujin lines are used, acts to increase size of the tiling
        number_of_bundles (int, optional): how many bundles are used, 5 generates Penrose tiling. Defaults to 5.
        grid_offsets (_type_, optional): the offset of each grid, if none then we set to 0.2. Defaults to None.
        angle_disorder (int, optional): adds a random angle to the gradient of each grid. Defaults to 0.

    Returns:
        Lattice: The generated quasiperiodic tiling
    """
    if grid_offsets is None:
        grid_offsets = np.full(number_of_bundles, 0.2)
    elif isinstance(grid_offsets, (float,int)):
        grid_offsets = np.full(number_of_bundles, grid_offsets)


    # find the angles and directions of each line bundle
    angles = np.arange(number_of_bundles) * (
        2 * np.pi / number_of_bundles) + angle_disorder * (
            np.random.random(number_of_bundles) - 0.5) * 2 * np.pi
    gradients = np.array([[np.cos(a), np.sin(a)] for a in angles])
    normals = np.array(
        [[np.cos(a + np.pi / 2), np.sin(a + np.pi / 2)] for a in angles])

    # find the offset for each line in the bundle
    line_offsets = np.arange(number_of_lines) - (number_of_lines - 1) // 2
    total_offsets = line_offsets[np.newaxis, :] + grid_offsets[:, np.newaxis]

    # each line is calculated using a vector equation and these store the c intercept for each line
    c_values = total_offsets[:, :, np.newaxis] * normals[:, np.newaxis, :]

    # how many vertices we expect to be in the system
    number_of_vertices = (number_of_lines * number_of_lines *
                          (number_of_bundles - 1) * number_of_bundles) // 2

    # find the vertices
    all_vertices = np.zeros([number_of_vertices, 2])  # positions
    all_bundle_intersections = np.zeros(
        [number_of_vertices, 2])  # index of the bundles that make the vertex
    all_line_intersections = np.zeros(
        [number_of_vertices, 2])  # index of the line for this intersection
    all_line_positions = np.zeros(
        [number_of_vertices,
         2])  # value of \mu parameter on each line for the intersection

    count = 0
    # for each pair of bundles
    for b1 in range(number_of_bundles):
        for b2 in range(b1 + 1, number_of_bundles):

            m1 = gradients[b1]
            m2 = gradients[b2]
            M = np.array([m1, -m2]).T
            inv = la.inv(M)

            # for each pair of lines find the intersection
            for l1 in range(number_of_lines):
                for l2 in range(number_of_lines):

                    c = c_values[b2, l2] - c_values[b1, l1]
                    nu = inv @ c
                    pos1 = m1 * nu[0] + c_values[b1, l1]
                    # pos2 = m2*nu[1] + c_values[b2,l2]

                    all_vertices[count] = pos1
                    all_bundle_intersections[count] = np.array([b1, b2])
                    all_line_intersections[count] = np.array([l1, l2])
                    all_line_positions[count] = nu
                    count += 1

    # find the edges
    number_edges = number_of_bundles * number_of_lines * (
        (number_of_bundles - 1) * number_of_lines - 1)

    # indices of all the edges
    all_edges = np.full([number_edges, 2], -1)

    # label which bundle each edge is in
    edge_bundles = np.full([number_edges], -1)

    for b in range(number_of_bundles):

        for l_num in range(number_of_lines):
            insertion_point = np.nonzero(all_edges == -1)[0][0]
            vi = np.where(
                (all_line_intersections == l_num) * (all_bundle_intersections == b))
            vertex_number = vi[0]

            nu_values = all_line_positions[vi]
            order = np.argsort(nu_values)

            new_edges = np.concatenate([
                vertex_number[order[:-1], np.newaxis], vertex_number[order[1:],
                                                                     np.newaxis]
            ],
                                       axis=1)
            edge_bundles[insertion_point:insertion_point +
                         new_edges.shape[0]] = b
            all_edges[insertion_point:insertion_point +
                      new_edges.shape[0]] = new_edges

    # scale this to live in the [0,1] range and turn it into a lattice object
    furthest_point = np.max(all_vertices)
    scaling = 0.98 / (2 * furthest_point)
    scaled_verts = all_vertices * scaling + 0.5
    lattice = Lattice(scaled_verts, all_edges, all_edges * 0)

    # make the dual lattice
    dual = make_dual(lattice, True)

    # remove all the the lattice outside a boundary - get rid of the star otside bit
    # vertex_radii = np.sqrt(np.sum((dual.vertices.positions - 0.5)**2, axis=1))
    # radius = scaling * (number_of_lines - 1) / 2
    # remove = np.where(vertex_radii > radius)[0]
    # dual_clipped = remove_trailing_edges(remove_vertices(dual, remove))

    # new_gradients = scaling*gradients
    # new_normals = scaling*normals
    new_c_values = scaling * c_values + 0.5
    starting_positions = new_c_values[:, 0, :]

    def find_pent_index(point):
        shifted_vals = point[np.newaxis, :] - starting_positions
        distance = np.sum(shifted_vals * normals, axis=1) / scaling
        return distance // 1

    def map_to_position(index):
        x = np.sum(index * np.cos(angles))
        y = np.sum(index * np.sin(angles))
        return np.array([x, y])

    # remove all the the lattice outside the pentagon
    all_indices = np.apply_along_axis(find_pent_index, 1, dual.vertices.positions)
    mask = np.any((all_indices < 0) + (all_indices >= number_of_lines),1)
    remove = np.where(mask)
    dual_clipped = remove_trailing_edges(remove_vertices(dual, remove))

    new_points = np.zeros_like(dual_clipped.vertices.positions)
    for n, point in enumerate(dual_clipped.vertices.positions):
        new_points[n] = map_to_position(find_pent_index(point))

    scale2 = 0.9 / (2 * np.max(np.abs(new_points)))
    new_points = new_points * scale2 + 0.5

    final_lattice = Lattice(new_points, dual_clipped.edges.indices,
                            dual_clipped.edges.crossing)

    return final_lattice
