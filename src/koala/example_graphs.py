from koala.lattice import Lattice
import numpy as np
from koala.voronization import generate_lattice
from koala.graph_utils import rotate

def two_tri():
    vertices = np.array([
        [0.5,0.5],
        [0.5,0.95],
        [0.1,0.1],
        [0.9,0.1]
    ])
    edge_indices = np.array([
        [0,1],
        [0,2],
        [0,3],
        [1,2],
        [2,3]
    ])

    edge_crossing = np.zeros_like(edge_indices)
    lattice = Lattice(vertices,edge_indices,edge_crossing)
    return lattice

def tri_square_pent():
    vertices = np.array([
        [0.18,0.18],
        [0.22,0.48],
        [0.46,0.27],
        [0.45,0.65],
        [0.67,0.43],
        [0.52, 0.88],
        [0.9,0.9],
        [0.92,0.55]
    ])

    edge_indices = np.array([
        [0,1],
        [0,2],
        [1,2],
        [1,3],
        [2,4],
        [3,4],
        [3,5],
        [5,6],
        [6,7],
        [4,7]
    ])

    edge_crossing = np.zeros_like(edge_indices)
    lattice = Lattice(vertices,edge_indices,edge_crossing)
    return lattice

def tutte_graph():
    """
    Returns a tutte graph, a cubic graph with no Hamiltonian cycle, but is three-colorable. 

    Returns:
        vertices: np.array shape (nvertices, ndim) - A list of the positions of all the vertices that make up the graph
        adjacency: np.array shape (nedges, 2) - A list of all edges in the graph, containing the indices of connected vertices
    """
    vertices = np.array([
        [0.518, 0.586], 
        [0.294, 0.986], 
        [0.504, 0.99], 
        [0.69, 0.99], 
        [0.998, 0.616], 
        [0.872, 0.374], 
        [0.746, 0.152], 
        [0.024, 0.558], 
        [0.17, 0.382], 
        [0.334, 0.15], 
        [0.454, 0.54], 
        [0.518, 0.67], 
        [0.592, 0.53], 
        [0.35, 0.548], 
        [0.436, 0.484], 
        [0.342, 0.502], 
        [0.296, 0.478], 
        [0.336, 0.418], 
        [0.408, 0.404], 
        [0.332, 0.93], 
        [0.214, 0.502], 
        [0.138, 0.558], 
        [0.226, 0.43], 
        [0.282, 0.38], 
        [0.368, 0.272], 
        [0.394, 0.822], 
        [0.464, 0.732], 
        [0.638, 0.894], 
        [0.55, 0.734], 
        [0.696, 0.274], 
        [0.62, 0.482], 
        [0.658, 0.55], 
        [0.768, 0.568], 
        [0.906, 0.6], 
        [0.508, 0.774], 
        [0.674, 0.5], 
        [0.508, 0.83], 
        [0.728, 0.482], 
        [0.424, 0.864], 
        [0.556, 0.894], 
        [0.414, 0.922], 
        [0.506, 0.934], 
        [0.784, 0.506], 
        [0.842, 0.482], 
        [0.76, 0.376], 
        [0.824, 0.412]
    ])
    avg_pos = np.sum(vertices, axis = 0)/vertices.shape[0]
    vertices -= avg_pos - np.array([0.5,0.5])

    adjacency = np.array([
        [1 , 11], 
        [1 , 12], 
        [1 , 13], 
        [2 , 3], 
        [2 , 8], 
        [2 , 20], 
        [3 , 4], 
        [3 , 42], 
        [4 , 5], 
        [4 , 28], 
        [ 5 , 6], 
        [ 5 , 34], 
        [ 6 , 7], 
        [ 6 , 46], 
        [ 7 , 10], 
        [ 7 , 30], 
        [ 8 , 9], 
        [ 8 , 22], 
        [ 9 , 10], 
        [ 9 , 23], 
        [ 10 , 25], 
        [ 11 , 14], 
        [ 11 , 15], 
        [ 12 , 27], 
        [ 12 , 29], 
        [ 13 , 31], 
        [ 13 , 32], 
        [ 14 , 16], 
        [ 14 , 22], 
        [ 15 , 16], 
        [ 15 , 19], 
        [ 16 , 17], 
        [ 17 , 18], 
        [ 17 , 21], 
        [ 18 , 19], 
        [ 18 , 24], 
        [ 19 , 25], 
        [ 20 , 26], 
        [ 20 , 41], 
        [ 21 , 22], 
        [ 21 , 23], 
        [ 23 , 24], 
        [ 24 , 25], 
        [ 26 , 27], 
        [ 26 , 39], 
        [ 27 , 35], 
        [ 28 , 29], 
        [ 28 , 40], 
        [ 29 , 35], 
        [ 30 , 31], 
        [ 30 , 45], 
        [ 31 , 36], 
        [ 32 , 33], 
        [ 32 , 36], 
        [ 33 , 34], 
        [ 33 , 43], 
        [ 34 , 44], 
        [ 35 , 37], 
        [ 36 , 38], 
        [ 37 , 39], 
        [ 37 , 40], 
        [ 38 , 43], 
        [ 38 , 45], 
        [ 39 , 41], 
        [ 40 , 42], 
        [ 41 , 42], 
        [ 43 , 44], 
        [ 44 , 46], 
        [ 45 , 46]
    ])

    adjacency -= 1

    edge_crossing = np.zeros_like(adjacency)
    lattice = Lattice(vertices,adjacency,edge_crossing)
    return lattice

def n_ladder(n_sites: int, wobble = False):
    """Produces a strip-ladder type graph - good for testing if things work in strip geometry

    :param n_sites: number of sites in the x-direction
    :type n_sites: int
    :param wobble: adds a sin wobble to the ladder shape - for testing if plaquettes come out that shouldnt be there, defaults to False
    :type wobble: bool, optional
    :return: a lattice for the ladder system
    :rtype: Lattice
    """
    
    x_positions = np.linspace(0.05,0.95,n_sites)
    y_1_positions = 0.3*np.ones(n_sites)
    y_2_positions = 0.7*np.ones(n_sites)

    if wobble == True:
        y_1_positions += 0.1*np.sin(x_positions*2*np.pi)
        y_2_positions += 0.1*np.sin(x_positions*2*np.pi)

    vertices = np.concatenate((np.array([x_positions,y_1_positions]).T,np.array([x_positions,y_2_positions]).T))
    
    edges_bottom = np.array([np.arange(n_sites),(np.arange(n_sites)+1)%n_sites]).T
    crossing_hor = np.zeros_like(edges_bottom)
    crossing_hor[-1,:] = [1,0] 
    edges_top = edges_bottom + n_sites
    edges_across = np.array([np.arange(n_sites),np.arange(n_sites)+n_sites]).T
    crossing_across = np.zeros_like(edges_bottom)

    edges = np.concatenate([edges_bottom,edges_top,edges_across])
    crossing = np.concatenate([crossing_hor,crossing_hor,crossing_across])


    out = Lattice(vertices,edges,crossing)
    return out


def multi_graph() -> Lattice:
    """Returns a graph with multiple edges between the same two sites

    :return: A multiple-y connected graph
    :rtype: Lattice
    """
    return Lattice(
        vertices = np.array([[0.5,0.7], [0.5,0.3]]),
        edge_indices = np.array([[0,1],[0,0],[0,1],[1,1]]),
        edge_crossing = np.array([[0,0],[1,0],[1,0],[1,0]]),
    )


def bridge_graph():
    """gives a simple example of a graph with a bridge - somethiing that could mess up the plaquette finder (but shouldnt any more!)

    :return: A simple lattice with a bridge
    :rtype: Lattice
    """
    vertices = np.array([
        [0.1,0.2],
        [0.1,0.8],
        [0.4,0.5],
        [0.6,0.5],
        [0.9,0.8],
        [0.9,0.2]
    ])

    edges = np.array([
        [0,1],
        [1,2],
        [0,2],
        [2,3],
        [3,4],
        [4,5],
        [5,3]
    ])

    edge_crossing = np.array([[0,0]]*edges.shape[0])

    lat = Lattice(vertices,edges,edge_crossing)

    return lat


def generate_honeycomb(n_horizontal_cells: int, return_coloring = False)-> Lattice:
    """Generates a regular honeycomb lattice with n_horizonta_cells number of cells in the x-direction, and a similar amount in the y direction
    but slightly fudged to fir a square system

    :param n_horizontal_cells: number of cells wide you want the lattice
    :type n_horizontal_cells: int
    :param return_coloring: if True also returns a regular kitaev-style coloring for all the edges, defaults to False
    :type return_coloring: bool, optional
    :return: a lattice or a lattice and a coloring
    :rtype: Lattice, np.ndarray (optional)
    """
    
    # define a basic unit cell with four sites
    unit_cell_dimensions = np.array(
        [1, np.sqrt(3)]
    )

    site_coordinates = np.array([
        [0.25, np.sqrt(3)/12],
        [0.25, 5*np.sqrt(3)/12],
        [0.75, 7*np.sqrt(3)/12],
        [0.75, 11*np.sqrt(3)/12],
    ])

    # find how many unit cells you can fit in the square
    n_vertical = int(np.round(n_horizontal_cells/unit_cell_dimensions[1]))
    h_steps = np.arange(n_horizontal_cells)
    v_steps = np.arange(n_vertical)


    dims_h, dims_y = np.meshgrid(h_steps, v_steps)
    dims_h = dims_h.flatten()
    dims_v = dims_y.flatten()

    # note there is a bit of squishing going on here - hexagons dont tile to a square geometry so we come as close as we can
    # and then squish the y-axis to make up the difference. For n_horizontal_cells > 4 this is barely noticeable.
    full_scale = np.array([
        unit_cell_dimensions[0]*n_horizontal_cells,
        unit_cell_dimensions[1]*n_vertical
    ])
    all_sites = np.concatenate([site_coordinates + np.array([h*unit_cell_dimensions[0] , 0.01+ v*unit_cell_dimensions[1]]) for h,v in zip(dims_h, dims_v)])/full_scale


    # now we add the edges
    internal_ed = np.array([
        [0,1],
        [2,1],
        [2,3]
    ])
    int_edges = np.concatenate([internal_ed + 4*n for n in np.arange(n_vertical*n_horizontal_cells)])

    def next_direction(n_horizontal, n_vertical, n, shift):
        row =  n // n_horizontal
        new_row = (row+shift[1])%n_vertical
        new_column = (n+shift[0])%n_horizontal
        position = new_row*n_horizontal + new_column
        return(position)

    ext_hor = np.array([ [2+4*n , 1+4*next_direction(n_horizontal_cells, n_vertical, n,[1,0])] for n in np.arange(n_horizontal_cells*n_vertical) ])
    ext_ver = np.array([ [ 4*next_direction(n_horizontal_cells, n_vertical, n,[0,1]), 3+4*n] for n in np.arange(n_horizontal_cells*n_vertical) ])
    ext_diag = np.array([ [ 4*next_direction(n_horizontal_cells, n_vertical, n,[1,1]), 3+4*n] for n in np.arange(n_horizontal_cells*n_vertical) ])


    # add the edge_crossings
    crossing_int = np.zeros_like(int_edges)
    crossing_hor = np.zeros_like(ext_hor)
    crossing_hor[np.arange(n_vertical)*n_horizontal_cells +(n_horizontal_cells-1),0] = 1
    crossing_ver = np.zeros_like(ext_ver)
    crossing_ver[np.arange(n_horizontal_cells*(n_vertical-1),n_horizontal_cells*n_vertical) , 1] = -1
    crossing_diag = np.zeros_like(ext_diag)
    crossing_diag[np.arange(n_vertical)*n_horizontal_cells +(n_horizontal_cells-1),0] = -1
    crossing_diag[np.arange(n_horizontal_cells*(n_vertical-1),n_horizontal_cells*n_vertical) , 1] = -1


    edges = np.concatenate([int_edges, ext_hor, ext_ver, ext_diag])
    crossing = np.concatenate([crossing_int, crossing_hor, crossing_ver, crossing_diag])

    coloring = np.array( [0,2,0]*n_vertical*n_horizontal_cells +[1,1] * n_vertical*n_horizontal_cells + [2]* n_vertical*n_horizontal_cells )
    
    if return_coloring:
        return Lattice(all_sites, edges, crossing), coloring
    else:
        return Lattice(all_sites, edges, crossing) 


def generate_hex_square_oct(n_cells: int)-> Lattice:
    """Generates a lattice containing squares, hexagons and octagons

    :param n_cells: Number of unit cells in the x or y directions
    :type n_cells: int
    :return: the lattice object for this system
    :rtype: Lattice
    """


    site_coordinates  = np.array([
        [0.5,0.17],
        [0.2,0.35],
        [0.2,0.65],
        [0.5,0.82],
        [0.8,0.65],
        [0.8,0.35]
    ])

    h_steps = np.arange(n_cells)
    v_steps = np.arange(n_cells)

    dims_h, dims_y = np.meshgrid(h_steps, v_steps)
    dims_h = dims_h.flatten()
    dims_v = dims_y.flatten()

    all_sites = np.concatenate([site_coordinates + np.array([h , v]) for h,v in zip(dims_h, dims_v)]) /n_cells

    internal_ed = np.array([
        [0,1],
        [1,2],
        [2,3],
        [3,4],
        [4,5],
        [5,0]
    ])

    int_edges = np.concatenate([internal_ed + 6*n for n in np.arange(n_cells**2)])
    int_crossing = np.zeros_like(int_edges)

    def next_direction(cells, n, shift):
        row =  n // cells
        new_row = (row+shift[1])%cells
        new_column = (n+shift[0])%cells
        position = new_row*cells + new_column
        return(position)

    ext_hor_1 = np.array([ [4+6*n , 2+6*next_direction(n_cells, n,[1,0])] for n in np.arange(n_cells**2) ])
    crossing_hor_1 = np.zeros_like(ext_hor_1)
    crossing_hor_1[np.arange(n_cells)*n_cells +(n_cells-1),0] = 1
    
    ext_hor_2 = np.array([ [ 1+6*next_direction(n_cells, n,[1,0]), 5+6*n] for n in np.arange(n_cells**2) ])
    crossing_hor_2 = np.zeros_like(ext_hor_2)
    crossing_hor_2[np.arange(n_cells)*n_cells +(n_cells-1),0] = -1
    # ext_ver = np.array([ [ 5+6*next_direction( cells, n,[1,0]), 1+6*n] for n in np.arange(cells**2) ])

    ext_ver = np.array([ [ 6*next_direction(n_cells, n,[0,1]), 3+6*n] for n in np.arange(n_cells**2) ])
    crossing_ver = np.zeros_like(ext_ver)
    crossing_ver[np.arange(n_cells*(n_cells-1),n_cells**2) , 1] = -1


    edges = np.concatenate([int_edges, ext_hor_1,ext_hor_2,ext_ver])
    crossing = np.concatenate([int_crossing, crossing_hor_1,crossing_hor_2, crossing_ver])


    return Lattice(all_sites, edges, crossing)

def generate_tri_non(n_cells, return_coloring = False):
    """generate a lattice of nonagons and triangles

    :param n_cells: system dimensions. If an int is passed then the system is n_cells x n_cells. If an iterable is passed the system is n_cells[0] x n_cells[1]
    :type n_cells: int or iter[int]
    :param return_coloring: optional arg to return a coloring of the edges, defaults to False
    :type return_coloring: bool, optional
    :return: the lattice and optionally a coloring 
    :rtype: Lattice or Lattice, np.ndarray
    """

    unit_points = np.array([
        [0.4,0.1],
        [0.1,0.4],
        [0.4,0.4],
        [0.6,0.6]
    ])

    unit_edges = np.array([
        [0,1],
        [1,2],
        [2,0],
        [3,2],
        [3,0],
        [1,3]
    ])


    unit_crossing = np.array([
        [0,0],
        [0,0],
        [0,0],
        [0,0],
        [0,1],
        [-1,0]
    ])
    lattice = tile_unit_cell(unit_points, unit_edges, unit_crossing,[1.3,1], n_cells)

    try:
        iterator = iter(n_cells)
    except TypeError:
        nx = n_cells
        ny = n_cells
    else:
        nx = n_cells[0]
        ny = n_cells[1]

    coloring = np.array([1,2,0,1,2,0] * nx*ny)

    if return_coloring:
        return lattice, coloring
    else:
        return lattice

# helper function for tile_unit_cell 
def _next_cell_number(n_horizontal, n_vertical, n, shift):
    """gives you the index of the next unit cell in the list of all unit cells

    :param n_horizontal: number unit cells in x direction
    :type n_horizontal: int
    :param n_vertical: number of unit cells in y direction
    :type n_vertical: int
    :param n: what index you are currently at
    :type n: int
    :param shift: whoch way you want to shift to
    :type shift: list
    """
    y =  n // n_horizontal
    new_y = (y+shift[1])%n_vertical
    new_x = (n+shift[0])%n_horizontal
    position = new_y*n_horizontal + new_x
    return(position)


# helper function for tile_unit_cell 
def _crossing(n_x, n_y, n, shift):
    """tells you if the shift crosses PBC in the x or why direction

    :param n_x: system size in x
    :type n_x: int
    :param n_y: system_size in y
    :type n_y: int
    :param n: what index you are currently at
    :type n: int
    :param shift: whoch way you want to shift to
    :type shift: list
    """

    x = n%n_x
    y =  n // n_x
    
    x_new = x + shift[0]
    x_looped = x//n_x != x_new //n_x

    y_new = y + shift[1]
    y_looped = y//n_y != y_new //n_y

    return([shift[0]*x_looped, shift[1]*y_looped])

# TODO - what does unit cell dimensions even do here???
def tile_unit_cell(unit_points: np.ndarray, unit_edges: np.ndarray, unit_crossing: np.ndarray, unit_cell_dimensions:np.ndarray, n_xy) -> Lattice:
    """given a description of a unit cell, this tiles it to make a full lattice

    :param unit_points: coordinates of points in the unit cell
    :type unit_points: np.ndarray
    :param unit_edges: indices of edges in unit cell
    :type unit_edges: np.ndarray
    :param unit_crossing: whether the edges cross to the next unit cell in x and y direction
    :type unit_crossing: np.ndarray
    :param unit_cell_dimensions: size of unit cell
    :type unit_cell_dimensions: np.ndarray
    :param n_xy: number of tilings you want. if this is iterable then assume it has the form [nx, ny]. If an int then we assume we want [n,n]
    :type n_xy: int or list
    :return: the final lattice  
    :rtype: Lattice
    """

    try:
        iterator = iter(n_xy)
    except TypeError:
        nx = n_xy
        ny = n_xy
    else:
        nx = n_xy[0]
        ny = n_xy[1]
    
    x_steps = np.arange(nx)
    y_steps = np.arange(ny)
    x_shifts,y_shifts = np.meshgrid(x_steps, y_steps)
    x_shifts = x_shifts.flatten()
    y_shifts = y_shifts.flatten()

    n_cells = nx*ny

    all_sites = np.concatenate([unit_points + np.array([h , v]) for h,v in zip(x_shifts, y_shifts)])

    all_sites[:,0] = all_sites[:,0]/nx
    all_sites[:,1] = all_sites[:,1]/ny

    n_internal_edges = unit_edges.shape[0]
    n_internal_sites = unit_points.shape[0]
    all_edges = np.zeros((n_cells* n_internal_edges,2), dtype = 'int')
    all_crossings = np.zeros((n_cells* n_internal_edges ,2),dtype = 'int')

    for n_position, xy_pos in enumerate(zip(x_shifts, y_shifts)):
        for n_edge, ec in enumerate(zip(unit_edges, unit_crossing)):
            edge, crossing = ec
            n_index = n_position*n_internal_edges + n_edge

            all_edges[n_index] = [edge[0] + n_position*n_internal_sites, edge[1] + n_internal_sites* _next_cell_number(nx,ny,n_position, crossing)]
            all_crossings[n_index] = _crossing(nx,ny,n_position, crossing)

    return Lattice(all_sites, all_edges, all_crossings)


def single_plaquette(n_sides: int) -> Lattice:
    """Makes a graph consisting of a single plaquette with n_sides sides

    :param n_sides: how many sides the polygon should have
    :type n_sides: int
    :raises Exception: n_sides must be >2
    :return: a lattice containing a single polygon
    :rtype: Lattice
    """

    if(n_sides < 3):
        raise Exception('single_plaquette needs at least three sides')

    angles = np.linspace(0,2*np.pi, n_sides, endpoint=False)
    point_0 = np.array([0.,0.4])

    vertices = np.zeros((n_sides,2))
    for n in range(n_sides):

        vertices[n] = rotate(point_0, angles[n])

    vertices = vertices+0.5

    edges = np.concatenate([np.arange(n_sides)[:,np.newaxis],np.roll(np.arange(n_sides),-1)[:,np.newaxis]], axis= 1)
    crossing = np.zeros_like(edges)

    return Lattice(vertices, edges, crossing)

def ground_state_ansatz(n):
    l = -(-1) ** ((n-3) // 2)
    return l

from koala import voronization, graph_color, flux_finder
from koala.lattice import cut_boundaries

def make_amorphous(L, return_points = False, open_boundary_conditions = False):
    points = np.random.uniform(size=(L**2,2))
    lattice = voronization.generate_lattice(points)
    if open_boundary_conditions: lattice = cut_boundaries(lattice)
    coloring = graph_color.color_lattice(lattice)
    gs_flux_sector = np.array([ground_state_ansatz(p.n_sides) for p in lattice.plaquettes], dtype = np.int8)
    ujk = flux_finder.find_flux_sector(lattice, gs_flux_sector)
    if return_points: return lattice, coloring, ujk, return_points
    return lattice, coloring, ujk

def make_honeycomb(L):
    lattice, coloring = generate_honeycomb(L, return_coloring = True)
    ujk = np.ones(lattice.n_edges, dtype = np.int8)
    return lattice, coloring.astype(np.int8), ujk

def concave_plaquette():
    vertices = np.array([
        [0.1,0.1],
        [0.6,0.9],
        [0.5,0.5],
        [0.9,0.6],
    ])

    edges = np.array([
        [0,1],
        [1,2],
        [2,3],
        [3,0],
    ])

    edge_crossing = np.array([[0,0]]*edges.shape[0])

    lat = Lattice(vertices,edges,edge_crossing)

    return lat