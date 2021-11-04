#graph colouring g_ij where i is node and edge is color
#two types of contraints:
    #a bunch of atmost1s for each of g_i1 gi2 gi3
    #a bunch of not(g_i1 and g_j1) and not(gi2 and gj2) and not(gi3 and gj3)

#CNF is an AND of ORs so something like:
#(a OR b OR c) AND (b or d) AND () AND
import pysat
import numpy as np
#pysat.params['data_dirs'] = '/Users/tom/pysat' ???
from pysat.solvers import Solver
from pysat.card import *

def vertex_color(vertices, adjacency, n_colors):
    #graph colouring g_ij where i is node and edge is color
    #two types of contraints:
        #a bunch of atmost1s for each of g_i1 gi2 gi3
        #a bunch of not(g_i1 and g_j1) and not(gi2 and gj2) and not(gi3 and gj3)

    #CNF is an AND of ORs so something like:
    #(a OR b OR c) AND (b or d) AND () AND
        
    n_vertices = len(vertices)
    n_reserved_literals = n_colors * n_vertices
    l = np.arange(n_reserved_literals, dtype = int).reshape(n_vertices, n_colors) + 1
    s = Solver(name='g3')

    # encoding will be:
    # g[i,j] == 1 means vertex i has color j and 0 means it doesn't have that color

    # and we have a mapping from i,j to literals which is
    #l[i,j] = n_colors * i + j

    #we need to allocate n_vertices*n_colors to represent our main variables
    #the encoding process will introduce some dummy variables too so we'll make sure they don't overlap
    vpool = IDPool(start_from=n_vertices * n_colors)

    #the first constraint is that at most one of [gi0, gi1, gi2... gin_colors] is true
    #classmethod atleast(lits, bound=1, top_id=None, vpool=None, encoding=1)
    for i in range(n_vertices):
        lits = list(map(int, [l[i,0], l[i,1], l[i,2],]))
        cnf = CardEnc.equals(lits=lits, bound = 1, vpool = vpool, encoding=EncType.pairwise)
        s.append_formula(cnf)
        
    #the second contraint is that niehgbouring nodes are not the same color
    #a bunch of not(g_i1 and g_j1) and not(gi2 and gj2) and not(gi3 and gj3)

    #de morgans says not(g_i1 and g_j1) <=> (not(g_i1) or not(g_j1))
    #so the second statement becomes
    # (-gi1 OR -gj2) AND (-gi2 OR -gj2) AND (-gi3 OR -gj3)
    for i,j in adjacency:
        cnf = [[-int(l[i,k]), -int(l[j,k])] for k in range(n_colors)]
        
        s.append_formula(cnf)


    s.solve()
    s.get_model()
    solution = np.array(s.get_model()).reshape(n_vertices, n_colors).argmax(axis = -1)
    return solution

def edge_color(adjacency, n_colors):
    s = Solver(name='g3')
    n_edges = adjacency.shape[0]
    n_reserved_literals = n_colors * n_edges

    #define the integer literals we will use
    l = np.arange(n_reserved_literals, dtype = int).reshape(n_edges, n_colors) + 1

    # encoding will be:
    # g[i,j] == 1 means edge i has color j and 0 means it doesn't have that color

    # and we have a mapping from i,j to literals which is
    #l[i,j] = n_colors * i + j

    #we need to allocate n_edges*n_colors to represent our main variables
    #the encoding process will introduce some dummy variables too so we'll make sure they don't overlap
    vpool = IDPool(start_from=n_reserved_literals)

    #the first constraint is that at most one of [gi0, gi1, gi2... gin_colors] is true
    #classmethod atleast(lits, bound=1, top_id=None, vpool=None, encoding=1)
    for i in range(n_edges):
        lits = list(map(int, [l[i,0], l[i,1], l[i,2],]))
        cnf = CardEnc.equals(lits=lits, bound = 1, vpool = vpool, encoding=EncType.pairwise)
        #cnf = CardEnc.equals(lits=[int(g[i,j]) for j in range(n_colors)], bound = 1, vpool = vpool, encoding=EncType.pairwise)
        s.append_formula(cnf)
        
    #the second contraint is that niehgbouring nodes are not the same color
    #a bunch of not(g_i1 and g_j1) and not(gi2 and gj2) and not(gi3 and gj3)

    #de morgans says not(g_i1 and g_j1) <=> (not(g_i1) or not(g_j1))
    #so the second statement becomes
    # (-gi1 OR -gj2) AND (-gi2 OR -gj2) AND (-gi3 OR -gj3)

    def neighbours(edge_i, adjacency):
        edge = adjacency[edge_i]
        v1 = edge[0]
        v2 = edge[1]
        mask = np.any(v1 == adjacency, axis = -1) | np.any(v2 == adjacency, axis=-1)
        mask[edge_i] = False #not a neighbour of itself
        return np.where(mask)[0]

    for i in range(n_edges):
        for j in neighbours(i, adjacency):
            #cnf = [[-l[i,0], -l[j,0]], [-l[i,1], -l[j,1]], [-l[i,2], -l[j,2]]]
            cnf = [[-int(l[i,k]), -int(l[j,k])] for k in range(3)]

            s.append_formula(cnf)


    s.solve()
    s.get_model()
    solution = np.array(s.get_model()).reshape(n_edges, n_colors).argmax(axis = -1)
    return solution

#examples!
# n = 1000

# from topamorph.pointsets import generate_random
# points = generate_random(n)

# from topamorph.voronization import generate_pbc_voronoi_adjacency, cut_boundaries
# vertices, adjacency = generate_pbc_voronoi_adjacency(points)

# from topamorph.plotting import plot_lattice
# colors = np.random.choice(np.array(['red', 'green', 'blue']), size = vertices.shape[0])

# from topamorph.SAT import vertex_color
# solution = vertex_color(vertices, adjacency, n_colors = 3)
# colors = np.array(['orange', 'b', 'k'])[solution]

# plot_lattice(vertices, adjacency, scatter_args = dict(color = colors))

# from topamorph.SAT import edge_color
# solution = edge_color(adjacency, n_colors)
# colors = np.array(['orange', 'b', 'k'])[solution]

# plot_lattice(vertices, adjacency, edge_colors = colors)