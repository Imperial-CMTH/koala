############################################################################
#                    Use SAT solvers to color graphs                       #
#                                                                          #
############################################################################
#TODO:
# add the code to fix a specic node to have a specific color
# - refactor so that the two functions share the same main code, this can be done by
# reformulating the edge coloring in terms of an adjacency matrix of edges.
# - make sure that clauses for edges i,j are not being added twice as i,j and j,i
# - test the code that breaks the permutation degeneracy

import numpy as np
from pysat.solvers import Solver
from pysat.card import IDPool, CardEnc, EncType
import itertools as it
from .graph_utils import edge_neighbours
from .lattice import Lattice
from .graph_utils import clockwise_edges_about

# """
# Both routines in this file encode their problems in Conjunctive Normal Form (CNF) and then pass them to
# a SAT solver through pysat (called python-sat in pip).

# ## Encoding
# Conjunctive normal form just means we have a bunch of clauses (i, j, ... k) where each integer represents
# a boolean variable and they're all OR'd together. The clauses are then and together to make a boolean formula:
# f(x_1, x_2, x_3... x_N) = (x_2 OR x_5) AND (-x_1) AND ... AND (x_5 OR -x_4 OR -x_3 OR -x_2 OR -x_1)

# -x_i means NOT(x_i)
# Encoded in python F would just be [(2, 5), (-1), (5, -4, -3, -2, -1)]

# The SAT solver tries to find a solution that assigs a True/False value to each x_i such that f(...) = True

# For both graph and edge coloring we define l_ij == True if vertex/edge i has color j and False otherwise.
# This means we need to use n_colors * n_vertices + 1 integers (0 is not allowed because -0 = 0)

# We then encode two kinds of constraints:

# ### At most one of the colors can be assigned to a vertex
# `cnf = CardEnc.equals(lits=lits, bound = 1, vpool = vpool, encoding=EncType.pairwise)``
# where lits = [l_i1, l_i2, l_i3, ... l_i,n_colors], bound = 1 means exactly 1 of them must be true,
# vpool is an object to make sure we don't reuse the same variable, and encoding chooses on of multiople ways to envode this constraint in CNF

# For instance for three colors [r,g,b], encoding=EncType.pairwise would spits out
# r OR g OR b #ensures one of more is true
# (-r OR -b) AND (-r OR -g) AND (-b OR -g) #ensures no pair is simultanenously true.

# ### No two adjacent vertices can have the same color
# For this one we want that for all edges that connect i,j:
# NOT(l_i1 AND l_i1) AND NOT(l_i2 AND l_i2) AND NOT(l_i3 AND l_i3)

# We use NOT(l_i1 AND l_i1) = (-l_i1 OR -l_i1) to encode this in CNF.

# At the end we get back a solutions that looks like [1, -2, -3, -4, 5, 6...]
# which means vertex 0 is red, vertex 1 is green etc

# """


def vertex_color(adjacency: np.ndarray, n_colors: int = 4, all_solutions=False):
    """Return a coloring of the vertices using n_colors.
    Note the code assumes that all vertices actually appear in the adjacency list
    because it calculates n_vertices = np.max(adjacency) + 1

    Args:
        adjacency (np.ndarray, (... ,2)): A list of pairs of vertex indices representing edges.
        n_colors (int): The maximum number of colors we can use.
        all_solutions (bool, optional): If True, outputs a list of all possible solutions for the coloring. Defaults to False.

    Returns:
        np.ndarray: An integer label for each vertex.
    """

    #graph coloring g_ij where i is node and edge is color
    #two types of contraints:
    #a bunch of atmost1s for each of g_i1 gi2 gi3
    #a bunch of not(g_i1 and g_j1) and not(gi2 and gj2) and not(gi3 and gj3)

    #CNF is an AND of ORs so something like:
    #(a OR b OR c) AND (b or d) AND () AND

    n_vertices = np.max(adjacency) + 1
    n_reserved_literals = n_colors * n_vertices

    #l[i,j] = k where x_k is the variable that tells us if vertex i has color j
    l = np.arange(n_reserved_literals, dtype=int).reshape(n_vertices,
                                                          n_colors) + 1

    #we need to allocate n_reserved_literals to represent our main variables
    #the encoding process might introduce some dummy variables too so we'll make sure they don't overlap with a vpool
    vpool = IDPool(start_from=n_vertices * n_colors)

    #many different solvers can be used
    with Solver(name="g3") as s:
        #constraint: nodes only have one color
        for i in range(n_vertices):
            lits = [int(l[i, j]) for j in range(n_colors)]
            cnf = CardEnc.equals(lits=lits,
                                 bound=1,
                                 vpool=vpool,
                                 encoding=EncType.pairwise)
            s.append_formula(cnf)

        #constraint: nieghbouring nodes are not the same color
        for i, j in adjacency:
            cnf = [[-int(l[i, k]), -int(l[j, k])] for k in range(n_colors)]
            s.append_formula(cnf)

        solveable = s.solve()
        if solveable:
            if all_solutions:
                solutions = np.array(list(s.enum_models())).reshape(
                    (-1, n_vertices, n_colors)).argmax(axis=-1)
                return solveable, solutions
            else:
                solution = np.array(s.get_model()).reshape(
                    n_vertices, n_colors).argmax(axis=-1)
                return solveable, solution

        if not solveable:
            return solveable, s.get_core()


def edge_color(lattice: Lattice,
               n_colors: int = 3,
               all_solutions=False,
               n_solutions=None,
               fixed=()):
    """Return a coloring of the edges using n_colors different labels.

    Args:
        adjacency (np.ndarray): A list of pairs of vertex indices representing edges.
        n_colors (int): The maximum number of colors we can use.
        all_solutions (bool, optional): If True, outputs a list of all possible solutions for the coloring. Defaults to False.
        n_solutions (int, optional): If not None, returns up to this number of solutions. Defaults to None.
        fixed ( list of tuples (label, edge_index), optional): A list of edges indices that are fixed to a particular label. Defaults to [].

    Returns:
        np.ndarray: A label for every edge. if n_solutions or all_solutions are used, gives an array containing all the colorings.
    """

    s = Solver(name="g3")
    adjacency = lattice.edges.indices
    n_edges = adjacency.shape[0]
    n_reserved_literals = n_colors * n_edges

    #define the integer literals we will use
    l = np.arange(n_reserved_literals, dtype=int).reshape(n_edges, n_colors) + 1

    # we need to allocate n_edges*n_colors to represent our main variables
    # the encoding process will introduce some dummy variables too so we'll make
    # sure they don't overlap
    vpool = IDPool(start_from=n_reserved_literals)

    with Solver(name="g3") as s:
        #the first constraint is that each edge had one color
        for i in range(n_edges):
            lits = [int(l[i, j]) for j in range(n_colors)]
            cnf = CardEnc.equals(lits=lits,
                                 bound=1,
                                 vpool=vpool,
                                 encoding=EncType.pairwise)
            s.append_formula(cnf)

        #the second contraint is that nieghbouring nodes are not the same color
        for i in range(n_edges):
            for j in edge_neighbours(lattice, i):
                cnf = [[-int(l[i, k]), -int(l[j, k])] for k in range(n_colors)]
                s.append_formula(cnf)

        #fix any edges to the colors given in fixed
        cnf = [[int(l[edge, color],)] for color, edge in fixed]
        s.append_formula(cnf)

        solveable = s.solve()
        if solveable:
            if all_solutions:
                solutions = np.array(
                    np.array(list(s.enum_models())).reshape(
                        (-1, n_edges, n_colors)).argmax(axis=-1))
                return solveable, solutions

            elif n_solutions is not None:
                models = it.islice(s.enum_models(), n_solutions)
                solutions = np.array(
                    np.array(list(models)).reshape(
                        (-1, n_edges, n_colors)).argmax(axis=-1))
                return solveable, solutions

            else:
                solution = np.array(s.get_model()).reshape(
                    n_edges, n_colors).argmax(axis=-1)
                return solveable, solution

        elif not solveable:
            return solveable, s.get_core()


def color_lattice(lattice: Lattice):
    """Return an asignment of one of 3 labels to each lattice edge such that mp edges with the same label meet at a vertex.
    If such a coloring cannot be found raises a ValueError.

    Args:
        Lattice (lattice): The lattice to color.
    """
    #fix the coloring to be clockwise about vertex 0
    fixed = enumerate(clockwise_edges_about(vertex_index=0, g=lattice))
    solveable, solution = edge_color(lattice, n_colors=3, fixed=fixed)
    if not solveable:
        raise ValueError("No coloring exists for this lattice.")
    return solution.astype(np.int8)
