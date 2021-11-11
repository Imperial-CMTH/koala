from koala.amorphous_lattice import AmorphousLattice
import numpy as np
import kwant

def test_creation():
    #setup
    num_sites = 10; dimension = 2
    position_list = np.random.random(size = [num_sites, dimension])

    #actions
    a_lat = AmorphousLattice(norbs = 1, position_list = position_list)
    a_syst = kwant.Builder()
    a_syst[[a_lat(i) for i in range(num_sites)]] = 1

    a_syst[a_lat(0), a_lat(1)] = 1

    #assert( np.allclose(a_lat.pos((0,)), position_list[0]))
    
    #assert(np.allclose(a_lat.positions([(0,), (1,)]), position_list[(0,1), :]))