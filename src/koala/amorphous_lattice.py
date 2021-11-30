##############################
###    Work in progress    ###
##############################

import tinyarray as ta
import kwant

class AmorphousLattice(kwant.builder.SiteFamily):
    """
    A SiteFamily to represent an Amorphous Lattice
    instantiate with the number of orbitals per lattice site and the positions of each site
    
    ```
    num_sites = 100; dimension = 2
    a_lat = AmorphousLattice(norbs = 1, positions = np.random.random(size = [num_sites, dimension]))
    ```
    """
    def __init__(self, norbs, position_list):
        self.position_list = position_list
        super().__init__(canonical_repr = f"amorphous_lattice_with_{norbs}_orbs", name = "Amorphous Lattice", norbs = norbs)
    
    def __str__(self): return f"Amorphous Lattice with {self.norbs} orbitals"
    
    def normalize_tag(self, tag):
        tag = ta.array(tag, int)
        if len(tag) != 1: raise ValueError("Dimensionality mismatch.")
        return tag
    
#     def positions(self, tags):
#         """Return the real-space positions of the sites with the given tags."""
#         return self.position_list[[t[0] for t in tags], :]

    def pos(self, tag):
        """Return the real-space position of the site with a given tag."""
        return self.position_list[tag][0]