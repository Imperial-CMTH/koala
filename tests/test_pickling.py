from koala import example_graphs as eg
import pickle
from pathlib import Path
from koala.graph_utils import cut_boundaries
test_data_dir = Path(__file__).parent / "data"

def test_backwards_compatability():
    with open(test_data_dir / "pickled_lattice_V0.pickle", "rb") as f:
        lattice_V0 = pickle.load(f)

    with open(test_data_dir / "pickled_lattice_V1.pickle", "rb") as f:
        lattice_V1 = pickle.load(f)
        
    assert lattice_V0 == lattice_V1
    assert [p.n_sides for p in lattice_V0.plaquettes] == [p.n_sides for p in lattice_V1.plaquettes]
    
def test_round_trip():
    lattice, _, _ = eg.make_amorphous(10)
    string = pickle.dumps(lattice)
    new_lattice = pickle.loads(string)
    # print(lattice == new_lattice)
    assert lattice == new_lattice
    assert [p.n_sides for p in lattice.plaquettes] == [p.n_sides for p in new_lattice.plaquettes]

def test_round_trip_open_boundaries():
    #now check with cut boundaries
    lattice, _, _ = eg.make_amorphous(10)
    lattice = cut_boundaries(lattice)
    string = pickle.dumps(lattice)
    new_lattice = pickle.loads(string)
    # print(lattice == new_lattice)
    assert lattice == new_lattice
    assert [p.n_sides for p in lattice.plaquettes] == [p.n_sides for p in new_lattice.plaquettes]