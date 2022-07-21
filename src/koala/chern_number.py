from koala.lattice import Lattice
import numpy as np
import scipy.linalg as la

def crosshair_marker(lattice: Lattice, projector: np.ndarray, crosshair_position: np.ndarray):
    """Generate the crosshair marker for a lattice and Hamiltonian

    :param lattice: the lattice on which the Hamiltonian is placed 
    :type lattice: Lattice
    :param projector: A projectro onto a set of occupied states
    :type projector: np.ndarray
    :param crosshair_position: the position of the crosshair in the bulk
    :type crosshair_position: np.ndarray
    :return: an array giving the marker value at every point in the system
    :rtype: np.ndarray
    """


    positions = lattice.vertices.positions
    theta_x_vec = np.diag(1*(positions[:,0] < crosshair_position[0]))
    theta_y_vec = np.diag(1*(positions[:,1] < crosshair_position[1]))

    crosshair_marker = 4*np.pi*np.diag(projector@theta_x_vec@projector@theta_y_vec@projector).imag

    return crosshair_marker


def chern_marker(lattice: Lattice, projector: np.ndarray):
    """generate the Chern marker for the system

    :param lattice: the lattice 
    :type lattice: Lattice
    :param projector: a projector onto a set of occupied states
    :type projector: np.ndarray
    :return: the Chern marker value for each point in the system
    :rtype: np.ndarray
    """

    positions = lattice.vertices.positions
    X = np.diag(positions[:,0])
    Y = np.diag(positions[:,1])

    chern_marker = 4*np.pi*np.diag(projector@X@projector@Y@projector).imag

    return chern_marker