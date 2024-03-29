from koala.lattice import Lattice
import numpy as np


def crosshair_marker(lattice: Lattice, projector: np.ndarray,
                     crosshair_position: np.ndarray):
    """Generate the crosshair marker for a lattice and Hamiltonian

    Args:
        lattice (Lattice): the lattice on which the Hamiltonian is
            placed
        projector (np.ndarray): A projectro onto a set of occupied
            states
        crosshair_position (np.ndarray): the position of the crosshair
            in the bulk

    Returns:
        np.ndarray: an array giving the marker value at every point in
        the system
    """

    positions = lattice.vertices.positions
    theta_x_vec = np.diag(1 * (positions[:, 0] < crosshair_position[0]))
    theta_y_vec = np.diag(1 * (positions[:, 1] < crosshair_position[1]))

    c_marker = 4 * np.pi * np.diag(
        projector @ theta_x_vec @ projector @ theta_y_vec @ projector).imag

    return c_marker


def chern_marker(lattice: Lattice, projector: np.ndarray):
    """generate the Chern marker for the system

    Args:
        lattice (Lattice): the lattice
        projector (np.ndarray): a projector onto a set of occupied
            states

    Returns:
        np.ndarray: the Chern marker value for each point in the system
    """

    positions = lattice.vertices.positions
    x = np.diag(positions[:, 0])
    y = np.diag(positions[:, 1])

    c_marker = 4 * np.pi * np.diag(
        projector @ x @ projector @ y @ projector).imag

    return c_marker
