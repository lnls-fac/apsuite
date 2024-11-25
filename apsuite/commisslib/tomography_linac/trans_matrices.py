"""."""
import numpy as np


def rotation_matrix(theta):
    """."""
    return np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def drift(lenght):
    """."""
    return np.array([[1, lenght], [0, 1]])


def quad(kl):
    """."""
    return np.array([[1, 0], [-kl, 1]])


def to_scrn_matrix(kl):
    """Transfer matrix from the end of QF2-2 to the screen in the LINAC.

    Args:
    kl (float): Integrated quadrupole strength.

    Returns:
    array: Transfer matrix as a NumPy array.
    """
    l1 = 0.1
    l2 = 2.8775
    return drift(l2) @ quad(kl) @ drift(l1)
