"""."""
import numpy as _np


def rotation_matrix(theta):
    """."""
    return _np.array(
        [[_np.cos(theta), -_np.sin(theta)], [_np.sin(theta), _np.cos(theta)]]
    )


def drift(lenght):
    """."""
    return _np.array([[1, lenght], [0, 1]])


def quad(kl):
    """."""
    return _np.array([[1, 0], [-kl, 1]])


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
