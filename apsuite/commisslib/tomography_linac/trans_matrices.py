"""."""
import numpy as _np


def matrix_rotation(theta):
    """."""
    return _np.array(
        [[_np.cos(theta), -_np.sin(theta)], [_np.sin(theta), _np.cos(theta)]]
    )


def matrix_drift(lenght):
    """."""
    return _np.array([[1, lenght], [0, 1]])


def matrix_quad(kl):
    """."""
    return _np.array([[1, 0], [-kl, 1]])


def matrix_to_scrn(kl):
    """Transfer matrix from the end of QF2-2 to the screen in the LINAC.

    Args:
    kl (float): Integrated quadrupole strength.

    Returns:
    array: Transfer matrix as a NumPy array.
    """
    l1 = 0.1
    l2 = 2.8775
    return matrix_drift(l2) @ matrix_quad(kl) @ matrix_drift(l1)
