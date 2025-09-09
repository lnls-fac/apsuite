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


def matrix_from_qf3_to_scrn(kl):
    """Transfer matrix from the start of QF3 to Scren-05 in the LINAC.

    Args:
    kl (float): Integrated quadrupole strength.

    Returns:
        array: Transfer matrix as a NumPy array.
    """
    # this value was taken from the emittance measurement script and matches
    # the .dwg drawing:
    l2 = 2.8775
    # This is the magnet length provided by the LINAC technical design report.
    # It is also being used by the emittance measurement script. However,
    # There is still doubt whether this is the physical or the effective
    # length of the magnet
    quad_len = 0.112
    return matrix_drift(l2) @ matrix_quad(kl) @ matrix_drift(quad_len/2)
