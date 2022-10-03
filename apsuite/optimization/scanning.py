"""Multidimensional Simple Scan method for Minimization."""


from threading import Thread as _Thread
import numpy as _np


class SimpleScan:
    """."""

    def __init__(self):
        """."""
        self._position = _np.array([])
        self._delta = _np.array([])
        self._curr_dim = 0
        self._stop = False

    def set_limits(self, upper=None, lower=None):
        """."""
        self._upper_limits = upper
        self._lower_limits = lower
        self.ndim = len(upper)

    def _optimize(self, npoints):
        """."""
        self._delta = _np.zeros(npoints)
        func = _np.zeros(self._ndim)
        best = _np.zeros(self._ndim)

        for i in range(self._ndim):
            self._delta = _np.linspace(
                self._lower_limits[i], self._upper_limits[i], npoints)
            self._curr_dim = i
            func[i], best[i] = self.calc_obj_fun()
            self._position[i] = best[i]

        print('Best result is: ' + str(best))
        print('Figure of merit is: ' + str(_np.min(func)))
