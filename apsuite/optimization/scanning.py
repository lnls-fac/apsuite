#!/usr/bin/env python-sirius
"""Multidimensional Simple Scan method for Minimization."""


from threading import Thread as _Thread
import numpy as _np


class SimpleScan:
    """."""

    @property
    def ndim(self):
        """."""
        return self._ndim

    @ndim.setter
    def ndim(self, value):
        """."""
        self._ndim = value

    @property
    def position(self):
        """."""
        return self._position

    @position.setter
    def position(self, value):
        """."""
        self._position = value

    def __init__(self):
        """."""
        self._lower_limits = _np.array([])
        self._upper_limits = _np.array([])
        self._ndim = 0
        self._position = _np.array([])
        self._delta = _np.array([])
        self._curr_dim = 0
        self._stop = False
        self._thread = _Thread(target=self._optimize, daemon=True)
        self.initialization()

    def initialization(self):
        """."""
        raise NotImplementedError

    def calc_obj_fun(self):
        """Return arrays with dimension of search space."""
        raise NotImplementedError

    def set_limits(self, upper=None, lower=None):
        """."""
        self._upper_limits = upper
        self._lower_limits = lower
        self.ndim = len(upper)

    def start(self):
        """."""
        if not self._thread.is_alive():
            self._stop = False
            self._thread = _Thread(target=self._optimize, daemon=True)
            self._thread.start()

    def stop(self):
        """."""
        self._stop = True

    @property
    def isrunning(self):
        """."""
        return self._thread.is_alive()

    def _optimize(self, npoints):
        """."""
        self._delta = _np.zeros(npoints)
        func = _np.zeros(self._ndim)
        best = _np.zeros(self._ndim)

        for i in range(self._ndim):
            self._delta = _np.linspace(
                                self._lower_limits[i],
                                self._upper_limits[i],
                                npoints)
            self._curr_dim = i
            func[i], best[i] = self.calc_obj_fun()
            self._position[i] = best[i]

        print('Best result is: ' + str(best))
        print('Figure of merit is: ' + str(_np.min(func)))
