#!/usr/bin/env python-sirius

import numpy as np


'''Multidimensional Simple Scan method for Minimization'''


class SimpleScan:

    def __init__(self):
        """."""
        self._lower_limits = np.array([])
        self._upper_limits = np.array([])
        self._position = np.array([])
        self._delta = np.array([])
        self._curr_dim = 0
        self.initialization()
        self._ndim = len(self._upper_limits)

    def initialization(self):
        """."""
        raise NotImplementedError

    def calc_obj_fun(self):
        """Return arrays with dimension of search space."""
        raise NotImplementedError

    def start_optimization(self, npoints):
        """."""
        self._delta = np.zeros(npoints)
        f = np.zeros(self._ndim)
        best = np.zeros(self._ndim)

        for i in range(self._ndim):
            self._delta = np.linspace(
                                self._lower_limits[i],
                                self._upper_limits[i],
                                npoints)
            self._curr_dim = i
            f[i], best[i] = self.calc_obj_fun()
            self._position[i] = best[i]

        print('Best result is: ' + str(best))
        print('Figure of merit is: ' + str(np.min(f)))
