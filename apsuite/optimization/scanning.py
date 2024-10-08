"""Multidimensional Simple Scan method for Minimization."""

import numpy as _np

from .base import OptimizeParams as _OptimizeParams, Optimize as _Optimize


class SimpleScanParams(_OptimizeParams):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.number_of_steps = 10

    def __str__(self):
        """."""
        stg = self._TMPD('number_of_steps', self.number_of_steps, '')
        stg += super().__str__(self)
        return stg


class SimpleScan(_Optimize):
    """."""

    def __init__(self, use_thread=True):
        """."""
        super().__init__(
            self, params=SimpleScanParams(), use_thread=use_thread)
        self.params = SimpleScanParams()

    def set_limits(self, upper=None, lower=None):
        """."""
        self._upper_limits = upper
        self._lower_limits = lower

    def _optimize(self):
        """."""
        num_pts = self.params.number_of_steps
        num_dims = self.params.initial_position.shape[-1]
        shape = num_dims * (num_pts, )
        size = num_pts**num_dims

        low = self.params.limit_lower
        high = self.params.limit_upper
        delta = high - low

        self.objfuncs_best = _np.zeros(size, dtype=float)
        self.positions_best = _np.zeros((size, num_dims), dtype=float)
        for i in range(size):
            ivec = _np.unravel_index(i, shape)
            pos = low + (delta * ivec)/(num_pts - 1)
            self.positions_best[i] = pos
            self.objfuncs_best[i] = self._objective_func(pos)
