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
        stg = self._TMPD.format('number_of_steps', self.number_of_steps)
        stg += super().__str__(self)
        return stg

    def to_dict(self):
        """."""
        dic = super().to_dict()
        dic['number_of_steps'] = self.number_of_steps

    def from_dict(self, dic):
        """."""
        super().from_dict(dic)
        self.number_of_steps = dic.get('number_of_steps', self.number_of_steps)


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

        self.best_objfuncs = _np.zeros(size, dtype=float)
        self.best_positions = _np.zeros((size, num_dims), dtype=float)
        for i in range(size):
            ivec = _np.unravel_index(i, shape)
            pos = low + (delta * ivec)/(num_pts - 1)
            self.best_positions[i] = pos
            self.best_objfuncs[i] = self._objective_func(pos)
