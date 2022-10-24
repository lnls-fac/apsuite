"""Simulated Annealing Algorithm for Minimization."""
from threading import Thread as _Thread, Event as _Event
import logging as _log

import numpy as _np

from mathphys.functions import get_namedtuple as _get_namedtuple


class OptimizeParams:
    """."""

    _TMPD = '{:30s}: {:10d}\n'
    _TMPF = '{:30s}: {:10.3f}\n'
    _TMPS = '{:30s}: {:10s}\n'

    BoundaryPolicy = _get_namedtuple('BoundaryPolicy', ('ToBoundary', 'ToNaN'))

    def __init__(self):
        """."""
        self._initial_position = _np.array([])
        self._limit_upper = _np.array([])
        self._limit_lower = _np.array([])
        self.max_number_iters = 100
        self.max_number_evals = 1000
        self._boundary_policy = self.BoundaryPolicy.ToNaN

    @property
    def boundary_policy_str(self):
        """."""
        return self.BoundaryPolicy._fields[self._boundary_policy]

    @property
    def boundary_policy(self):
        """."""
        return self._boundary_policy

    @boundary_policy.setter
    def boundary_policy(self, val):
        """."""
        if isinstance(val, str) and val in self.BoundaryPolicy._fields:
            self._boundary_policy = self.BoundaryPolicy._fields.index(val)
        elif int(val) in self.BoundaryPolicy:
            self._boundary_policy = int(val)

    @property
    def initial_position(self):
        """."""
        return self._initial_position

    @initial_position.setter
    def initial_position(self, arr):
        """."""
        self._initial_position = _np.array(arr, ndmin=1)

    @property
    def limit_upper(self):
        """."""
        return self._limit_upper

    @limit_upper.setter
    def limit_upper(self, arr):
        """."""
        self._limit_upper = _np.array(arr, ndmin=1)

    @property
    def limit_lower(self):
        """."""
        return self._limit_lower

    @limit_lower.setter
    def limit_lower(self, arr):
        """."""
        self._limit_lower = _np.array(arr, ndmin=1)

    def __str__(self):
        """."""
        stg = ''
        stg += self._TMPD.format('max_number_iters', self.max_number_iters)
        stg += self._TMPD.format('max_number_evals', self.max_number_evals)
        stg += self._TMPS.format(
            'boundary_policy',
            self.BoundaryPolicy._fields[self.boundary_policy])
        if self.is_positions_consistent():
            stg += self.print_positions(
                self.limit_lower, names=['limit_lower'], print_header=True)
            stg += self.print_positions(
                self.limit_upper, names=['limit_upper'], print_header=False)
            stg += self.print_positions(
                self.initial_position, names=['initial_position'],
                print_header=False)
        else:
            stg += '\n' + '#'*50
            stg += '\ninitial_position not consistent with '
            stg += 'limit_lower and limit_upper !!\n'
            stg += '#'*50 + '\n'
        return stg

    def print_positions(self, pos, names=None, print_header=True):
        """."""
        pos = _np.array(pos, ndmin=2)

        if names is None:
            names = [f'pos. {i:5d}' for i in range(pos.shape[0])]

        stg = ''
        if print_header:
            dirs = ''.join([
                '{:^15s}'.format(f'Dir. {i:d}') for i in range(pos.shape[-1])])
            stg += '\n' + f"{'Positions':20s}" + dirs

        tmp = '\n{:20s}'+'{:^15.3g}'*pos.shape[-1]

        for p, name in zip(pos, names):
            stg += tmp.format(name, *p)
        return stg

    def normalize_positions(self, pos, is_pos=True):
        """Normalize positions to interval [0, 1] in all search directions.

        Args:
            pos (numpy.ndarray, (N, M)): Unnormalized position or direction.
                If `pos` is bi-dimensional, each row is considered to be a
                position or direction vector.
            is_pos (bool, optional): Whether pos is a position or a direction
                in search space. Since directions are not associated with a
                specific point, the `limit_lower` is not subtracted from it
                before scale normalization. Defaults to True.

        Returns:
            numpy.ndarray, (N, M): Normalized position vector with same shape
                as `pos`.

        """
        npos = pos.copy()
        if is_pos:
            npos -= self.limit_lower
        npos /= (self.limit_upper - self.limit_lower)
        return npos

    def denormalize_positions(self, npos, is_pos=True):
        """Bring a normalized position or direction back to original space.

        Args:
            npos (numpy.ndarray, (N, M)): Unnormalized position or direction.
                If `pos` is bi-dimensional, each row is considered to be a
                position or direction vector.
            is_pos (bool, optional): Whether pos is a position or a direction
                in search space. Since directions are not associated with a
                specific point, the `limit_lower` is not added to it after
                scale correction. Defaults to True.

        Returns:
            numpy.ndarray, (N, M): Un-Normalized position vector with same
                shape as `npos`.
        """
        pos = npos * (self.limit_upper - self.limit_lower)
        if is_pos:
            pos += self.limit_lower
        return pos

    def check_and_adjust_boundary(self, pos):
        """Check whether position is outside boundary and set nans when needed.

        Args:
            pos (numpy.ndarray, (N, M)): Unnormalized position. If `pos` is
                bi-dimensional, each row is considered to be a position vector.

        Returns:
            numpy.ndarray, (N, M): new position with numpy.nans applied when
                needed.

        """
        valu = self.limit_upper.copy()
        vall = self.limit_lower.copy()
        idu = pos > valu
        idl = pos < vall
        if self.boundary_policy == self.BoundaryPolicy.ToNaN:
            pos = _np.where(idu, _np.nan, pos)
            pos = _np.where(idl, _np.nan, pos)
        else:
            pos = _np.where(idu, self.limit_upper, pos)
            pos = _np.where(idl, self.limit_lower, pos)
        return pos

    def is_positions_consistent(self, pos=None):
        """."""
        pos = pos if pos is not None else self.initial_position
        wrong = (
            (pos.shape[-1] != self.limit_lower.shape[-1]) or
            (pos.shape[-1] != self.limit_upper.shape[-1]) or
            _np.any(pos <= self.limit_lower) or
            _np.any(pos >= self.limit_upper))
        return not wrong

    def to_dict(self) -> dict:
        """."""
        return {
            'max_number_iters': self.max_number_iters,
            'max_number_evals': self.max_number_evals,
            'boundary_policy': self.boundary_policy,
            'limit_lower': self.limit_lower,
            'limit_upper': self.limit_upper,
            'initial_position': self.initial_position,
            }

    def from_dict(self, dic: dict):
        """."""
        self.max_number_iters = dic.get(
            'max_number_iters', self.max_number_iters)
        self.max_number_evals = dic.get(
            'max_number_evals', self.max_number_evals)
        self.boundary_policy = dic.get('boundary_policy', self.boundary_policy)
        self.limit_lower = dic.get('limit_lower', self.limit_lower)
        self.limit_upper = dic.get('limit_upper', self.limit_upper)
        self.initial_position = dic.get(
            'initial_position', self.initial_position)


class Optimize:
    """."""

    def __init__(self, params, use_thread=True):
        """."""
        self.use_thread = use_thread
        self.params = params

        self._thread = _Thread()
        self._stopevt = _Event()

        self._num_objective_evals = 0
        self.best_positions = _np.array([], ndmin=2)
        self.best_objfuncs = _np.array([], ndmin=2)

    @property
    def num_objective_evals(self):
        """."""
        return self._num_objective_evals

    @property
    def isrunning(self):
        """."""
        if self._thread is not None:
            return self._thread.is_alive()
        return False

    def start(self):
        """."""
        if self.use_thread:
            if not self._thread.is_alive():
                self._thread = _Thread(target=self._optimize, daemon=True)
                self._stopevt.clear()
                self._thread.start()
        else:
            self._optimize()

    def stop(self):
        """."""
        if self.use_thread:
            self._stopevt.set()

    def join(self):
        """."""
        if self.use_thread:
            self._thread.join()

    def to_dict(self):
        """."""
        return {
            'params': self.params.to_dict(),
            'use_thread': self.use_thread,
            'best_positions': self.best_positions,
            'best_objfuncs': self.best_objfuncs,
            }

    def from_dict(self, dic):
        """."""
        if 'params' in dic:
            self.params.from_dict(dic['params'])
        self.use_thread = dic.get('use_thread', self.use_thread)
        self.best_positions = dic.get('best_positions', self.best_positions)
        self.best_objfuncs = dic.get('best_objfuncs', self.best_objfuncs)

    def _objective_func(self, pos):
        self._num_objective_evals += 1
        pos = self.params.check_and_adjust_boundary(pos)
        res = []
        for posi in _np.array(pos, ndmin=2):
            if _np.any(_np.isnan(posi)):
                _log.warning('Position out of boundaries. Returning NaN.')
                res.append(_np.nan)
            res.append(self.objective_function(pos))
        return _np.array(res)

    def _optimize(self):
        """."""
        raise NotImplementedError()

    def objective_function(self, pos):
        """."""
        raise NotImplementedError()
