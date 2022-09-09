"""Simulated Annealing Algorithm for Minimization."""
from threading import Thread as _Thread, Event as _Event

import numpy as _np


class OptimizeParams:
    """."""

    _TMPD = '{:30s}: {:10d}\n'
    _TMPF = '{:30s}: {:10.3f}\n'
    _TMPS = '{:30s}: {:10s}\n'

    def __init__(self):
        """."""
        self._initial_position = _np.array([])
        self._limit_upper = _np.array([])
        self._limit_lower = _np.array([])
        self.max_number_iters = 100

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
        stg = self._TMPD.format('max_number_iters', self.max_number_iters)
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

    def is_positions_consistent(self, pos=None):
        """."""
        pos = pos if pos is not None else self.initial_position
        wrong = (
            (pos.shape[-1] != self.limit_lower.shape[-1]) or
            (pos.shape[-1] != self.limit_upper.shape[-1]) or
            _np.any(pos < self.limit_lower) or
            _np.any(pos > self.limit_upper))
        return not wrong

    def to_dict(self):
        """."""
        return {
            'max_number_iters': self.max_number_iters,
            'limit_lower': self.limit_lower,
            'limit_upper': self.limit_upper,
            'initial_position': self.initial_position,
            }

    def from_dict(self, dic):
        """."""
        self.max_number_iters = dic.get(
            'max_number_iters', self.max_number_iters)
        self.limit_lower = dic.get(
            'limit_lower', self.limit_lower)
        self.limit_upper = dic.get(
            'limit_upper', self.limit_upper)
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

        self.current_position = _np.array([])
        self.best_positions = _np.array([], ndmin=2)
        self.best_objfuncs = _np.array([], ndmin=2)

        self.initialization()

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
            'current_position': self.current_position,
            'best_positions': self.best_positions,
            'best_objfuncs': self.best_objfuncs,
            }

    def from_dict(self, dic):
        """."""
        if 'params' in dic:
            self.params.from_dict(dic['params'])
        self.use_thread = dic.get('use_thread', self.use_thread)
        self.current_position = dic.get(
            'current_position', self.current_position)
        self.best_positions = dic.get('best_positions', self.best_positions)
        self.best_objfuncs = dic.get('best_objfuncs', self.best_objfuncs)

    def initialization(self):
        """."""
        pass

    def _optimize(self):
        """."""
        raise NotImplementedError()

    def objective_function(self, pos):
        """."""
        raise NotImplementedError()
