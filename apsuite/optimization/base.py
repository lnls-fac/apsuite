"""Simulated Annealing Algorithm for Minimization."""
import logging as _log

import matplotlib.pyplot as _mplt
import numpy as _np

from mathphys.functions import get_namedtuple as _get_namedtuple

from ..utils import ParamsBaseClass as _Params, ThreadedMeasBaseClass as _Base


class OptimizationAborted(Exception):
    """."""

    pass


class OptimizeParams(_Params):
    """."""

    _TMPD = '{:30s}: {:10d} {:s}\n'.format
    _TMPF = '{:30s}: {:10.3f} {:s}\n'.format
    _TMPS = '{:30s}: {:10s} {:s}\n'.format

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
        stg += self._TMPD('max_number_iters', self.max_number_iters, '')
        stg += self._TMPD('max_number_evals', self.max_number_evals, '')
        stg += self._TMPS(
            'boundary_policy',
            self.BoundaryPolicy._fields[self.boundary_policy], '')
        if self.are_positions_consistent():
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

    def are_positions_consistent(self, pos=None):
        """."""
        pos = pos if pos is not None else self.initial_position
        wrong = (
            (pos.shape[-1] != self.limit_lower.shape[-1]) or
            (pos.shape[-1] != self.limit_upper.shape[-1]) or
            _np.any(pos <= self.limit_lower) or
            _np.any(pos >= self.limit_upper))
        return not wrong


class Optimize(_Base):
    """."""

    def __init__(self, params, use_thread=True, isonline=True):
        """."""
        super().__init__(
            params=params, target=self._target_func, isonline=isonline)
        self.use_thread = use_thread

        self._num_objective_evals = 0
        self.positions_evaluated = []
        self.positions_best = []
        self.objfuncs_evaluated = []
        self.objfuncs_best = []
        self.objfuncs_cumul_optima_idcs = None

    def to_dict(self) -> dict:
        """Dump all relevant object properties to dictionary.

        Returns:
            dict: contains all relevant properties of object.

        """
        dic = super().to_dict()
        dic['num_objective_evals'] = self.num_objective_evals
        dic['use_thread'] = self.use_thread
        dic['positions_evaluated'] = self.positions_evaluated
        dic['positions_best'] = self.positions_best
        dic['objfuncs_evaluated'] = self.objfuncs_evaluated
        dic['objfuncs_best'] = self.objfuncs_best
        dic['objfuncs_cumul_optima_idcs'] = self.objfuncs_cumul_optima_idcs

        return dic

    def from_dict(self, info: dict):
        """Update all relevant info from dictionary.

        Args:
            info (dict): dictionary with all relevant info.

        Returns:
            keys_not_used (set): Set containing keys not used by
                `self.params` object.

        """
        super().from_dict(info)
        self._num_objective_evals = info['num_objective_evals']
        self.use_thread = info['use_thread']
        self.positions_evaluated = info['positions_evaluated']
        self.positions_best = info['positions_best']
        self.objfuncs_evaluated = info['objfuncs_evaluated']
        self.objfuncs_best = info['objfuncs_best']
        self.objfuncs_cumul_optima_idcs = info["objfuncs_cumul_optima_idcs"]

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
            super().start()
        else:
            super().target()

    def objective_function(self, pos):
        """Implement here the objective function."""
        raise NotImplementedError()

    def _optimize(self):
        """Implement here optimization algorithm."""
        raise NotImplementedError()

    def _initialization():
        """To be called before optimization starts.

        If the return value is False, optimization will not run.
        """
        return True

    def _finalization(self):
        """To be called after optimization ends."""
        pass

    def _objective_func(self, pos):
        self._num_objective_evals += 1
        pos = self.params.check_and_adjust_boundary(pos)
        self.positions_evaluated.extend(_np.array(pos, ndmin=2))
        res = []
        for posi in _np.array(pos, ndmin=2):
            if self._stopevt.is_set():
                raise OptimizationAborted
            if _np.any(_np.isnan(posi)):
                _log.warning('Position out of boundaries. Returning NaN.')
                res.append(_np.nan)
            else:
                res.append(self.objective_function(posi))
        res = _np.array(res)
        # the objective function must be a (m, n)-array
        # for the m individuals values of the n objectives

        if res.ndim == 1:
            res.shape = (res.size, 1)
            # single-obj, multi-individual = column array

        if res.shape == (1, 1):
            res = res.item()
            self.objfuncs_evaluated.append(res)
            # single-obj, single-individual = a scalar
            return res

        self.objfuncs_evaluated.extend(res)
        return res

    def _target_func(self):
        if not self._initialization():
            _log.error(
                'Interrupting: There was some problem with initialization. ')
            return
        try:
            self._optimize()
        except OptimizationAborted:
            _log.info('Exiting: stop event was set.')
        self._finalization()

    def plot_history():
        """Implement visualization of the optimzation history."""
        raise NotImplementedError()

    def plot_knobspace_slice(self, dir_idcs=(1, 2), obj_idx=None):
        """Plot slice of parameter space (knobs space).

        Shows a 2-dimensional slice of the knobs components with the objective
        function value as a color code.

        Args:
            dir_idcs (tuple, list): Indices (starting from 1) of the desired
            knobs directions (components). Eg.: (1,2), (1,3) etc. Defaults to
            (1, 2).

            obj_idx (int): Index (starting from 0) of the desired objective to
            consider for the color code indicating the objective landscape.

        Returns:
            fig, ax: matplotlib fig and ax
        """
        idx1, idx2 = dir_idcs
        idx1 -= 1
        idx2 -= 1

        pos_eval = _np.array(self.positions_evaluated)
        pos_best = _np.array(self.positions_best)
        objfuncs_eval = _np.array(self.objfuncs_evaluated)

        if obj_idx is not None:
            objfuncs_eval = objfuncs_eval[:, obj_idx]

        _, pos_cum_opt, _ = self.get_cumul_optima(obj_idx)

        knob1_eval = pos_eval[:, idx1]
        knob2_eval = pos_eval[:, idx2]

        knob1_cum_opt = pos_cum_opt[:, idx1]
        knob2_cum_opt = pos_cum_opt[:, idx2]

        knob1_enditer = pos_best[:, idx1]
        knob2_enditer = pos_best[:, idx2]

        fig, ax = _mplt.subplots()
        scatter = ax.scatter(
            x=knob1_eval, y=knob2_eval,
            c=objfuncs_eval,
            vmin=objfuncs_eval.min(),
            vmax=objfuncs_eval.max(),
            label="positions evaluated"
        )

        ax.plot(
            knob1_cum_opt,
            knob2_cum_opt,
            "o", color="red", mfc="none",
            label="cumulated optima"
        )

        ax.plot(
            knob1_enditer,
            knob2_enditer,
            "x", color="red",
            label="end of iter. optima"
        )

        ax.plot(
            knob1_enditer[-1],
            knob2_enditer[-1],
            "d", color="magenta",
            markersize=10,
            label="end of run optimum"
        )
        ax.set_xlabel(f"dir {idx1 + 1}")
        ax.set_ylabel(f"dir {idx2 + 1}")

        colorbar = fig.colorbar(scatter, ax=ax)
        if obj_idx is None:
            colorbar.set_label("objective function value")
        else:
            colorbar.set_label(f"objective {obj_idx} value")

        ax.legend()
        ax.set_title("parameter space slice")

        return fig, ax

    def _get_cumul_optima_idcs(self, obj_idx=None):
        """Get the indices of the optima found during objfunc evaluations.

        self.objfuncs_cumul_optima_idcs is an m-array with the
        accumulated optima indices along the `self.objfuncs_evaluated` list
        for the chosen objective.

        obj_idx (int): index of the desired objective to be compared during
            the evaluations. Defaults to None (case of single-objective
            algorithms)
        """
        funcs = _np.array(self.objfuncs_evaluated)

        if obj_idx is not None:
            funcs = funcs[:, obj_idx]

        optima = []
        mini = _np.inf
        for i, fun in enumerate(funcs):
            if not _np.isnan(fun) and fun < mini:
                optima.append(i)
                mini = fun
        self.objfuncs_cumul_optima_idcs = _np.array(optima)

    def get_cumul_optima(self, obj_idx=None):
        """Get the accumulated optima values & positions.

        For simple single-objective single-individual algorithms, returns the
        acumulated optima along the objective function evaluations. For multi-
        objective algorithms, returns the optimal along a certain objective.

        Args:
            obj_idx (int): index of the objective to be compared. Defaults to
            None (case of single objective).

        Returns:
            idcs (m-array): the indices of the m cumulated optima.

            pos ((m, n)-array): the m n-dimensional positions where the
            cumulated optima of the specified objective happens.

            vals (m-array): the values of the specified objective at the m
            optima.
        """
        if self.objfuncs_cumul_optima_idcs is None:
            self._get_cumul_optima_idcs(obj_idx)
        idcs = self.objfuncs_cumul_optima_idcs

        vals = _np.array(self.objfuncs_evaluated)[idcs]
        pos = _np.array(self.positions_evaluated)[idcs]

        if obj_idx is not None:
            vals = vals[:, obj_idx]

        return idcs, pos, vals
