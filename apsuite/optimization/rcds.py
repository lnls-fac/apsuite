"""Robust Conjugate Direction Search Algorithm for Minimization."""
from threading import Thread as _Thread, Event as _Event
import logging as _log

import numpy as _np
import matplotlib.pyplot as _mplt

from .base import Optimize as _Optimize, OptimizeParams as _OptimizeParams


class RCDSParams(_OptimizeParams):

    def __init__(self):
        """."""
        super().__init__()
        self.initial_stepsize = 0.1  # in normalized units.
        self.noise_level = 0.0  # in units of the objective function.
        self.tolerance = 1e-5  # in relative units.
        self.initial_search_directions = _np.array([], ndmin=2)
        self.update_search_directions = True

    def __str__(self):
        """."""
        stg = self._TMPF.format('initial_stepsize', self.initial_stepsize)
        stg += self._TMPF.format('noise_level', self.noise_level)
        stg += self._TMPF.format('tolerance', self.tolerance)
        stg += self._TMPS.format(
            'update_search_directions', str(self.update_search_directions))
        stg += super().__str__()
        names = [f'Search Dir. {i:d}' for i in range(
            self.initial_search_directions.shape[0])]
        stg += self.print_positions(
            self.initial_search_directions, names=names, print_header=False)
        return stg


class RCDS(_Optimize):
    """The original algorithm was developed by X. Huang from SLAC."""

    _GOLDEN_RATIO = (1 + _np.sqrt(5))/2
    _TINY = 1e-25

    def __init__(self, use_thread=True):
        """."""
        super().__init__(RCDSParams(), use_thread=use_thread)
        self._num_objective_evals = 0

        self.figure, self.axes = _mplt.subplots(1, 1)
        self.line_data = self.axes.plot([1, 1], [1, 1], 'or', label='Data')[0]
        self.line_fit = self.axes.plot([1, 1], [1, 1], '-b', label='Fit')[0]
        self.axes.legend(loc='best')
        self.axes.set_ylabel('Objective Function')
        self.axes.set_xlabel('Direction Search Steps')

    @property
    def num_objective_evals(self):
        """."""
        return self._num_objective_evals

    def _check_lim(self):
        # If particle position exceeds the boundary, set the boundary value
        if self._pos_lim_upper is not None:
            over = self._position > self._pos_lim_upper
            self._position[over] = self._pos_lim_upper[over]
        if self._pos_lim_lower is not None:
            under = self._position < self._pos_lim_lower
            self._position[under] = self._pos_lim_lower[under]

    def bracketing_min(self, pos0, func0, dir, step):
        """Brackets the minimum

        Args:
            pos0 (n-dimensional np.array): starting point in param space
            func0 (float): objective function at pos0
            dir (n-dimensional np.array): direction vector
            step (float): initial step

        Returns:
            info (dict): dictionary containing the input args as well as a
                list of such evaluations and steps 'xflist', the step for
                which the minimum is attained 'delta_min', the minimum point
                'pos_min' and the obj. func minimum value, 'func_min'.
        """
        if _np.isnan(func0):
            func0 = self._objective_func(pos0)

        # mins is a tuple with informations about the minimum so far:
        mins = pos0, 0, func0

        # Lists with properties of all evaluations made along this direction:
        delta_array = [0]
        func_array = [func0]

        d_array, f_array, mins = self.search_bound(
            pos0, mins, dir, step, direction='positive')
        delta_array.extend(d_array)
        func_array.extend(f_array)

        # if False, no need to search bound in the negative direction:
        if func0 <= (mins[-1] + 3 * self.params.noise_level):
            d_array, f_array, mins = self.search_bound(
                pos0, mins, dir, -step, direction='negative')
            delta_array.extend(d_array)
            func_array.extend(f_array)

        delta_array = _np.array(delta_array)
        func_array = _np.array(func_array)

        # subtract delta_min since pos_min is the new reference
        delta_array -= mins[1]

        inds = _np.argsort(delta_array)
        delta_array = delta_array[inds]
        func_array = func_array[inds]
        return delta_array, func_array, mins

    def search_bound(self, pos0, mins, dir, step, direction='pos', nsigma=3):
        """."""
        direction = direction.lower()[:3]  # pos or neg

        pos = pos0 + step * dir
        func = self._objective_func(pos)

        # Lists with properties of all evaluations made along this direction:
        delta_array = [step]
        func_array = [func]

        if func < mins[-1]:
            mins = pos, step, func

        while func < (mins[-1] + nsigma * self.params.noise_level):
            step_backup = step
            if abs(step) < 0.1:
                # NOTE: wouldn't it be only phi istead of 1 + phi?
                step *= (1 + self._GOLDEN_RATIO)
            else:
                step += 0.1 if direction.startswith('pos') else -0.1

            pos = pos0 + step * dir
            func = self._objective_func(pos)

            if _np.isnan(func):
                step = step_backup
                break
            delta_array.append(step)
            func_array.append(func)

            if func < mins[-1]:
                mins = pos, step, func

        return delta_array, func_array, mins

    def linescan(self, delta_array, func_array, pos0, func0, dir, npts=6):
        """."""
        idx = _np.argsort(delta_array)
        delta_array = delta_array[idx]
        func_array = func_array[idx]

        if _np.isnan(func0):
            func0 = self._objective_func(pos0)

        deltas = _np.linspace(delta_array[0], delta_array[-1], npts)
        funcs = deltas * _np.nan

        idx = _np.argmin(
            _np.abs(deltas[:, None] - delta_array[None, :]), axis=0)
        deltas[idx] = delta_array
        funcs[idx] = func_array

        for idx, delta in enumerate(deltas):
            if _np.isnan(funcs[idx]):
                pos = pos0 + delta * dir
                funcs[idx] = self._objective_func(pos)

        mask = ~_np.isnan(funcs)
        deltas = deltas[mask]
        funcs = funcs[mask]

        if deltas.size == 0:
            return pos0, 0, func0

        coeffs = _np.polynomial.polynomial.polyfit(deltas, funcs, deg=2)
        if deltas.size < 3 or coeffs[-1] <= 0:  # wrong concavity
            idx_min = funcs.argmin()
            pos_min = pos0 + deltas[idx_min] * dir
            func_min = funcs[idx_min]
            return pos_min, deltas[idx_min], func_min

        delta_v = _np.linspace(deltas[0], deltas[-1], 1000)
        func_v = _np.polynomial.polynomial.polyval(delta_v, coeffs)

        self.line_data.set_data(deltas, funcs)
        self.line_fit.set_data(delta_v, func_v)
        self.axes.set_xlim([delta_v.min(), delta_v.max()])
        self.axes.set_ylim([func_v.min(), func_v.max()])
        self.figure.tight_layout()
        self.figure.show()
        _mplt.pause(3)

        idx_min = _np.argmin(func_v)
        pos_min = pos0 + delta_v[idx_min] * dir
        func_min = func_v[idx_min]
        return pos_min, delta_v[idx_min], func_min

    def _normalize_positions(self, pos):
        npos = (pos - self.params.limit_lower)
        npos /= (self.params.limit_upper - self.params.limit_lower)
        return npos

    def _denormalize_positions(self, npos):
        pos = npos * (self.params.limit_upper - self.params.limit_lower)
        pos += self.params.limit_lower
        return pos

    def _objective_func(self, pos):
        self._num_objective_evals += 1
        pos = self._denormalize_positions(pos)
        return self.objective_function(pos)

    def _optimize(self):
        """Xiaobiao's version of Powell's direction search algorithm (RCDS).

        Xiaobiao implements his own bracketing and linescan.

        """
        self._num_objective_evals = 0
        search_dirs = self.params.initial_search_directions.copy()
        search_dirs = self._normalize_positions(search_dirs)
        search_dirs /= _np.sum(search_dirs*search_dirs, axis=1)[:, None]

        step = self.params.initial_stepsize
        tol = self.params.tolerance
        max_iters = self.params.max_number_iters
        pos0 = self._normalize_positions(self.params.initial_position)

        func0 = self._objective_func(pos0)
        init_func = func0
        pos_min, func_min = pos0, func0
        hist_best_pos, hist_best_func = [pos_min], [func_min]

        for iter in range(max_iters):
            _log.info(f'Iteration {iter+1:04d}/{max_iters:04d}\n')
            max_decr = 0
            max_decr_dir = 0

            # NOTE: where does this step division come from?
            # Not in numerical recipes. Check Powell' method again!
            step /= 1.20

            for idx in range(search_dirs.shape[0]):
                dir = search_dirs[idx]
                delta_array, func_array, mins = self.bracketing_min(
                    pos_min, func_min, dir, step)

                _log.info(
                    f'Dir. {idx+1:d}. Obj. Func. Min: {mins[-1]}')

                pos_idx, _, func_idx = self.linescan(
                    delta_array, func_array, mins[0], mins[-1], dir)

                if (func_min - func_idx) > max_decr:
                    max_decr = func_min - func_idx
                    max_decr_dir = idx
                    _log.info(
                        f'Largest obj.func delta = {max_decr:f}, updated.')
                func_min = func_idx
                pos_min = pos_idx

            # Define an extension point colinear with pos0 and pos_min:
            pos_e = 2 * pos_min - pos0
            func_e = 0
            if self.params.update_search_directions:
                _log.info('Evaluating objective func. at extension point...')
                func_e = self._objective_func(pos_e)
                _log.info('Done!\n')

            # Calculate new normalized direction:
            diff = pos_min - pos0
            new_dir = diff/_np.linalg.norm(diff)
            # used for checking orthogonality with other directions:
            max_dotp = (new_dir.T @ search_dirs).max()

            # Numerical Recipes conditions (same order)
            cond1 = func0 <= func_e
            cond2_lhs = 2 * (func0 - 2 * func_min + func_e)
            cond2_lhs *= (func0 - func_min - max_decr)**2
            cond2_rhs = max_decr * (func_e - func0)**2
            cond2 = cond2_lhs >= cond2_rhs
            if not self.params.update_search_directions:
                pass
            elif cond1 or cond2:
                _log.info(
                    f'Direction {max_decr_dir+1:d} not replaced: '
                    f'Condition 1: {cond1}; Condition 2: {cond2}')
            elif max_dotp < 0.9:  # only accept if reasonably orthogonal
                _log.info(f'Replacing direction {max_decr_dir+1:d}')
                search_dirs[max_decr_dir:-1] = search_dirs[max_decr_dir+1:]
                search_dirs[-1] = new_dir

                delta_array, func_array, mins = self.bracketing_min(
                    pos_min, func_min, new_dir, step)
                _log.info(
                    f'Iteration {iter+1:d}, New dir. {max_decr_dir+1:d}, '
                    f'Obj. Func. Min {func_min:f}')
                pos_idx, _, func_idx = self.linescan(
                    delta_array, func_array, mins[0], mins[-1], dir)
                func_min = func_idx
                pos_min = pos_idx
            else:
                _log.info('Direction replacement conditions were met.')
                _log.info(
                    f'Skipping new direction {max_decr_dir+1:d}: '
                    f'max dot product {max_dotp:f}')

            hist_best_pos.append(pos_min)
            hist_best_func.append(func_min)

            # Numerical recipes does:
            # cond = 2*(func0-func_min) <= \
            #       tol*(abs(func0)+abs(func_min)) + self._TINY
            cond = 2 * (func0 - func_min) <= tol * (abs(func0) + abs(func_min))
            # if abs(func_min) < tol:
            if (cond and tol > 0) or self._stopevt.is_set():
                _log.info(
                    f'Quiting : Condition: {cond}; func0: {func0}, '
                    f'func_min: {func_min}: Event: {self._stopevt.is_set()}')

                break

            func0 = func_min
            pos0 = pos_min

        stg = 'Finished! \n'
        stg += f'f_0 = {init_func:4.2e}\n'
        stg += f'f_min = {func_min:4.2e}\n'
        ratio = func_min/init_func
        stg += f'f_min/f0 = {ratio:4.2e}\n'
        _log.info(stg)

        self.hist_best_positions = self._denormalize_positions(
            _np.array(hist_best_pos, ndmin=2))
        self.hist_best_objfunc = _np.array(hist_best_func, ndmin=2)
        self.best_direction = self._denormalize_positions(search_dirs)
