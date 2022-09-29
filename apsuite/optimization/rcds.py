"""Robust Conjugate Direction Search Algorithm for Minimization."""
from threading import Thread as _Thread, Event as _Event
import logging as _log

import numpy as _np
import matplotlib.pyplot as _mplt

from mathphys.functions import get_namedtuple as _get_namedtuple

from .base import Optimize as _Optimize, OptimizeParams as _OptimizeParams


class RCDSParams(_OptimizeParams):
    """Parameters used in RCDS Optimization algorithm."""

    OutlierMethod = _get_namedtuple(
        'OutlierMethod', ('Xiaobiao', 'STD', 'None_'))

    def __init__(self):
        """."""
        super().__init__()
        self.max_number_evals = 1500
        self.initial_stepsize = 0.1  # in normalized units.
        self.noise_level = 0.0  # in units of the objective function.
        self.tolerance = 1e-5  # in relative units.
        self.orthogonality_threshold = 0.9
        self.update_search_directions = True
        self.linescan_num_pts = 6
        self.linescan_min_pts_fit = 4
        self.outlier_method = self.OutlierMethod.Xiaobiao
        self.outlier_max_err_factor = 3.0
        self.outlier_percentile_limit = 0.25
        self.initial_search_directions = _np.array([], ndmin=2)

    def __str__(self):
        """."""
        stg = ''
        stg += self._TMPD.format('max_number_iters', self.max_number_iters)
        stg += self._TMPD.format('max_number_evals', self.max_number_evals)
        stg += self._TMPF.format('initial_stepsize', self.initial_stepsize)
        stg += self._TMPF.format('noise_level', self.noise_level)
        stg += self._TMPF.format('tolerance', self.tolerance)
        stg += self._TMPF.format(
            'orthogonality_threshold', self.orthogonality_threshold)
        stg += self._TMPS.format(
            'update_search_directions', str(self.update_search_directions))
        stg += self._TMPD.format('linescan_num_pts', self.linescan_num_pts)
        stg += self._TMPD.format(
            'linescan_min_pts_fit', self.linescan_min_pts_fit)
        stg += self._TMPS.format(
            'outlier_method',
            self.OutlierMethod._fields[self.outlier_method])
        stg += self._TMPF.format(
            'outlier_max_err_factor', self.outlier_max_err_factor)
        stg += self._TMPF.format(
            'outlier_percentile_limit', self.outlier_percentile_limit)

        stg += super().__str__()
        names = [f'Search Dir. {i:d}' for i in range(
            self.initial_search_directions.shape[0])]
        stg += self.print_positions(
            self.initial_search_directions, names=names, print_header=False)
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
        pos = _np.where(idu, _np.nan, pos)
        pos = _np.where(idl, _np.nan, pos)
        return pos

    def remove_outlier(self, deltas, funcs, fits):
        """."""
        max_err_factor = self.outlier_max_err_factor

        error = funcs - fits
        if self.outlier_method == self.OutlierMethod.Xiaobiao:
            perlim = self.outlier_percentile_limit

            if deltas.size < 3:
                return deltas, funcs

            indcs = _np.argsort(error)
            # since error is sorted, all diff are positive:
            diff = _np.diff(error[indcs])

            if deltas.size < 5:
                if diff[-1] > max_err_factor*diff[:-1].mean():
                    return deltas[indcs[:-1]], funcs[indcs[:-1]]
                if diff[0] > max_err_factor*diff[1:].mean():
                    return deltas[indcs[1:]], funcs[indcs[1:]]
                return deltas, funcs

            upn = max(int(error.size*(1-perlim)), 3)
            dnn = max(int(error.size*perlim), 2)
            std = diff[dnn:upn].mean()

            # For the larger diffs, the index after the first diff larger than
            # the threshold indicates the outliers:
            upcut = None
            idx = (diff[upn:] > max_err_factor*std).nonzero()[0]
            if idx.size > 0:
                # we need to increment here, to get the appropriate index:
                upcut = upn + idx.min() + 1

            # While for the smaller diffs, the index of the first diff larger
            # than the threshold indicates the outliers:
            dncut = None
            idx = (diff[:dnn] > max_err_factor*std).nonzero()[0]
            if idx.size > 0:
                # here we don't have to increment, because we started from 0:
                dncut = idx.min()

            mask = _np.zeros(error.shape, dtype=bool)
            mask[dncut:upcut] = True
            mask = _np.sort(indcs[mask])
        elif self.outlier_method == self.OutlierMethod.STD:
            std = error.std()
            mask = _np.abs(error) <= max_err_factor * std
        else:
            mask = _np.ones(error.shape, dtype=bool)

        return deltas[mask], funcs[mask]


class RCDS(_Optimize):
    """The original algorithm was developed by X. Huang from SLAC."""

    _GOLDEN_RATIO = (1 + _np.sqrt(5))/2
    _TINY = 1e-25

    def __init__(self, use_thread=True):
        """."""
        super().__init__(RCDSParams(), use_thread=use_thread)
        self._num_objective_evals = 0
        self.final_search_directions = _np.array([], dtype=float)

    @property
    def num_objective_evals(self):
        """."""
        return self._num_objective_evals

    def bracketing_min(self, pos0, func0, dir, step):
        """Bracket the minimum.

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

        if _np.isnan(func0):
            raise ValueError('Objective function is Nan for initial position.')

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

        # Lists with properties of all evaluations made along this direction:
        delta_array = []
        func_array = []

        pos = pos0 + step * dir
        func = self._objective_func(pos)
        if _np.isnan(func):
            return delta_array, func_array, mins

        delta_array.append(step)
        func_array.append(func)

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
                step = step_backup  # retrieve last valid step
                break
            delta_array.append(step)
            func_array.append(func)

            if func < mins[-1]:
                mins = pos, step, func

        return delta_array, func_array, mins

    def linescan(self, delta_array, func_array, pos0, func0, dir):
        """."""
        idx = _np.argsort(delta_array)
        delta_array = delta_array[idx]
        func_array = func_array[idx]

        if _np.isnan(func0):
            func0 = self._objective_func(pos0)

        if _np.isnan(func0):
            raise ValueError('Objective function is Nan for initial position.')

        deltas = _np.linspace(
            delta_array[0], delta_array[-1], self.params.linescan_num_pts)
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
            _log.warning('        empty list on linescan!')
            return pos0, 0, func0
        elif deltas.size < self.params.linescan_min_pts_fit:
            idx_min = funcs.argmin()
            pos_min = pos0 + deltas[idx_min] * dir
            _log.warning('        small number of points in linescan!')
            return pos_min, deltas[idx_min], funcs[idx_min]

        cfs = _np.polynomial.polynomial.polyfit(deltas, funcs, deg=2)
        fits = _np.polynomial.polynomial.polyval(deltas, cfs)
        deltas, funcs = self.params.remove_outlier(deltas, funcs, fits)
        if deltas.size < fits.size:
            _log.info(
                f'        # outliers removed = {fits.size-deltas.size:d},   '
                f'')
            cfs = _np.polynomial.polynomial.polyfit(deltas, funcs, deg=2)

        # wrong concavity must also count:
        if deltas.size <= self.params.linescan_min_pts_fit or cfs[-1] <= 0:
            idx_min = funcs.argmin()
            pos_min = pos0 + deltas[idx_min] * dir
            return pos_min, deltas[idx_min], funcs[idx_min]

        delta_v = _np.linspace(deltas[0], deltas[-1], 1000)
        func_v = _np.polynomial.polynomial.polyval(delta_v, cfs)

        self.line_data.set_data(deltas, funcs)
        self.line_fit.set_data(delta_v, func_v)
        self.axes.set_xlim([delta_v.min()*1.1, delta_v.max()*1.1])
        self.axes.set_ylim([func_v.min()*0.9, func_v.max()*1.1])
        self.figure.tight_layout()
        self.figure.show()
        _mplt.pause(2)

        idx_min = _np.argmin(func_v)
        pos_min = pos0 + delta_v[idx_min] * dir
        return pos_min, delta_v[idx_min], func_v[idx_min]

    def _objective_func(self, pos):
        self._num_objective_evals += 1
        pos = self.params.denormalize_positions(pos)
        pos = self.params.check_and_adjust_boundary(pos)
        if _np.any(_np.isnan(pos)):
            return _np.nan
        return self.objective_function(pos)

    def _optimize(self):
        """Xiaobiao's version of Powell's direction search algorithm (RCDS).

        Xiaobiao implements his own bracketing and linescan.

        """
        self.figure, self.axes = _mplt.subplots(1, 1)
        self.line_data = self.axes.plot([1, 1], [1, 1], 'or', label='Data')[0]
        self.line_fit = self.axes.plot([1, 1], [1, 1], '-b', label='Fit')[0]
        self.axes.legend(loc='best')
        self.axes.set_ylabel('Objective Function')
        self.axes.set_xlabel('Direction Search Steps')

        self._num_objective_evals = 0
        search_dirs = self.params.initial_search_directions.copy()
        search_dirs = self.params.normalize_positions(
            search_dirs, is_pos=False)
        search_dirs /= _np.sqrt(_np.sum(
            search_dirs*search_dirs, axis=1))[:, None]

        step = self.params.initial_stepsize
        tol = self.params.tolerance
        max_iters = self.params.max_number_iters
        max_evals = self.params.max_number_evals
        pos0 = self.params.normalize_positions(self.params.initial_position)

        func0 = self._objective_func(pos0)
        init_func = func0
        pos_min, func_min = pos0, func0
        hist_best_pos, hist_best_func = [pos_min], [func_min]

        _log.info(f'Starting Optimization. Initial ObjFun: {func0:.3g}')
        for iter in range(max_iters):
            _log.info(f'\nIteration {iter+1:04d}/{max_iters:04d}')
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
                    f'    Bracketing Dir. {idx+1:d}:  '
                    f'Delta = ['
                    f'{delta_array[0]:.3g}, {delta_array[-1]:.3g}]   '
                    f'Objfun = {mins[-1]:.3g}')

                pos_idx, delta_idx, func_idx = self.linescan(
                    delta_array, func_array, mins[0], mins[-1], dir)
                _log.info(
                    f'    Linescan in Dir. {idx+1:d}:    '
                    f'Delta = {delta_idx:.3g}   Objfun = {func_idx:.3g}')

                if (func_min - func_idx) > max_decr:
                    max_decr = func_min - func_idx
                    max_decr_dir = idx
                    _log.info(
                        f'    Updated ObjFun largest decrease {max_decr:.3f}.')
                func_min = func_idx
                pos_min = pos_idx
                _log.info('')

            # Define an extension point colinear with pos0 and pos_min:
            pos_e = 2 * pos_min - pos0
            func_e = 0
            if self.params.update_search_directions:
                _log.info('    Checking extension point.')
                func_e = self._objective_func(pos_e)

            # Calculate new normalized direction:
            diff = pos_min - pos0
            new_dir = diff/_np.linalg.norm(diff)
            # used for checking orthogonality with other directions:
            max_dotp = (search_dirs @ new_dir).max()

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
                    f'    Dir. {max_decr_dir+1:d} not replaced:\n'
                    f'        func0 >= func_e: {cond1}\n'
                    f'        cond 2nd deriv.: {cond2}')
            # only accept if reasonably orthogonal:
            elif max_dotp < self.params.orthogonality_threshold:
                _log.info(f'    Replacing direction {max_decr_dir+1:d}')
                search_dirs[max_decr_dir:-1] = search_dirs[max_decr_dir+1:]
                search_dirs[-1] = new_dir

                delta_array, func_array, mins = self.bracketing_min(
                    pos_min, func_min, new_dir, step)
                _log.info(
                    f'        Bracket result:  '
                    f'Delta = [{delta_array[0]:.3g}, {delta_array[-1]:.3g}]   '
                    f'Objfun = {mins[-1]:.3g}')

                pos_idx, _, func_idx = self.linescan(
                    delta_array, func_array, mins[0], mins[-1], dir)
                _log.info(
                    f'        Linescan result: Delta = {delta_idx:.3g}   '
                    f'Objfun = {func_idx:.3g}')
                func_min = func_idx
                pos_min = pos_idx
            else:
                _log.info(
                    f'    Dir. {max_decr_dir+1:d} not replaced: '
                    f'max dot product {max_dotp:.3f}')

            hist_best_pos.append(pos_min)
            hist_best_func.append(func_min)

            # Numerical recipes does:
            # cond = 2*(func0-func_min) <= \
            #       tol*(abs(func0)+abs(func_min)) + self._TINY
            cond = 2 * abs(func0-func_min) <= tol*(abs(func0)+abs(func_min))
            # if abs(func_min) < tol:
            if (cond and tol > 0):
                _log.info(
                    f'Exiting: Init ObjFun = {func0:.3g},  '
                    f'Final ObjFun = {func_min:.3g}')
                break
            elif self._stopevt.is_set():
                _log.info('Exiting: stop event was set.')
                break
            elif self._num_objective_evals > max_evals:
                _log.info('Exiting: Maximum number of evaluations reached.')
                break

            func0 = func_min
            pos0 = pos_min
            _log.info(
                f'End of iteration {iter+1:04d}: '
                f'Final ObjFun = {func_min:.3g}')

        stg = '\n Finished! \n'
        stg += f'Numer of iterations: {iter+1:04d}\n'
        stg += f'Numer of evaluations: {self.num_objective_evals:04d}\n'
        stg += f'f_0 = {init_func:.3g}\n'
        stg += f'f_min = {func_min:.3g}\n'
        stg += f'f_min/f0 = {func_min/init_func:.3g}\n'
        _log.info(stg)

        self.best_positions = self.params.denormalize_positions(
            _np.array(hist_best_pos, ndmin=2))
        self.best_objfuncs = _np.array(hist_best_func, ndmin=2)

        self.final_search_directions = self.params.denormalize_positions(
            search_dirs, is_pos=False)
        self.final_search_directions /= _np.linalg.norm(
            self.final_search_directions, axis=0)
