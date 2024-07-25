"""Robust Conjugate Direction Search Algorithm for Minimization."""
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
        self._boundary_policy = self.BoundaryPolicy.ToNaN
        self.initial_stepsize = 0.01  # in normalized units.
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

    @_OptimizeParams.boundary_policy.setter
    def boundary_policy(self, _):
        """In RCDS the boundary policy must always be: ToNaN."""
        self._boundary_policy = self.BoundaryPolicy.ToNaN

    def __str__(self):
        """."""
        stg = ''
        stg += self._TMPF('initial_stepsize', self.initial_stepsize, '')
        stg += self._TMPF('noise_level', self.noise_level, '')
        stg += self._TMPF('tolerance', self.tolerance, '')
        stg += self._TMPF(
            'orthogonality_threshold', self.orthogonality_threshold, '')
        stg += self._TMPS(
            'update_search_directions', str(self.update_search_directions), '')
        stg += self._TMPD('linescan_num_pts', self.linescan_num_pts, '')
        stg += self._TMPD(
            'linescan_min_pts_fit', self.linescan_min_pts_fit, '')
        stg += self._TMPS(
            'outlier_method',
            self.OutlierMethod._fields[self.outlier_method], '')
        stg += self._TMPF(
            'outlier_max_err_factor', self.outlier_max_err_factor, '')
        stg += self._TMPF(
            'outlier_percentile_limit', self.outlier_percentile_limit, '')

        stg += super().__str__()
        names = [f'Search Dir. {i:d}' for i in range(
            self.initial_search_directions.shape[0])]
        stg += self.print_positions(
            self.initial_search_directions, names=names, print_header=False)
        return stg

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

    def __init__(self, use_thread=True, isonline=True):
        """."""
        super().__init__(
            RCDSParams(), use_thread=use_thread, isonline=isonline)
        self.nr_iterations = 0
        self.nr_evals_by_iter = []

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

        idx_min = _np.argmin(func_v)
        pos_min = pos0 + delta_v[idx_min] * dir
        return pos_min, delta_v[idx_min], func_v[idx_min]

    def _objective_func(self, pos):
        pos = self.params.denormalize_positions(pos)
        return super()._objective_func(pos)

    def _initialization(self):
        """."""
        return self.params.are_positions_consistent()

    def _finalization(self):
        """."""
        self._get_cumulated_optimum_indices()
        idx = self.cumulated_optimum_idcs[-1]
        stg = '\n Finished! \n'
        stg += f'Number of iterations: {self.nr_iterations+1:04d}\n'
        stg += f'Number of evaluations: {self.num_objective_evals:04d}\n'
        init_func = self.objfuncs_evaluated[0]
        func_min = self.objfuncs_evaluated[idx]
        stg += f'f_0 = {init_func:.3g}\n'
        stg += f'f_min = {func_min:.3g}\n'
        stg += f'f_min/f0 = {func_min/init_func:.3g}\n'
        _log.info(stg)

    def _optimize(self):
        """Xiaobiao's version of Powell's direction search algorithm (RCDS).

        Xiaobiao implements his own bracketing and linescan.

        """
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
        pos_min, func_min = pos0, func0
        hist_best_pos, hist_best_func = [pos_min], [func_min]

        _log.info(f'Starting Optimization. Initial ObjFun: {func0:.3g}')
        for iter in range(max_iters):
            _log.info(f'\nIteration {iter+1:04d}/{max_iters:04d}')

            max_decr = 0
            max_decr_dir = 0

            nr_evaluations = len(self.objfuncs_evaluated)
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
            self.positions_best = self.params.denormalize_positions(
                _np.array(hist_best_pos, ndmin=2))
            self.objfuncs_best = hist_best_func

            _tmp_sdirs = self.params.denormalize_positions(
                search_dirs, is_pos=False)
            _tmp_sdirs /= _np.linalg.norm(_tmp_sdirs, axis=0)
            self.final_search_directions = _tmp_sdirs

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
                break
            elif self._num_objective_evals > max_evals:
                _log.info('Exiting: Maximum number of evaluations reached.')
                break

            func0 = func_min
            pos0 = pos_min
            self.nr_iterations = iter + 1
            evals_by_iter = len(self.objfuncs_evaluated) - nr_evaluations
            self.nr_evals_by_iter.append(evals_by_iter)
            _log.info(
                f'End of iteration {iter+1:04d}: '
                f'Final ObjFun = {func_min:.3g}')

    def plot_history(self, show_iters=True, log=False):
        """Plot the history of obj. func. and knobs throughout evaluations.

        Args:
            show_iters (bool, optional): plot vertical bars signaling an
            iteration. Defaults to True.

            log (bool, optional): whether to display the obj func plot in log
            scale. Defaults to False.

        Returns:
            fig, ax: matplolib figure and axes
        """
        idcs, pos_cum_opt, objfuncs_cum_opt = self.get_cumulated_optimum()

        fig, axs = _mplt.subplots(2, 1, figsize=(12, 12), sharex=True)
        ax = axs[0]
        ax.plot(
            self.objfuncs_evaluated,
            color="C0", alpha=0.4,
            label="evaluations",
        )
        ax.plot(
            idcs, objfuncs_cum_opt,
            "o-", mfc="none",
            color="C0",
            label="cumulative optimum",
        )
        ax.set_ylabel("objective function")
        ax.set_xlabel("evaluations")

        if show_iters:
            ymin, ymax = ax.get_ylim()
            ax.vlines(
                x=_np.cumsum(self.nr_evals_by_iter),
                ymin=ymin, ymax=ymax,
                colors="gray", alpha=0.2,
                label="new iteration",
            )
        if log:
            ax.set_yscale("log")
        ax.legend()

        ax = axs[1]
        colors = _mplt.rcParams["axes.prop_cycle"].by_key()["color"]

        for i, knob in enumerate(_np.array(self.positions_evaluated).T):
            color = colors[i % len(colors)]

            ax.plot(knob, alpha=0.4, color=color)
            ax.plot(
                idcs,
                pos_cum_opt[:, i],
                "o-", mfc="none",
                color=color,
                label=f"knob {i:2d}"
            )

        if show_iters:
            ymin, ymax = ax.get_ylim()
            ax.vlines(
                x=_np.cumsum(self.nr_evals_by_iter),
                ymin=ymin, ymax=ymax,
                colors="gray", alpha=0.2)

        ax.set_ylabel("knobs")
        ax.set_xlabel("evaluations")
        ax.legend()

        fig.tight_layout()
        fig.show()

        return fig, axs

    def plot_knobspace_slice(self, knobs_idcs=(0, 1)):
        """Plot slice of parameter space (knobs space).

        Args:
            knobs_idcs (tuple, list): Indices of the desired knobs . Defaults
            to (0, 1).

        Returns:
            fig, ax: matplotlib fig and ax
        """
        idx1, idx2 = knobs_idcs
        pos_eval = _np.array(self.positions_evaluated)
        objfuncs_eval = _np.array(self.objfuncs_evaluated)
        _, pos_cum_opt, _ = self.get_cumulated_optimum()

        knob1_eval = pos_eval[:, idx1]
        knob2_eval = pos_eval[:, idx2]

        knob1_cum_opt = pos_cum_opt[:, idx1]
        knob2_cum_opt = pos_cum_opt[:, idx2]

        fig, ax = _mplt.subplots()
        scatter = ax.scatter(
            x=knob1_eval, y=knob2_eval,
            c=objfuncs_eval,
            vmin=objfuncs_eval.min(),
            vmax=objfuncs_eval.max(),
        )

        ax.scatter(
            x=knob1_cum_opt,
            y=knob2_cum_opt,
            marker='x', color='red'
        )

        ax.set_xlabel(f"knob {idx1}")
        ax.set_ylabel(f"knob {idx2}")

        colorbar = fig.colorbar(scatter, ax=ax)
        colorbar.set_label('objective function value')

        return fig, ax

    def get_cumulated_optimum(self):
        """Gives the accumulated optimum values & positions.

        Assumes `objfuncs_evaluated` has the structure of a single-objective,
        single-popoulation algorithm, i.e., it is a simple list of scalars.

        Returns:
            idcs (m-array): the indices of the m cumulated optima. Repeats the
            last optimum.

            pos ((m, n)-array): the m n-dimensional positions of the cumulated
            minima.

            vals (m-array): the values of the objective function at the m
            cumulated minima.
        """
        if self.cumulated_optimum_idcs is None:
            self._get_cumulated_optimum_indices()
        evals = self.num_objective_evals
        idcs = self.cumulated_optimum_idcs
        idcs = _np.append(idcs, evals - 1) if idcs[-1] != evals - 1 else idcs

        vals = _np.array(self.objfuncs_evaluated)[idcs]
        vals[-1] = vals[-2]
        pos = _np.array(self.positions_evaluated)[idcs]
        pos[-1] = pos[-2]

        return idcs, pos, vals
