"""."""

import math as _math
import numpy as _np

from apsuite.optimization.base import Optimize, OptimizeParams


class LeastSquaresParams(OptimizeParams):
    """."""

    _TMPE = '{:30s}: {:10.3e} {:s}\n'.format

    def __init__(self):
        """."""
        super().__init__()
        self._boundary_policy = self.BoundaryPolicy.ToBoundary
        self.max_number_iters = 10
        self.max_number_evals = _np.inf
        self.abs_tol_convergence = 1e-6
        self.rel_tol_convergence = 1e-3
        self.rcond = 1e-15
        self.damping_constant = 1.0
        self.damping_factor = 10.0
        self.max_damping_constant = 1e10
        self.tikhonov_reg_constants = 0.0
        self.jacobian = None
        self.jacobian_update_rate = 0
        self.verbose = True
        self.errorbars = None

    def __str__(self):
        """."""
        stg = super().__str__()
        stg += '\n'
        stg += '\nLeastSquaresParams:\n'
        stg += '\n'
        stg += self._TMPE('abs_tol_convergence', self.abs_tol_convergence, '')
        stg += self._TMPE('rel_tol_convergence', self.rel_tol_convergence, '')
        stg += self._TMPE('rcond', self.rcond, '')
        stg += self._TMPF('damping_constant', self.damping_constant, '')
        stg += self._TMPF('damping_factor', self.damping_factor, '')
        stg += self._TMPE(
            'max_damping_constant', self.max_damping_constant, ''
        )
        # stg += self._TMPF(
        #     'tikhonov_reg_constants', self.tikhonov_reg_constants, ''
        # )
        stg += self._TMPD(
            'jacobian_update_rate', self.jacobian_update_rate, ''
        )
        stg += self._TMPS('verbose', str(self.verbose), '')
        return stg


class LeastSquaresOptimize(Optimize):
    """Least-squares optimization via Levenberg–Marquardt style loop."""

    def __init__(
        self,
        params=None,
        merit_figure_goal=None,
        jacobian=None,
        use_thread=True,
        isonline=False,
    ):
        """."""
        super().__init__(
            params=LeastSquaresParams() if params is None else params,
            use_thread=use_thread,
            isonline=isonline,
        )
        self.merit_figure_goal = merit_figure_goal
        self.jacobian = jacobian

        self.history_chi2 = []

    def objective_function(self, pos):
        """."""
        return self.calc_residual(pos)

    def calc_chi2(self, residual):
        """."""
        chi2 = _np.sum(residual**2)
        self.history_chi2.append(chi2)
        return chi2

    def calc_chi2_tikhonov(self, residual, vtmat, tikhonov, delta_pos):
        """."""
        chi2 = self.calc_chi2(residual)
        chi2 += _np.linalg.norm(vtmat @ (tikhonov * delta_pos))**2
        return chi2

    def calc_residual(self, pos, merit_figure_goal=None):
        """."""
        merit_figure_meas = self.calc_merit_figure(pos)

        if merit_figure_goal is None:
            merit_figure_goal = self.merit_figure_goal

        res = merit_figure_meas - merit_figure_goal
        if self.params.errorbars is not None:
            res = res / self.params.errorbars
        return res

    def calc_merit_figure(self, pos):
        """."""
        _ = pos
        raise NotImplementedError(
            'Problem-specific figure of merit needs to be implemented'
        )

    def calc_jacobian(self, pos, step=1e-4):
        """."""
        jacobian_t = list()
        pos0 = pos.copy()
        for i in range(len(pos)):
            pos[i] += step / 2
            figm_pos = self.calc_merit_figure(pos)
            pos[i] -= step
            figm_neg = self.calc_merit_figure(pos)
            pos[i] = pos0[i]

            jac_col = (figm_pos - figm_neg) / step
            if self.params.errorbars is not None:
                jac_col = jac_col / self.params.errorbars
            jacobian_t.append(jac_col)
        return _np.array(jacobian_t).T

    def _optimize(self):
        """LM-like optimization loop."""

        def print_(*args, **kwargs):
            if not self.params.verbose:
                return
            print(*args, **kwargs)

        niter = self.params.max_number_iters
        atol = self.params.abs_tol_convergence
        rtol = self.params.rel_tol_convergence
        rcond = self.params.rcond

        damping = self.params.damping_constant
        damping_factor = self.params.damping_factor
        damping_max = self.params.max_damping_constant

        tikhonov = _np.asarray(self.params.tikhonov_reg_constants)
        jacobian_update_rate = self.params.jacobian_update_rate
        jacobian = self.jacobian

        pos = self.params.initial_position.copy()
        self.normal_eq_matrix = []

        res = self._objective_func(pos)
        chi2 = self.calc_chi2(res)
        print_(f'initial chi²: {chi2:.6g}')

        if jacobian is None:
            jacobian = self.calc_jacobian(pos)
            self.jacobian = jacobian
        U, S, Vt = _np.linalg.svd(jacobian, full_matrices=False)

        if tikhonov.ndim == 0:
            tikhonov = _np.full(S.size, tikhonov)
        elif tikhonov.ndim == 1:
            if tikhonov.size != S.size:
                raise ValueError('tikhonov_reg_constants size mismatch')

        for it in range(niter):
            print_(f'iteration {it:03d}')

            if jacobian_update_rate and it:
                if not it % jacobian_update_rate:
                    jacobian = self.calc_jacobian(pos)
                    self.jacobian = jacobian
                    U, S, Vt = _np.linalg.svd(jacobian, full_matrices=False)

            res = self.objfuncs_evaluated[-1]

            matrix = jacobian.T @ jacobian
            matrix += damping * _np.diag(_np.diag(matrix))
            if _np.any(tikhonov):
                matrix += _np.diag(tikhonov**2)
            self.normal_eq_matrix.append(matrix)

            if rcond is not None:
                mask = S < rcond * S[0]  # not sure mask should compare S
            else:
                mask = _np.zeros_like(S, dtype=bool)

            den = S**2 * (1 + damping) + tikhonov**2
            S_inv = S / den
            S_inv[mask] = 0

            delta = (Vt.T  @ (S_inv * (U.T @ res)))

            if _np.any(_np.isnan(delta)):
                print_('\tInvalid step direction. Aborting.')
                break

            pos_trial = pos - delta
            chi2_old = chi2

            try:
                res_trial = self._objective_func(pos_trial)
                if _np.any(_np.isnan(res_trial)):
                    raise ValueError
                if _np.any(tikhonov):
                    chi2_trial = self.calc_chi2_tikhonov(
                        res_trial, Vt.T, tikhonov, delta
                    )
                else:
                    chi2_trial = self.calc_chi2(res_trial)
                success = chi2_trial < chi2_old
            except Exception:
                success = False

            if success:
                pos = pos_trial
                res = res_trial
                chi2 = chi2_trial
                damping /= damping_factor
                print_(f'\tchi²: {chi2:.6g} (accepted)')
            else:
                damping *= damping_factor
                print_('\tchi² increased. Step rejected.')

            if damping > damping_max:
                print_('\tDamping exceeded maximum value. Aborting.')
                break

            if _math.isclose(chi2, chi2_old, rel_tol=rtol, abs_tol=atol):
                print_('\tConvergence tolerance reached.')
                break
