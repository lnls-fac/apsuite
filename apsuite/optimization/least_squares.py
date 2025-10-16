"""."""

import numpy as _np

from apsuite.optimization.base import Optimize, OptimizeParams


class LeastSquaresParams(OptimizeParams):
    """."""

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
        self.ridge_constant = 0.0
        self.jacobian = None
        self.jacobian_update_rate = 0
        self.verbose = True
        self.learning_rate = 1.0
        self.min_learning_rate = 1e-3
        self.backtracking_factor = 2
        self.patience = 5
        self.errorbars = None


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
        merit_figure = self.calc_merit_figure(pos)
        residual = self.calc_residual(merit_figure)
        return residual

    def calc_chi2(self, residual):
        """."""
        chi2 = _np.mean(residual**2)
        self.history_chi2.append(chi2)
        return chi2

    def calc_residual(self, merit_figure_meas, merit_figure_goal=None):
        """."""
        if merit_figure_goal is None:
            merit_figure_goal = self.merit_figure_goal

        res = merit_figure_meas - merit_figure_goal
        if self.params.errorbars is not None:
            res = res / self.params.errorbars
        return res

    def calc_merit_figure(self, pos):
        """."""
        raise NotImplementedError(
            'Problem-specific figure of merit needs to be implemented'
        )

    def calc_jacobian(self, pos=None, step=1e-4):
        """."""
        jacobian_t = list()
        pos0 = self.params.initial_position
        pos = pos0.copy() if pos is None else pos

        for i in range(len(pos)):
            pos[i] += step / 2
            figm_pos = self.calc_merit_figure(pos)
            pos[i] = pos0[i]
            pos[i] -= step / 2
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
        jacobian = self.jacobian
        damping_constant = self.params.damping_constant
        damping_factor = self.params.damping_factor
        ridge_constant = self.params.ridge_constant
        jacobian_update_rate = self.params.jacobian_update_rate
        lr = self.params.learning_rate
        lr_min = self.params.min_learning_rate
        lr_factor = self.params.backtracking_factor
        patience = self.params.patience

        pos0 = self.params.initial_position
        pos = pos0.copy()

        res = self._objective_func(pos)
        chi2 = self.calc_chi2(res)
        M = self.calc_jacobian(pos) if jacobian is None else jacobian

        # if c:
        #     chi2 += w * _np.std(res0) * _np.linalg.norm(pos0)

        print_(f'initial chi²: {chi2:.6g}')

        MTM = M.T @ M
        ridge_reg = ridge_constant * _np.eye(
            MTM.shape[0]
        )  # Ridge regularization

        for it in range(niter):
            print_(f'iteration {it:03d}')

            pos_init = pos.copy()

            if jacobian_update_rate and it:
                if not it % jacobian_update_rate:
                    M = self.calc_jacobian(pos)
                    MTM = M.T @ M

            lm_reg = _np.diag(
                damping_constant * _np.diag(MTM)
            )  # Levenberg-Macquardt regularization

            matrix = MTM + ridge_reg + lm_reg

            res = self.objfuncs_evaluated[-1]  # last evaluated residual
            delta = (
                _np.linalg.pinv(matrix, rcond=self.params.rcond) @ M.T @ res
            )  # LM/GN Normal equation
            # TODO: include processing of pseudo-inverse for singvals selection
            pos -= lr * delta  # position update

            if _np.any(_np.isnan(pos)):
                print_('\tNaNs detected in pos. Aborting.')
                break

            fails = 0
            while fails <= patience:
                try:
                    res = self._objective_func(pos)
                    break
                except Exception:
                    print_('Merit figure evaluation failed. Back-tracking.')
                    fails += 1
                    lr /= lr_factor  # reduce step size
                    lr = min(lr, lr_min)
                    pos = pos_init  # restore pos
                    pos -= lr * delta  # apply smaller step

            chi2_old = self.history_chi2[-1]
            chi2_new = self.calc_chi2(res)
            # if c:
            #     chi2 += w * _np.std(res) * _np.linalg.norm(pos)
            print_(f'\tchi²: {chi2_new:.6g}')

            delta_chi2 = chi2_old - chi2_new
            if abs(delta_chi2) < atol or abs(delta_chi2) / chi2_old < rtol:
                print_('\tConvergence tolerance reached. Exiting.')
                break

            if damping_constant:
                if delta_chi2 > 0:
                    damping_constant /= damping_factor
                    print_('\tImproved fit. Decreasing LM damping_constant.')
                else:
                    damping_constant *= damping_factor
                    print_('\tWorsened fit. Increasing LM damping_constant.')
