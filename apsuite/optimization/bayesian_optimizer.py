"""Bayesian Optimization (minimization) GPy Gaussian Procees w/ RBF kernel."""

import numpy as _np
import GPy as _GPy

from scipy.optimize import minimize as _minimize
from scipy.stats import norm as _norm

from .base import Optimize as _Optimize, OptimizeParams as _OptimizeParams


class BayesianOptimizerGPyParams(_OptimizeParams):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.num_init_random_pts = 5  # random points for initial sampling
        self.xi = 0.01  # Expected Improvement exploitation/exploration
        self.ard = True  # length-scale for each knob
        self.num_restarts_gp = 2  # GP hyperparams fit restarts
        self.num_restarts_acq = 2  # Acq. func. optimization restarts
        self.seed = 0

    def __str__(self):
        """."""
        stg = super().__str__()
        stg += '\n'
        stg += self._TMPD('num_init_random_pts', self.num_init_random_pts, '')
        stg += self._TMPF('xi', self.xi, '')
        stg += self._TMPS('ard', str(self.ard), '')
        stg += self._TMPD('num_restarts_gp', self.num_restarts_gp, '')
        stg += self._TMPD('num_restarts_acq', self.num_restarts_acq, '')
        return stg


class BayesianOptimizerGPy(_Optimize):
    """Bayesian Optimizer with GP surrogate (RBF kernel) and EI acq function.

    Implements initial sampling, Gaussian Process (GP) fit and next point
    selection via Expected Improvement (EI) maximization.
    """

    def __init__(self, use_thread=True, isonline=True):
        """."""
        super().__init__(
            params=BayesianOptimizerGPyParams(),
            use_thread=use_thread,
            isonline=isonline,
        )
        self._rng = _np.random.default_rng(self.params.seed)
        self._gp_model = None
        self._y_mean = 0.0
        self._y_std = 1.0

    def _optimize(self):
        dim = self.params.limit_lower.size

        for _ in range(self.params.num_init_random_pts):
            if self.num_objective_evals >= self.params.max_number_evals:
                return
            pos_normalized = self._rng.uniform(0.0, 1.0, size=dim)
            pos = self.params.denormalize_positions(pos_normalized)
            self._evaluate_and_register(pos)

        n_bo_iters = self.params.max_number_iters - len(self.positions_best)
        for _ in range(max(n_bo_iters, 0)):
            if self.num_objective_evals >= self.params.max_number_evals:
                return
            self._fit_gp()
            pos_normalized = self._propose_next(dim)
            pos = self.params.denormalize_positions(pos_normalized)
            self._evaluate_and_register(pos)

    def _evaluate_and_register(self, pos):
        obj = self._objective_func(pos)
        self._update_best(pos, obj)

    def _update_best(self, pos, obj):
        is_first = not self.objfuncs_best
        is_better = (
            (not is_first)
            and (not _np.isnan(obj))
            and (obj < self.objfuncs_best[-1])
        )
        if is_first or is_better:
            self.objfuncs_best.append(obj)
            self.positions_best.append(pos)
        else:
            self.objfuncs_best.append(self.objfuncs_best[-1])
            self.positions_best.append(self.positions_best[-1])

    def _fit_gp(self):
        pos_eval = _np.array(self.positions_evaluated)
        obj_eval = _np.array(self.objfuncs_evaluated).reshape(-1, 1)

        valid = ~_np.isnan(obj_eval).ravel()
        pos_eval = pos_eval[valid]
        obj_eval = obj_eval[valid]

        pos_eval_normalized = self.params.normalize_positions(pos_eval)

        self._y_mean = obj_eval.mean()
        self._y_std = obj_eval.std() if obj_eval.std() > 1e-9 else 1.0
        yn = (obj_eval - self._y_mean) / self._y_std  # normalize data

        dim = pos_eval_normalized.shape[-1]
        kernel = _GPy.kern.RBF(input_dim=dim, ARD=self.params.ard)  # RBF + ARD
        kernel += _GPy.kern.White(input_dim=dim)  # white noise kernel

        self._gp_model = _GPy.models.GPRegression(
            X=pos_eval_normalized, Y=yn, kernel=kernel
        )
        self._gp_model.optimize_restarts(
            num_restarts=self.params.num_restarts_gp, verbose=False
        )  # restarts of the GP hyperparams optimization
        # if num_restarts_gp == 0, only a single optimization is done
        # which is equivalent to _gp_model.optimize().

    def _predict(self, pos_normalized):
        mu_n, var_n = self._gp_model.predict(pos_normalized.reshape(1, -1))
        mu = mu_n[0, 0] * self._y_std + self._y_mean
        sigma = _np.sqrt(max(var_n[0, 0], 1e-12)) * self._y_std
        return mu, sigma

    def _expected_improvement(self, pos_normalized):
        mu, sigma = self._predict(pos_normalized)
        sigma = max(sigma, 1e-9)

        y_best = _np.nanmin(self.objfuncs_evaluated)  # minimization
        imp = y_best - mu - self.params.xi
        z = imp / sigma
        ei = imp * _norm.cdf(z) + sigma * _norm.pdf(z)
        return -ei  # minimze negative EI = maximize EI

    def _propose_next(self, dim):
        best_npos, best_val = None, _np.inf
        bounds = [(0.0, 1.0)] * dim

        for _ in range(self.params.num_restarts_acq + 1):
            x0 = self._rng.uniform(0.0, 1.0, size=dim)
            res = _minimize(
                self._expected_improvement,
                x0=x0,
                bounds=bounds,
                method='L-BFGS-B',
            )
            if res.fun < best_val:
                best_val, best_npos = res.fun, res.x

        return best_npos
