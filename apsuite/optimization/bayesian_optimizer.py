"""Bayesian Optimization (minimization) Gaussian Process.

Implements a single-armed bandit Trust Region Bayesian Optimization
(TuRBO-1). Reference [1]: Erikson D. et al, Scalable Global Optimization via
Local Bayesian Optimization, 2019. https://arxiv.org/pdf/1910.01739.
"""

import numpy as _np
import GPy as _GPy

from scipy.optimize import minimize as _minimize
from scipy.stats import norm as _norm

from mathphys.functions import get_namedtuple as _get_namedtuple
from .base import Optimize as _Optimize, OptimizeParams as _OptimizeParams


class BayesianOptimizerGPyParams(_OptimizeParams):
    """."""

    AcqFuncType = _get_namedtuple(
        'AcqFuncType', ('UpperConfidenceBound', 'ExpectedImprovement')
    )
    TrustRegionMode = _get_namedtuple(
        'TrustRegionMode',
        ('NoTrustRegion', 'FixedTrustRegion', 'AdaptiveTrustRegion'),
    )

    def __init__(self):
        """."""
        super().__init__()
        self.num_init_random_pts = 5  # random points for initial sampling
        self.acq_func_type = self.AcqFuncType.UpperConfidenceBound
        self.xi_param = 0.01  # Expected Improvement exploitation/exploration
        self.beta_param = 0.01  # UCB exploitation/exploration trade-off
        self.automatic_relevance_determination = (
            True  # independent length-scale for each knob
        )
        self.num_restarts_gp = 2  # GP hyperparams fit restarts
        self.num_restarts_acq = 2  # Acq. func. optimization restarts
        self.seed = 0
        self.trust_region_mode = self.TrustRegionMode.NoTrustRegion
        self.trust_region_scale = 0.2  # relative radius in normalized
        # [0,1]^dim space: used as a fixed radius in FixedTrustRegion, and as
        # L0 (initial length) in AdaptiveTrustRegion

        # for the following params, refer to Appendix D of Ref. [1]
        self.trust_region_length_min = 0.5**7  # AdaptiveTrustRegion only:
        # restart is triggered once L falls below this value
        self.trust_region_length_max = 1.6  # AdaptiveTrustRegion only: upper
        # bound on L growth
        self.trust_region_success_tol = 3  # AdaptiveTrustRegion only:
        # consecutive successes required to double L
        self.trust_region_failure_tol = None  # AdaptiveTrustRegion only:
        # consecutive failures required to halve  L;
        # None uses max(4, dim), the heuristic from the original TuRBO
        # paper

    def __str__(self):
        """."""
        stg = super().__str__()
        stg += '\n'
        stg += self._TMPD('num_init_random_pts', self.num_init_random_pts, '')
        stg += self._TMPS(
            'acq_func_type', self.AcqFuncType._fields[self.acq_func_type], ''
        )
        stg += self._TMPF('xi_param', self.xi_param, '')
        stg += self._TMPF('beta_param', self.beta_param, '')
        stg += self._TMPS(
            'automatic_relevance_determination',
            str(self.automatic_relevance_determination),
            '',
        )
        stg += self._TMPD('num_restarts_gp', self.num_restarts_gp, '')
        stg += self._TMPD('num_restarts_acq', self.num_restarts_acq, '')
        stg += self._TMPD('seed', self.seed, '')
        stg += self._TMPS(
            'trust_region_mode',
            self.TrustRegionMode._fields[self.trust_region_mode],
            '',
        )
        stg += self._TMPF('trust_region_scale', self.trust_region_scale, '')
        stg += self._TMPF(
            'trust_region_length_min', self.trust_region_length_min, ''
        )
        stg += self._TMPF(
            'trust_region_length_max', self.trust_region_length_max, ''
        )
        stg += self._TMPD(
            'trust_region_success_tol', self.trust_region_success_tol, ''
        )
        return stg


class BayesianOptimizerGPy(_Optimize):
    """Bayesian Optimizer with GP surrogate (RBF kernel) and EI/UCB.

    Implements initial sampling, Gaussian Process (GP) fit and next point
    selection via Expected Improvement (EI) or Upper Confidence Bound (UCB)
    maximization, according to `params.acq_func_type`.

    Optionally restricts the acquisition search to a trust region around
    the best known point, either with a fixed radius (FixedTrustRegion) or
    an adaptive radius that expands/contracts and restarts on collapse
    (AdaptiveTrustRegion, TuRBO-1 style), according to
    `params.trust_region_mode`.

    Reference: Erikson D. et al, Scalable Global Optimization via Local
        Bayesian Optimization, 2019. https://arxiv.org/pdf/1910.01739.
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

        # adaptive trust region state
        self._tr_length = self.params.trust_region_scale
        self._tr_center = None
        self._tr_success_count = 0
        self._tr_failure_count = 0
        self._tr_failure_tol = 4
        self._tr_best_obj = _np.inf

    def _optimize(self):
        dim = self.params.limit_lower.size
        TRM = self.params.TrustRegionMode

        for _ in range(self.params.num_init_random_pts):
            if self.num_objective_evals >= self.params.max_number_evals:
                return
            pos_normalized = self._rng.uniform(0.0, 1.0, size=dim)
            pos = self.params.denormalize_positions(pos_normalized)
            self._evaluate_and_register(pos)

        if self.params.trust_region_mode == TRM.AdaptiveTrustRegion:
            self._init_trust_region(dim)

        n_bo_iters = self.params.max_number_iters - len(self.positions_best)
        for _ in range(max(n_bo_iters, 0)):
            if self.num_objective_evals >= self.params.max_number_evals:
                return
            self._fit_gp()
            pos_normalized = self._propose_next(dim)
            pos = self.params.denormalize_positions(pos_normalized)
            obj = self._evaluate_and_register(pos)

            if self.params.trust_region_mode == TRM.AdaptiveTrustRegion:
                self._update_trust_region(obj, pos_normalized)

    def _evaluate_and_register(self, pos):
        obj = self._objective_func(pos)
        self._update_best(pos, obj)
        return obj

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

    # ------------------------------------------------------------------ #
    # Trust region
    # ------------------------------------------------------------------ #

    def _init_trust_region(self, dim):
        self._tr_length = self.params.trust_region_scale
        self._tr_success_count = 0
        self._tr_failure_count = 0
        self._tr_failure_tol = (
            self.params.trust_region_failure_tol
            if self.params.trust_region_failure_tol is not None
            else max(4, dim)
        )

        obj_eval = _np.array(self.objfuncs_evaluated, dtype=float)
        if obj_eval.size and not _np.isnan(obj_eval).all():
            best_idx = _np.nanargmin(obj_eval)
            best_pos = _np.array(self.positions_evaluated)[best_idx]
            self._tr_center = self.params.normalize_positions(
                best_pos.reshape(1, -1)
            )[0]
            self._tr_best_obj = obj_eval[best_idx]
        else:
            self._tr_center = _np.full(dim, 0.5)
            self._tr_best_obj = _np.inf

    def _update_trust_region(self, obj, pos_normalized):
        improved = (not _np.isnan(obj)) and (obj < self._tr_best_obj - 1e-12)
        if improved:
            self._tr_best_obj = obj
            self._tr_center = pos_normalized
            self._tr_success_count += 1
            self._tr_failure_count = 0
        else:
            self._tr_failure_count += 1
            self._tr_success_count = 0

        if self._tr_success_count >= self.params.trust_region_success_tol:
            self._tr_length = min(
                2.0 * self._tr_length, self.params.trust_region_length_max
            )
            self._tr_success_count = 0
        elif self._tr_failure_count >= self._tr_failure_tol:
            self._tr_length /= 2.0
            self._tr_failure_count = 0

        if self._tr_length < self.params.trust_region_length_min:
            self._restart_trust_region()

    def _restart_trust_region(self):
        dim = self.params.limit_lower.size
        self._tr_length = self.params.trust_region_scale
        self._tr_success_count = 0
        self._tr_failure_count = 0

        for _ in range(self.params.num_init_random_pts):
            if self.num_objective_evals >= self.params.max_number_evals:
                break
            pos_normalized = self._rng.uniform(0.0, 1.0, size=dim)
            pos = self.params.denormalize_positions(pos_normalized)
            obj = self._evaluate_and_register(pos)
            if not _np.isnan(obj) and obj < self._tr_best_obj:
                self._tr_best_obj = obj
                self._tr_center = pos_normalized

    def _get_global_best_normalized(self, dim):
        obj_eval = _np.array(self.objfuncs_evaluated, dtype=float)
        if obj_eval.size == 0 or _np.isnan(obj_eval).all():
            return _np.full(dim, 0.5)
        best_idx = _np.nanargmin(obj_eval)
        best_pos = _np.array(self.positions_evaluated)[best_idx]
        return self.params.normalize_positions(best_pos.reshape(1, -1))[0]

    def _get_lengthscale_weights(self, dim):
        if self._gp_model is None:
            return _np.ones(dim)
        # lengthscales = _np.atleast_1d(self._gp_model.rbf.lengthscale.values)
        lengthscales = _np.atleast_1d(
            self._gp_model.kern.parts[0].lengthscale.values
        )

        if lengthscales.size != dim:
            lengthscales = _np.full(dim, lengthscales[0])
        geo_mean = _np.exp(_np.mean(_np.log(lengthscales)))
        return lengthscales / geo_mean

    def _get_trust_region_bounds(self, dim):
        mode = self.params.trust_region_mode
        TRM = self.params.TrustRegionMode

        if mode == TRM.NoTrustRegion:
            return [(0.0, 1.0)] * dim

        if mode == TRM.FixedTrustRegion:
            center = self._get_global_best_normalized(dim)
            length = self.params.trust_region_scale
        else:  # AdaptiveTrustRegion
            center = self._tr_center
            length = self._tr_length

        weights = self._get_lengthscale_weights(dim)
        half_widths = 0.5 * length * weights
        lower = _np.clip(center - half_widths, 0.0, 1.0)
        upper = _np.clip(center + half_widths, 0.0, 1.0)
        return list(zip(lower, upper))

    def _fit_gp(self):
        pos_eval = _np.array(self.positions_evaluated)
        obj_eval = _np.array(self.objfuncs_evaluated).reshape(-1, 1)

        valid = ~_np.isnan(obj_eval).ravel()
        pos_eval = pos_eval[valid]
        obj_eval = obj_eval[valid]

        pos_eval_normalized = self.params.normalize_positions(pos_eval)
        dim = pos_eval_normalized.shape[-1]

        if self.params.trust_region_mode != (
            self.params.TrustRegionMode.NoTrustRegion
        ):
            # restrict the GP fit to points inside the current trust
            # region, falling back to the full dataset if too few remain
            bounds = self._get_trust_region_bounds(dim)
            lower = _np.array([b[0] for b in bounds])
            upper = _np.array([b[1] for b in bounds])
            inside = _np.all(
                (pos_eval_normalized >= lower)
                & (pos_eval_normalized <= upper),
                axis=1,
            )
            if inside.sum() >= max(3, dim + 1):  # Ref 1 Heuristic
                pos_eval_normalized = pos_eval_normalized[inside]
                obj_eval = obj_eval[inside]

        self._y_mean = obj_eval.mean()
        self._y_std = obj_eval.std() if obj_eval.std() > 1e-9 else 1.0
        yn = (obj_eval - self._y_mean) / self._y_std  # normalize data

        kernel = _GPy.kern.RBF(
            input_dim=dim, ARD=self.params.automatic_relevance_determination
        )  # RBF + ARD
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
        imp = y_best - mu - self.params.xi_param
        z = imp / sigma
        ei = imp * _norm.cdf(z) + sigma * _norm.pdf(z)
        return -ei  # minimize negative EI = maximize EI

    def _upper_confidence_bound(self, pos_normalized):
        mu, sigma = self._predict(pos_normalized)
        sigma = max(sigma, 1e-9)

        ucb = mu - _np.sqrt(self.params.beta_param) * sigma
        # is actually Lower Confidence Bound (LCB)
        return ucb

    def _get_acquisition_function(self):
        acq_func_type = self.params.acq_func_type
        if acq_func_type == self.params.AcqFuncType.ExpectedImprovement:
            return self._expected_improvement
        if acq_func_type == self.params.AcqFuncType.UpperConfidenceBound:
            return self._upper_confidence_bound
        raise ValueError(f'Unknown acquisition function type: {acq_func_type}')

    def _propose_next(self, dim):
        best_npos, best_val = None, _np.inf
        bounds = self._get_trust_region_bounds(dim)
        acq_func = self._get_acquisition_function()

        for _ in range(self.params.num_restarts_acq + 1):
            x0 = _np.array([self._rng.uniform(lo, hi) for lo, hi in bounds])
            res = _minimize(acq_func, x0=x0, bounds=bounds, method='L-BFGS-B')
            if res.fun < best_val:
                best_val, best_npos = res.fun, res.x

        return best_npos
