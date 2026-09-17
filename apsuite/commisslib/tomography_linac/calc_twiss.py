"""Determine Twiss parameters from tomography data."""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

from scipy.stats import multivariate_normal
from scipy.optimize import least_squares, curve_fit


class TwissFromDistribution:
    """Emittance and Twiss parameters from Gaussian fitting.

    Parameters
    ----------
    reconstruction : DistribReconstruction
        Reconstruction object containing the computed distribution.
    """

    def __init__(self, reconstruction):
        """."""
        self.rec = reconstruction

        self.avg = None
        self.cov = None
        self.emit = None
        self.twiss = None
        self.fit_results = None

        self._process()

    def _process(self):

        gridx = self.rec.model_gridx  # [mm]
        gridy = self.rec.model_gridy  # [mrad]

        distrib = self.rec.get_model_distribution(raveled=False)

        res = self.fit_2d_gaussian(gridx, gridy, distrib)

        # Parameters and Twiss - SI units
        avg = res.x[:2] * 1e-3
        second_mom = res.x[2:] * 1e-6
        cov = self.get_cov_from_par(second_mom)

        if np.linalg.det(cov) <= 0:
            text = 'The fitted beam covariance matrix is not physical '
            text += '(det(cov) <= 0).'
            raise ValueError(text)

        emit = self.get_emit_from_cov(cov)
        twiss = self.get_twiss_from_cov(cov)  # (alpha, beta, gamma)

        # Fitted parameters errors
        params_cov_errors = self.calc_fit_params_err(res)

        # Fitted parameters errors - SI units
        avg_err = np.sqrt(np.diag(params_cov_errors[:2, :2])) * 1e-3
        second_mom_cov_err = params_cov_errors[2:, 2:] * 1e-12
        second_mom_err = np.sqrt(np.diag(second_mom_cov_err))
        cov_err = self.get_cov_from_par(second_mom_err)

        # Propagated erros
        second_mom_samples = self.sample_params(second_mom, second_mom_cov_err)
        valid = self._check_sampled_params(second_mom_samples)
        second_mom_samples = second_mom_samples[:, valid]

        emit_samples = self.get_emit_from_params(*second_mom_samples)
        twiss_samples = self.get_twiss_from_params(*second_mom_samples)

        emit_err = np.std(emit_samples, ddof=1)
        twiss_err = np.std(twiss_samples, axis=1, ddof=1)

        self.avg = [avg, avg_err]
        self.cov = [cov, cov_err]
        self.emit = [emit, emit_err]
        self.twiss = [twiss, twiss_err]
        self.fit_results = res

    def fit_2d_gaussian(self, gridx, gridy, distrib):
        """."""
        deltax = gridx[0, 1] - gridx[0, 0]
        deltay = gridy[1, 0] - gridy[0, 0]
        distrib = distrib / np.sum(distrib) / deltax / deltay  # Normalized

        def err_func(x):
            avg = x[:2]
            cov = self.get_cov_from_par(x[2:])
            gauss = self.bivariate_gaussian(avg, cov, gridx, gridy)
            return distrib.ravel() - gauss.ravel()

        x0 = np.array([0, 0, 1, 1, 0])  # x, x', <x^2>, <x'^2>, <xx'>

        res = least_squares(err_func, x0)

        return res

    @staticmethod
    def get_cov_from_par(par):
        """."""
        return np.diag(par[:2]) + np.fliplr(np.eye(2)) * par[2]

    @staticmethod
    def bivariate_gaussian(mean, cov, gridx, gridy):
        """."""
        try:
            data = np.vstack([gridx.ravel(), gridy.ravel()]).T
            vals1 = multivariate_normal.pdf(data, mean=mean, cov=cov)
        except Exception:
            vals1 = gridx * 0
        return vals1.reshape(*gridx.shape)

    @staticmethod
    def get_emit_from_cov(cov):
        """."""
        return np.sqrt(np.linalg.det(cov))

    @staticmethod
    def get_twiss_from_cov(cov):
        """."""
        emit = np.sqrt(np.linalg.det(cov))
        beta = cov[..., 0, 0] / emit
        gamma = cov[..., 1, 1] / emit
        alpha = -cov[..., 0, 1] / emit
        return np.array([alpha, beta, gamma])

    @staticmethod
    def get_sigmas_from_cov(cov):
        """."""
        eigvals, eigvecs = np.linalg.eig(cov)
        idx = np.argmax(eigvals)
        largest_evec = eigvecs[:, idx]

        angle = np.rad2deg(np.arctan2(*largest_evec[::-1]))
        sigma2, sigma1 = np.sqrt(np.sort(eigvals))

        return sigma1, sigma2, angle

    @staticmethod
    def calc_fit_params_err(res):
        """."""
        jac = res.jac
        nr_points, nr_params = jac.shape
        _, s_vals, vt_mat = np.linalg.svd(jac, full_matrices=False)

        threshold = np.finfo(float).eps * max(nr_points, nr_params) * s_vals[0]
        mask = s_vals > threshold

        s_vals = s_vals[mask]
        vt_mat = vt_mat[: s_vals.size]

        # Covariance matrix
        hess_inv = vt_mat.T @ np.diag(1 / s_vals**2) @ vt_mat
        cost = 2 * res.cost
        var = cost / (nr_points - nr_params)
        cov_mat = var * hess_inv

        return cov_mat

    @staticmethod
    def sample_params(params, covariance, nr_samples=100000, seed=None):
        """."""
        params = np.asarray(params, dtype=float)
        covariance = np.asarray(covariance, dtype=float)

        rng = np.random.default_rng(seed=seed)
        samples = rng.multivariate_normal(
            mean=params, cov=covariance, size=nr_samples
        ).T
        return samples

    @staticmethod
    def _check_sampled_params(samples):
        a, b, c = samples
        emit2_samples = a * b - c**2

        valid = emit2_samples > 0
        frac_invalid = np.mean(~valid)

        if frac_invalid > 0.05:
            text = 'Monte Carlo uncertainty propagation failed: '
            text += f'{frac_invalid:.1%} of samples are physically invalid '
            text += '(emittance² <= 0).'
            raise ValueError(text)

        return valid

    @classmethod
    def get_emit_from_params(cls, a, b, c):
        """."""
        cov = np.array([[a, c], [a, b]]).transpose(2, 0, 1)
        return cls.get_emit_from_cov(cov)

    @classmethod
    def get_twiss_from_params(cls, a, b, c):
        """."""
        cov = np.array([[a, c], [a, b]]).transpose(2, 0, 1)
        return cls.get_twiss_from_cov(cov)

    @property
    def alpha(self):
        """."""
        return [self.twiss[0][0], self.twiss[1][0]]

    @property
    def beta(self):
        """."""
        return [self.twiss[0][1], self.twiss[1][1]]

    @property
    def gamma(self):
        """."""
        return [self.twiss[0][2], self.twiss[1][2]]

    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    def __str__(self):
        """."""
        emit, emit_err = self.emit
        alpha, beta, gamma = self.twiss[0]
        alpha_err, beta_err, gamma_err = self.twiss[1]

        text = []
        text.append(
            'emittance = '
            + self.format_value_error(emit * 1e9, emit_err * 1e9)
            + ' nm.rad'
        )
        text.append('alpha     = ' + self.format_value_error(alpha, alpha_err))
        text.append(
            'beta      = ' + self.format_value_error(beta, beta_err) + ' m'
        )
        text.append(
            'gamma     = ' + self.format_value_error(gamma, gamma_err) + ' 1/m'
        )

        return '\n'.join(text)

    @staticmethod
    def format_value_error(value, error):
        """."""
        if error == 0:
            return f'{value} +- 0'

        decimals = -int(np.floor(np.log10(abs(error)))) + 1
        decimals = max(0, decimals)

        value_fmt = f'{value:.{decimals}f}'
        error_fmt = f'{error:.{decimals}f}'

        return f'{value_fmt} +- {error_fmt}'

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------

    def plot_fitting(self, ax=None, cmap='jet'):
        """."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        gridx = self.rec.model_gridx  # [mm]
        gridy = self.rec.model_gridy  # [mm]

        distrib = self.rec.get_model_distribution(raveled=False)

        ax.pcolormesh(gridx, gridy, distrib, cmap=cmap)

        sigma1, sigma2, angle = self.get_sigmas_from_cov(
            self.cov[0] * 1e6
        )  # [mm]

        ellipse = mpatch.Ellipse(
            xy=self.avg[0],
            width=2 * sigma1,
            height=2 * sigma2,
            angle=angle,
            color='r',
            fill=False,
            lw=2,
        )

        ax.add_patch(ellipse)

        emit, emit_err = self.emit
        alpha, beta, gamma = self.twiss[0]
        alpha_err, beta_err, gamma_err = self.twiss[1]

        text_emit = self.format_value_error(emit * 1e9, emit_err * 1e9)
        text_alpha = self.format_value_error(alpha, alpha_err)
        text_beta = self.format_value_error(beta, beta_err)
        text_gamma = self.format_value_error(gamma, gamma_err)

        text = (
            rf'$\epsilon$={text_emit} nm.rad'
            '\n'
            rf'$\alpha$={text_alpha}'
            '\n'
            rf'$\beta$={text_beta} m'
            '\n'
            rf'$\gamma$={text_gamma} 1/m'
        )

        ax.annotate(
            text,
            xy=(0.05, 0.95),
            fontsize=11,
            xycoords='axes fraction',
            ha='left',
            va='top',
            color='white',
        )

        ax.set_xlabel('Position [mm]')
        ax.set_ylabel('Angle [mrad]')

        return fig, ax


class QuadrupoleScan:
    """Emittance and Twiss parameters from quadrupole scan.

    Parameters
    ----------
    kls : array_like
        Quadrupole integrated strengths [1/m].
    sizes : array_like
        Measured beam sizes [mm].
    """

    DRIFT = 2.8775  # [m]

    def __init__(self, kls, sizes):
        """."""
        self.kls = np.asarray(kls, dtype=float)  # [1/m]
        self.sizes = np.asarray(sizes, dtype=float) * 1e-3  # [m]

        if self.sizes.shape != self.kls.shape:
            raise ValueError('Incompatible shape between kls and sizes.')

        popt, pcov = self._fit()
        self.fit_params = popt
        self.fit_covariance = pcov

        # Emittance and Twiss parameters - SI units
        emit, twiss = self._calc_parameters()
        emit_err, twiss_err = self._calc_errors()
        self.emit = [emit, emit_err]
        self.twiss = [twiss, twiss_err]

    @staticmethod
    def _parabola(kl, a, b, c):
        """."""
        return a * kl**2 + b * kl + c

    def _fit(self):
        """."""
        pos_sec_moms = self.sizes**2

        pos_sec_moms = pos_sec_moms.ravel()
        kls = self.kls.ravel()

        popt, pcov = curve_fit(self._parabola, kls, pos_sec_moms)
        return popt, pcov

    def _calc_parameters(self):
        """Calculate second moments, emittance and Twiss parameters."""
        params = self.fit_params
        second_moments = self.calc_second_moments(params)

        xx, xlxl, xxl = second_moments
        emit2 = xx * xlxl - xxl**2

        if emit2 < 0:
            raise ValueError(
                'The fitted second-moment matrix is not physical.'
            )

        emit = self.calc_emit(*second_moments)
        twiss = self.calc_twiss(*second_moments)
        return emit, twiss

    def _calc_errors(self, nr_samples=100000, seed=None):
        """Propagate fit uncertainties using Monte Carlo sampling."""
        rng = np.random.default_rng(seed=seed)

        fit_samples = rng.multivariate_normal(
            mean=self.fit_params, cov=self.fit_covariance, size=nr_samples
        ).T

        sec_mom_samples = self.calc_second_moments(fit_samples)
        valid = self._check_sampled_params(sec_mom_samples)
        sec_mom_samples = sec_mom_samples[:, valid]

        emit_samples = self.calc_emit(*sec_mom_samples)
        twiss_samples = self.calc_twiss(*sec_mom_samples)

        emit_err = np.std(emit_samples, ddof=1)
        twiss_err = np.std(twiss_samples, axis=1, ddof=1)

        return emit_err, twiss_err

    @staticmethod
    def calc_second_moments(params):
        """Second moments at entrance of quadrupole."""
        drift = QuadrupoleScan.DRIFT
        # Relation between parameters and second moments
        mat = np.array(
            [
                [drift**2, 0, 0],
                [-2 * drift, 0, -2 * drift**2],
                [1, drift**2, 2 * drift],
            ]
        )
        mat_inv = np.linalg.inv(mat)
        second_moments = mat_inv @ params  # [<x^2>, <x'^2>, <xx'>]
        return second_moments

    @staticmethod
    def calc_emit(xx, xlxl, xxl):
        """."""
        return np.sqrt(xx * xlxl - xxl**2)

    @staticmethod
    def calc_twiss(xx, xlxl, xxl):
        """."""
        emit = np.sqrt(xx * xlxl - xxl**2)
        alpha = -xxl / emit
        beta = xx / emit
        gamma = xlxl / emit
        return np.array([alpha, beta, gamma])

    @staticmethod
    def _check_sampled_params(samples):
        a, b, c = samples
        emit2_samples = a * b - c**2

        valid = emit2_samples > 0
        frac_invalid = np.mean(~valid)

        if frac_invalid > 0.05:
            text = 'Monte Carlo uncertainty propagation failed: '
            text += f'{frac_invalid:.1%} of samples are physically invalid '
            text += '(emittance² <= 0).'
            raise ValueError(text)

        return valid

    @property
    def alpha(self):
        """."""
        return [self.twiss[0][0], self.twiss[1][0]]

    @property
    def beta(self):
        """."""
        return [self.twiss[0][1], self.twiss[1][1]]

    @property
    def gamma(self):
        """."""
        return [self.twiss[0][2], self.twiss[1][2]]

    # ---------------------------------------------------------
    # PRINT
    # ---------------------------------------------------------

    def __str__(self):
        """."""
        emit, emit_err = self.emit
        alpha, beta, gamma = self.twiss[0]
        alpha_err, beta_err, gamma_err = self.twiss[1]

        text = []
        text.append(
            'emittance = '
            + self.format_value_error(emit * 1e9, emit_err * 1e9)
            + ' nm.rad'
        )
        text.append('alpha     = ' + self.format_value_error(alpha, alpha_err))
        text.append(
            'beta      = ' + self.format_value_error(beta, beta_err) + ' m'
        )
        text.append(
            'gamma     = ' + self.format_value_error(gamma, gamma_err) + ' 1/m'
        )

        return '\n'.join(text)

    @staticmethod
    def format_value_error(value, error):
        """."""
        if error == 0:
            return f'{value} +- 0'

        decimals = -int(np.floor(np.log10(abs(error)))) + 1
        decimals = max(0, decimals)

        value_fmt = f'{value:.{decimals}f}'
        error_fmt = f'{error:.{decimals}f}'

        return f'{value_fmt} +- {error_fmt}'

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------

    def plot_fitting(self, ax=None):
        """."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 5))
        else:
            fig = ax.figure

        kls_mean = np.mean(self.kls, axis=1)
        sizes_data = self.sizes
        sizes_mean = np.mean(sizes_data, axis=1)
        sizes_std = np.std(sizes_data, axis=1, ddof=1)

        params = self.fit_params
        kls_plot = np.linspace(kls_mean.min(), kls_mean.max(), 1000)

        sec_mon_plot = self._parabola(kls_plot, *params)
        sizes_plot = np.sqrt(sec_mon_plot)

        ax.errorbar(
            kls_mean,
            sizes_mean * 1e3,
            yerr=sizes_std * 1e3,
            fmt='o',
            markersize=4,
            capsize=3,
            color='C0',
            label='Data',
        )
        ax.plot(kls_plot, sizes_plot * 1e3, '--', color='C0', label='Fit')
        ax.set_xlabel('Kl [1/m]')
        ax.set_ylabel('Beam size [mm]')
        ax.legend()

        return fig, ax
