"""."""
import time as _time

import numpy as np
import numpy.polynomial.polynomial as np_poly

import pyaccel as pa
from siriuspy.devices import SOFB, DevLILLRF

from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass

from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import rcParams
rcParams.update({'font.size': 16, 'lines.linewidth': 2})


class ParamsDisp(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.delta_kly2_amp = -2
        self.wait_kly2 = 40
        self.timeout_orb = 10
        self.nr_points = 10
        # self.kly2_excit_coefs = [66.669, 1.098]  # old
        # self.kly2_excit_coefs = [71.90322743, 1.01026423]  # > 2.5nC
        self.kly2_excit_coeffs = [87.56545895, 0.80518365]  # < 2.5nC

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:}  {2:s}\n'.format
        stg = ftmp('delta_kly2_amp', self.delta_kly2_amp, '')
        stg += ftmp('wait_kly2', self.wait_kly2, '[s]')
        stg += ftmp('timeout_orb', self.timeout_orb, '[s]')
        stg += dtmp('nr_points', self.nr_points, '')
        coeffs_stg = ', '.join([f'{c:f}' for c in self.kly2_excit_coeffs])
        stg += stmp(
            'kly2_excit_coeffs', f'[{coeffs_stg:s}]', '')
        return stg


class MeasureDispTBBO(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(ParamsDisp(), isonline=isonline)
        self.isonline = isonline
        if self.isonline:
            self.devices = {
                'bo_sofb': SOFB(SOFB.DEVICES.BO),
                'tb_sofb': SOFB(SOFB.DEVICES.TB),
                'kly2': DevLILLRF(DevLILLRF.DEVICES.LI_KLY2)}

    @property
    def energy(self):
        """."""
        return np_poly.polyval(
            self.params.kly2_excit_coeffs, self.devices['kly2'].amplitude)

    @property
    def trajx(self):
        """."""
        return np.hstack(
            [self.devices['tb_sofb'].trajx, self.devices['bo_sofb'].trajx])

    @property
    def trajy(self):
        """."""
        return np.hstack(
            [self.devices['tb_sofb'].trajy, self.devices['bo_sofb'].trajy])

    @property
    def nr_points(self):
        """."""
        return min(
            self.devices['tb_sofb'].nr_points,
            self.devices['bo_sofb'].nr_points)

    @nr_points.setter
    def nr_points(self, value):
        self.devices['tb_sofb'].nr_points = int(value)
        self.devices['bo_sofb'].nr_points = int(value)

    def wait(self, timeout=10):
        """."""
        self.devices['tb_sofb'].wait_buffer(timeout=timeout)
        self.devices['bo_sofb'].wait_buffer(timeout=timeout)

    def reset(self, wait=0):
        """."""
        _time.sleep(wait)
        self.devices['tb_sofb'].cmd_reset()
        self.devices['bo_sofb'].cmd_reset()
        _time.sleep(1)

    def measure_dispersion(self):
        """."""
        self.nr_points = self.params.nr_points
        delta = self.params.delta_kly2_amp
        kly2_dev = self.devices['kly2']

        print('reading initial trajectory...')
        self.reset(3)
        self.wait(self.params.timeout_orb)
        orb = [-np.hstack([self.trajx, self.trajy]), ]
        ene0 = self.energy

        print('setting new Klystron2 amplitude...')
        origamp = kly2_dev.amplitude
        kly2_dev.amplitude = origamp + delta
        print(f"Klystron2 Amp: {kly2_dev.amplitude:.3f}")

        self.reset(self.params.wait_kly2)
        print('reading new trajectory...')
        self.reset(3)
        self.wait(self.params.timeout_orb)
        orb.append(np.hstack([self.trajx, self.trajy]))
        ene1 = self.energy

        print('restoring Klystron2 amplitude...')
        kly2_dev.amplitude = origamp
        print(f"Klystron2 Amp: {kly2_dev.amplitude:.3f}")
        d_ene = ene1/ene0 - 1
        print('finished!')
        return np.array(orb).sum(axis=0) / d_ene

    @staticmethod
    def calc_model_dispersionTBBO(model, bpms):
        """."""
        dene = 1e-4
        rin = np.array([
            [0, 0, 0, 0, dene/2, 0],
            [0, 0, 0, 0, -dene/2, 0]]).T
        rout, *_ = pa.tracking.line_pass(model, rin, bpms)
        dispx = (rout[0, 0, :] - rout[0, 1, :]) / dene
        dispy = (rout[2, 0, :] - rout[2, 1, :]) / dene
        return np.hstack([dispx, dispy])

    @staticmethod
    def set_septum_gradient(model, kxl, kyl, ksxl, ksyl):
        """."""
        ind = pa.lattice.find_indices(model, 'fam_name', 'InjSeptM66')
        nrsegs = len(ind)
        for i in ind:
            model[i].KxL = kxl/nrsegs
            model[i].KyL = kyl/nrsegs
            model[i].KsxL = ksxl/nrsegs
            model[i].KsyL = ksyl/nrsegs

    @staticmethod
    def get_septum_gradient(model):
        """."""
        ind = pa.lattice.find_indices(model, 'fam_name', 'InjSeptM66')
        kxl, kyl, ksxl, ksyl = 0, 0, 0, 0
        for i in ind:
            kxl += model[i].KxL
            kyl += model[i].KyL
            ksxl += model[i].KsxL
            ksyl += model[i].KsyL
        return kxl, kyl, ksxl, ksyl

    @staticmethod
    def err_func(grads, model, disp_meas):
        """."""
        kxl, kyl, ksxl, ksyl = grads
        MeasureDispTBBO.set_septum_gradient(model, kxl, kyl, ksxl, ksyl)
        bpm_idx = np.array(pa.lattice.find_indices(
            model, 'fam_name', 'BPM')).ravel()[1:]
        disp_model = MeasureDispTBBO.calc_model_dispersionTBBO(model, bpm_idx)
        err_vec = (disp_model - disp_meas)**2
        return err_vec

    @staticmethod
    def fit_septum_gradients(model, disp_meas, x0=None, bounds=None):
        """."""
        if x0 is None:
            x0 = MeasureDispTBBO.get_septum_gradient(model)

        if bounds is None:
            fit_grads = least_squares(
                fun=MeasureDispTBBO.err_func,
                x0=x0, args=(model, disp_meas), method='lm')
        else:
            fit_grads = least_squares(
                fun=MeasureDispTBBO.err_func, x0=x0,
                args=(model, disp_meas), bounds=bounds)
        return fit_grads

    @staticmethod
    def calc_fitting_error(fit_grads):
        """Least squares fitting error.

        Based on fitting error calculation of scipy.optimization.curve_fit
        do Moore-Penrose inverse discarding zero singular values.
        """
        _, smat, vhmat = np.linalg.svd(
            fit_grads['jac'], full_matrices=False)
        thre = np.finfo(float).eps * max(fit_grads['jac'].shape)
        thre *= smat[0]
        smat = smat[smat > thre]
        vhmat = vhmat[:smat.size]
        pcov = np.dot(vhmat.T / (smat*smat), vhmat)

        # multiply covariance matrix by residue 2-norm
        ysize = len(fit_grads['fun'])
        cost = 2 * fit_grads['cost']  # res.cost is half sum of squares!
        popt = fit_grads['x']
        if ysize > popt.size:
            # normalized by degrees of freedom
            s_sq = cost / (ysize - popt.size)
            pcov = pcov * s_sq
        else:
            pcov.fill(np.nan)
            print(
                '# of fitting parameters larger than # of data points!')
        return np.sqrt(np.diag(pcov))

    @staticmethod
    def plot_dispersion(disp_model, disp_meas, nr_bpms):
        """."""
        _, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(
            disp_model[:nr_bpms], '-o', color='tab:blue', label='model')
        axs[0].plot(
            disp_meas[:nr_bpms], 'o--', color='b', alpha=0.75, label='meas')

        axs[1].plot(disp_model[nr_bpms:], '-o', color='tab:red', label='model')
        axs[1].plot(
            disp_meas[nr_bpms:],
            'o--', color='C1', alpha=0.75, label='meas')

        axs[0].axvline(6-1/2, ls='--', color='k')
        axs[1].axvline(6-1/2, ls='--', color='k')

        axs[0].set_ylabel(r'$\eta_x$ [m]')
        axs[1].set_ylabel(r'$\eta_y$ [m]')
        axs[1].set_xlabel('BPM idx')
        axs[0].set_title('Propagated dispersion function TB-BO')

        axs[0].legend(fontsize=10)
        axs[1].legend(fontsize=10)

        plt.tight_layout()
        plt.show()
