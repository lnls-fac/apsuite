"""."""

import time as _time

import numpy as np
import numpy.polynomial.polynomial as np_poly

import pyaccel as pa
from siriuspy.devices import SOFB, DevLILLRF, EVG
from pymodels import tb, bo

from ..utils import (
    MeasBaseClass as _BaseClass,
    ParamsBaseClass as _ParamsBaseClass,
)

from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams.update({'font.size': 16, 'lines.linewidth': 2})


class ParamsDisp(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.kly2_amp_ini = 69.7
        self.kly2_amp_fin = 73.7
        self.kly2_amp_npts = 5
        self.wait_kly2 = 10
        self.timeout_orb = 10
        self.nr_points = 10
        self.injection_interval = 30
        # self.kly2_excit_coefs = [66.669, 1.098]  # old
        # self.kly2_excit_coefs = [71.90322743, 1.01026423]  # > 2.5nC
        self.kly2_excit_coeffs = [87.56545895, 0.80518365]  # < 2.5nC

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:}  {2:s}\n'.format
        stg = ftmp('kly2_amp_ini', self.kly2_amp_ini, '')
        stg += ftmp('kly2_amp_fin', self.kly2_amp_fin, '')
        stg += dtmp('kly2_amp_npts', self.kly2_amp_npts, '')
        stg += ftmp('wait_kly2', self.wait_kly2, '[s]')
        stg += ftmp('timeout_orb', self.timeout_orb, '[s]')
        stg += ftmp('injection_interval', self.injection_interval, '[s]')
        stg += dtmp('nr_points', self.nr_points, '')
        coeffs_stg = ', '.join([f'{c:f}' for c in self.kly2_excit_coeffs])
        stg += stmp('kly2_excit_coeffs', f'[{coeffs_stg:s}]', '')
        return stg


class MeasureDispTBBO(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(ParamsDisp(), isonline=isonline)
        self.isonline = isonline
        self._model = None
        self._model_bpms_idx = None
        if self.isonline:
            self.devices = {
                'bo_sofb': SOFB(SOFB.DEVICES.BO),
                'tb_sofb': SOFB(SOFB.DEVICES.TB),
                'kly2': DevLILLRF(DevLILLRF.DEVICES.LI_KLY2),
                'evg': EVG(),
            }

    @property
    def model(self):
        """."""
        if self._model is None:
            mod, *_ = tb.create_accelerator()
            modb = bo.create_accelerator()
            mod.extend(modb)
            self._model = mod
            self._model_bpms_idx = np.array(
                pa.lattice.find_indices(self._model, 'fam_name', 'BPM')
            ).ravel()[1:]
        return self._model

    @property
    def model_bpms_idx(self):
        """."""
        if self._model_bpms_idx is None:
            _ = self.model  # to populate bpms_idx
        return self._model_bpms_idx

    @property
    def energy(self):
        """."""
        return np_poly.polyval(
            self.params.kly2_excit_coeffs, self.devices['kly2'].amplitude
        )

    @property
    def trajx(self):
        """."""
        return np.hstack([
            self.devices['tb_sofb'].trajx,
            self.devices['bo_sofb'].trajx,
        ])

    @property
    def trajy(self):
        """."""
        return np.hstack([
            self.devices['tb_sofb'].trajy,
            self.devices['bo_sofb'].trajy,
        ])

    @property
    def trajsum(self):
        """."""
        return np.hstack([
            self.devices['tb_sofb'].sum,
            self.devices['bo_sofb'].sum,
        ])

    @property
    def traj(self):
        """."""
        return np.r_[self.trajx, self.trajy]

    def inject_and_get_data(self):
        """."""
        evg = self.devices['evg']
        trajs = []
        timestamp = []
        trajsum = []
        kly2_amp = []
        for i in range(self.params.nr_points):
            traj0 = self.traj
            evg.cmd_turn_on_injection()
            t0_ = _time.time()
            stg = f'    {i + 1:02d}/{self.params.nr_points:02d} -> '
            stg += 'Getting trajectory...'
            print(stg, end='\r', flush=True)
            for _ in range(50):
                if not np.any(np.isclose(traj0, self.traj)):
                    break
                _time.sleep(self.params.timeout_orb / 50)
            else:
                stg += ' Timed out waiting traj to update.'
            print(stg + '  Done!')
            trajs.append(self.traj)
            trajsum.append(self.trajsum)
            timestamp.append(_time.time())
            kly2_amp.append(self.devices['kly2'].amplitude)
            dtim = max(
                0, self.params.injection_interval - (_time.time() - t0_)
            )
            _time.sleep(dtim)
        return dict(
            trajs=trajs,
            trajsum=trajsum,
            timestamp=timestamp,
            kly2_amp=kly2_amp,
        )

    def measure_dispersion(self):
        """."""
        kly2_amps = np.linspace(
            self.params.kly2_amp_ini,
            self.params.kly2_amp_fin,
            self.params.kly2_amp_npts,
        )
        kly2_dev = self.devices['kly2']
        self.data = []

        origamp = kly2_dev.amplitude
        for i, kly2_amp in enumerate(kly2_amps):
            kly2_dev.amplitude = kly2_amp
            _time.sleep(0.5)
            print(
                f'{i + 1:02d}/{self.params.kly2_amp_npts:02d} --> '
                f'Klystron2 Amp: {kly2_amp:.3f}'
            )
            _time.sleep(self.params.wait_kly2)

            self.data.append(self.inject_and_get_data())

        print('Restoring Klystron2 amplitude...')
        kly2_dev.amplitude = origamp
        _time.sleep(1)
        print(f'Klystron2 Amp: {kly2_dev.amplitude:.3f}')
        print('Finished!')

    def process_data(self, fit_order=1, norm_strategy='bo_mean'):
        """."""
        if not self.data:
            raise ValueError(
                'No data to process. Run measure_dispersion() first.'
            )
        trajs = []
        kly2_amps = []
        for datum in self.data:
            trajs.extend(datum['trajs'])
            kly2_amps.extend(datum['kly2_amp'])
        trajs = np.array(trajs)
        kly2_amps = np.array(kly2_amps)
        xfit = kly2_amps - kly2_amps.mean()
        coefs, _ = np.polynomial.polynomial.polyfit(
            xfit, trajs, deg=fit_order, full=True
        )

        disp_model = self.calc_model_dispersion()
        disp_meas = coefs[1:].copy()
        if norm_strategy.lower().startswith('tb'):
            disp_meas *= disp_model[:6].std() / coefs[1][:6].std()
        else:
            disp_meas *= disp_model[6:56].mean() / coefs[1][6:56].mean()

        ress = [(trajs**2).sum(axis=0)]
        for i in range(1, fit_order+2):
            fit = np.polynomial.polynomial.polyval(xfit, coefs[:i])
            ress.append(((trajs - fit.T)**2).sum(axis=0))
        ress = np.array(ress)
        ratio = ress / ress[1][None, :]

        self.analysis = dict(
            fit_x=xfit,
            fit_coefs=coefs,
            fit_residue_order=ress,
            fit_rel_residue=ratio,
            disp_meas=disp_meas,
            disp_model=disp_model,
            trajs=trajs,
            kly2_amps=kly2_amps,
        )

    def get_dispersion_fitting_problems(
        self, order=1, rel_residue_threshold=0.01
    ):
        ratio = self.analysis['fit_rel_residue']
        idcs = (ratio[order+1] > rel_residue_threshold).nonzero()[0]
        nbpms = len(self.model_bpms_idx)
        fit_probs = [(i, 'h') for i in idcs if i < nbpms]
        fit_probs += [(i - nbpms, 'v') for i in idcs if i >= nbpms]
        return fit_probs

    def calc_model_dispersion(self):
        """."""
        dene = 1e-4
        rin = np.array([
            [0, 0, 0, 0, dene / 2, 0],
            [0, 0, 0, 0, -dene / 2, 0],
        ]).T
        rout, *_ = pa.tracking.line_pass(
            self.model, rin, self._model_bpms_idx
        )
        dispx = (rout[0, 0, :] - rout[0, 1, :]) / dene
        dispy = (rout[2, 0, :] - rout[2, 1, :]) / dene
        return np.hstack([dispx, dispy])

    def set_septum_gradient(self, kxl, kyl, ksxl, ksyl):
        """."""
        ind = pa.lattice.find_indices(self.model, 'fam_name', 'InjSeptM66')
        nrsegs = len(ind)
        for i in ind:
            self.model[i].KxL = kxl / nrsegs
            self.model[i].KyL = kyl / nrsegs
            self.model[i].KsxL = ksxl / nrsegs
            self.model[i].KsyL = ksyl / nrsegs

    def get_septum_gradient(self):
        """."""
        ind = pa.lattice.find_indices(self.model, 'fam_name', 'InjSeptM66')
        kxl, kyl, ksxl, ksyl = 0, 0, 0, 0
        for i in ind:
            kxl += self.model[i].KxL
            kyl += self.model[i].KyL
            ksxl += self.model[i].KsxL
            ksyl += self.model[i].KsyL
        return kxl, kyl, ksxl, ksyl

    def fit_septum_gradients(self, x0=None, bounds=None):
        """."""
        if not self.analysis:
            raise ValueError(
                'No analysis available. Run process_data() first.'
            )
        if x0 is None:
            x0 = self.get_septum_gradient()

        kwgs = dict(fun=self._err_func, x0=x0)
        if bounds is None:
            kwgs['method'] = 'lm'
        else:
            kwgs['bounds'] = bounds
        return least_squares(**kwgs)

    @staticmethod
    def calc_fitting_error(fit_grads):
        """Least squares fitting error.

        Based on fitting error calculation of scipy.optimization.curve_fit
        do Moore-Penrose inverse discarding zero singular values.
        """
        _, smat, vhmat = np.linalg.svd(fit_grads['jac'], full_matrices=False)
        thre = np.finfo(float).eps * max(fit_grads['jac'].shape)
        thre *= smat[0]
        smat = smat[smat > thre]
        vhmat = vhmat[: smat.size]
        pcov = np.dot(vhmat.T / (smat * smat), vhmat)

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
            print('# of fitting parameters larger than # of data points!')
        return np.sqrt(np.diag(pcov))

    # ---------------- plot methods ----------------

    def plot_dispersion(self, order=1, nr_bpms=None):
        """."""
        if nr_bpms is None:
            nr_bpms = len(self.model_bpms_idx)

        disp_model = self.calc_model_dispersion()
        disp_meas = self.analysis['disp_meas'][order-1].copy()

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        axs[0].plot(
            disp_model[:nr_bpms], '-o', color='tab:blue', label='model'
        )
        axs[0].plot(
            disp_meas[:nr_bpms], 'o--', color='b', alpha=0.75, label='meas'
        )

        axs[1].plot(disp_model[nr_bpms:], '-o', color='tab:red', label='model')
        axs[1].plot(
            disp_meas[nr_bpms:], 'o--', color='C1', alpha=0.75, label='meas'
        )

        axs[0].axvline(6 - 1 / 2, ls='--', color='k')
        axs[1].axvline(6 - 1 / 2, ls='--', color='k')

        axs[0].set_ylabel(r'$\eta_x$ [m]')
        axs[1].set_ylabel(r'$\eta_y$ [m]')
        axs[1].set_xlabel('BPM idx')
        axs[0].set_title(
            'Propagated Dispersion Function TB-BO (order {order})'
        )

        axs[0].legend(fontsize=10)
        axs[1].legend(fontsize=10)
        axs[0].grid(True, alpha=0.5, ls='--', lw=0.5, color='k')
        axs[1].grid(True, alpha=0.5, ls='--', lw=0.5, color='k')

        fig.tight_layout()
        return fig, axs

    def plot_traj_fitting_relative_residue(self, order=1):
        """."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ratio = self.analysis['fit_rel_residue'][order+1]
        nbpm = len(self.model_bpms_idx)

        ax.plot(ratio[:nbpm], '-o', label='Horizontal')
        ax.plot(ratio[nbpm:], '-o', label='Vertical')

        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1),
            ncol=2,
            fontsize='small'
        )
        ax.set_title('Relative Residue Fit Order N={order} by Order 0.')
        ax.set_xlabel('BPM Index')
        ax.set_ylabel(
            r'Relative residue $\chi^2_{y=P_N(x)}/\chi^2_{y=P_0(x)}$'
        )
        ax.grid(True, ls='--', alpha=0.4, color='k', lw=0.5)

        fig.tight_layout()
        return fig, ax

    def plot_dispersion_fit_at_bpm(self, bpm_idx=0, plane='h'):
        """."""
        ish = plane.lower().startswith(('h', 'x'))
        idx = bpm_idx
        if not ish:
            idx += len(self.model_bpms_idx)

        ratio = self.analysis['fit_rel_residue']
        kly2_amps = self.analysis['kly2_amps']
        xfit = self.analysis['fit_x']
        traj_points = self.analysis['trajs'][:, idx]
        coefs = self.analysis['fit_coefs'][:, idx]
        traj_fit = np.polynomial.polynomial.polyval(xfit, coefs)

        fig, ax = plt.subplots(figsize=(8, 5))

        stg = f'BPM {bpm_idx:d}, '
        stg += f"{'Horizontal' if ish else 'Vertical':s} Plane\n"
        stg += 'coefs = ['
        stg += ', '.join([f'{r:.2g}'for r in coefs])
        stg += ']    ratios = ['
        stg += ', '.join([f'{r:.2g}'for r in ratio[2:, idx]])
        stg += ']'
        ax.set_title(stg, fontsize='small')

        ax.plot(kly2_amps, traj_points, "o", label='Data')
        ax.plot(kly2_amps, traj_fit, label='Fit')
        ax.legend(loc='best')
        ax.set_xlabel('Klystron 2 Amplitude [%]')
        ax.set_ylabel('Trajectory [um]')
        ax.grid(True, ls='--', alpha=0.4, color='k', lw=0.5)

        fig.tight_layout()
        return fig, ax

    # --------------- helper methods ---------------

    def _err_func(self, grads):
        """."""
        disp_meas = self.analysis['disp_meas']
        kxl, kyl, ksxl, ksyl = grads
        self.set_septum_gradient(kxl, kyl, ksxl, ksyl)
        disp_model = self.calc_model_dispersion()
        err_vec = (disp_model - disp_meas) ** 2
        return err_vec
