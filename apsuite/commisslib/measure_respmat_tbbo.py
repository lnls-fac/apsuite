"""."""

import time as _time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import pyaccel as pa
from pymodels import li, tb, bo

# from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupply as _PowerSupply
from siriuspy.devices import EVG as _EVG
from siriuspy.devices import SOFB as _SOFB
from siriuspy.devices import DCCT as _DCCT
from siriuspy.search import PSSearch as _PSSearch
from threading import Event as _Event
from scipy.optimize import least_squares

# from ..optimization import SimulAnneal
from ..utils import (
    ThreadedMeasBaseClass as _BaseClass,
    ParamsBaseClass as _ParamsBaseClass,
)

rcParams.update({'font.size': 16, 'lines.linewidth': 2})


class Params(_ParamsBaseClass):
    """."""

    ALL_CORRS = tuple(
        _PSSearch.get_psnames({'sec': 'LI', 'dev': 'CH', 'idx': '7'})
        + _PSSearch.get_psnames({'sec': 'TB', 'dev': 'CH'})
        + _PSSearch.get_psnames({'sec': 'TB', 'dev': 'InjSept'})
        + _PSSearch.get_psnames({'sec': 'BO', 'dev': 'InjKckr'})
        + _PSSearch.get_psnames({'sec': 'LI', 'dev': 'CV', 'idx': '7'})
        + _PSSearch.get_psnames({'sec': 'TB', 'dev': 'CV'})
    )
    DLIMS = tuple(
        [50]  # LI-CH-7 [urad]
        + [200] * 6  # TB-CH [urad]
        + [0.1]  # InjSept [mrad]
        + [0.1]  # InjKckr [mrad]
        + [50]  # LI-CV-7 [urad]
        + [200] * 6  # TB-CV [urad]
    )

    def __init__(self):
        """."""
        super().__init__()
        self.corr_nrpts = 5
        self.corr_wait = 0.5  # [s]
        self.injection_interval = 3  # [s]
        self.timeout_orb = 10  # [s]
        self.nr_points = 10
        self.corrs2measure = list(Params.ALL_CORRS)

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        ttmp = '{0:24s} = {1}\n'.format
        stg = dtmp('corr_nrpts', self.corr_nrpts, '')
        stg += ftmp('corr_wait', self.corr_wait, '[s]')
        stg += ftmp('injection_interval', self.injection_interval, '[s]')
        stg += ftmp('timeout_orb', self.timeout_orb, '[s]')
        stg += dtmp('nr_points', self.nr_points, '')
        stg += 'Correctors to be measured:\n'
        stg += f'    {"corrs2measure":30s} Limits\n'
        for corr in self.corrs2measure:
            idx = self.ALL_CORRS.index(corr)
            lim = self.DLIMS[idx]
            stg += f'    {corr:30s} {lim:.2f}\n'
        return stg


class MeasureRespMatTBBO(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(params=Params(), target=self.measure_respmat)
        self.isonline = isonline
        self._model = None
        self._model_bpms_idx = None
        self._model_corrs_idx = None
        if isonline:
            self._create_devices()

    @property
    def model(self):
        """."""
        if self._model is None:
            model_li, *_ = li.create_accelerator()
            licv7_idx = pa.lattice.find_indices(model_li, 'fam_name', 'CV')[-1]

            model = model_li[licv7_idx:]
            model_tb, *_ = tb.create_accelerator(add_from_li_triplets=False)
            model_bo = bo.create_accelerator()

            self._len_li = len(model)
            self._len_tb = len(model_tb)
            self._len_bo = len(model_bo)

            model.extend(model_tb)
            model.extend(model_bo)

            self._model = model
            # Remove the first BPM in end of LI (not present in TB SOFB):
            self._model_bpms_idx = np.array(
                pa.lattice.find_indices(self._model, 'fam_name', 'BPM')
            )[1:]
            self._model_corrs_idx = self._find_model_corrs_idcs()

        return self._model

    @property
    def model_bpms_idx(self):
        """."""
        if self._model_bpms_idx is None:
            _ = self.model  # to populate bpms_idx
        return self._model_bpms_idx

    @property
    def model_corrs_idx(self):
        """."""
        if self._model_corrs_idx is None:
            _ = self.model  # to populate corrs_idx
        return self._model_corrs_idx

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
    def trajxy(self):
        """."""
        traj_xy = np.hstack([self.trajx, self.trajy])
        return traj_xy

    def inject_and_get_data(self, corr_name):
        """."""
        evg = self.devices['evg']
        traj_xy = list()
        traj_sum = list()
        timestamp = list()
        corr_strn = list()

        flag = _Event()

        def set_flag(*args, **kwgs):
            _ = args, kwgs
            flag.set()

        dcct_pv = self.devices['bo_dcct'].pv_object('RawReadings-Mon')
        dcct_pv.auto_monitor = True
        dcct_pv.add_callback(set_flag)
        flag.clear()

        for i in range(self.params.nr_points):
            traj_xy_0 = self.trajxy
            # evg.cmd_turn_on_injection()

            while not flag.wait(1):
                continue
            flag.clear()

            t0_ = _time.time()
            stg = f'    {i + 1:02d}/{self.params.nr_points:02d} -> '
            stg += 'Getting trajectory...'
            print(stg, end='\r', flush=True)
            if not self._wait_new_traj(traj_xy_0):
                stg += ' timed out waiting traj to update.'
            print(stg + '  done!')

            traj_xy_new = self.trajxy
            traj_sum_new = self.trajsum
            corr_strn_i = self.devices[corr_name].strength
            timestamp.append(_time.time())
            traj_xy.append(traj_xy_new)
            traj_sum.append(traj_sum_new)
            corr_strn.append(corr_strn_i)
            dtim = max(
                0, self.params.injection_interval - (_time.time() - t0_)
            )
            if i < self.params.nr_points - 1:
                _time.sleep(dtim)
            if self._stopevt.is_set():
                break

        dcct_pv.clear_callbacks()

        return dict(
            traj_xy=traj_xy,
            traj_sum=traj_sum,
            timestamp=timestamp,
            corr_strn=corr_strn,
        )

    def measure_respmat_corr(self, corr_name):
        """."""
        nrpts = self.params.corr_nrpts
        idx = self.params.ALL_CORRS.index(corr_name)
        kick_lim = self.params.DLIMS[idx]
        delta_strength = np.linspace(-kick_lim, kick_lim, nrpts)

        corr_dev = self.devices[corr_name]
        orig_strn = corr_dev.strength

        data = []
        try:
            for i, delta_strn in enumerate(delta_strength):
                print(
                    f'  {corr_name} {i + 1:02d}/{nrpts:02d} --> '
                    f'delta_strength: {delta_strn:.3f}'
                )
                new_strn = orig_strn + delta_strn
                if not self._set_device_corrector(corr_name, new_strn):
                    print('    Timedout waiting corrector. continuing...')
                _time.sleep(self.params.corr_wait)

                orb_data = self.inject_and_get_data(corr_name)
                data.append(orb_data)

                if self._stopevt.is_set():
                    break
        finally:
            print(f'  restoring {corr_name} strength...')
            if not self._set_device_corrector(corr_name, orig_strn):
                print('    Timedout waiting corrector to restore.')
            print(f'  {corr_name} strength: {corr_dev.strength:.3f}')
            if self._stopevt.is_set():
                print(f'  {corr_name} interrupted!')
            else:
                print(f'  {corr_name} finished!')
        return data

    def measure_respmat(self):
        """."""
        corrs = self.params.corrs2measure

        self.data = dict()
        print('Starting...')

        for idx, corr_name in enumerate(corrs):
            print(
                f'Varrying {corr_name:<20s} ({idx + 1:02d}/{len(corrs):02d})'
            )
            self.data[corr_name] = self.measure_respmat_corr(corr_name)
            if self._stopevt.is_set():
                break

        print('Finished.')

    def process_data(self, fit_order=1):
        """Process measured trajectories and build measured response matrix."""
        if not self.data:
            raise ValueError('No data to process. Run measurement first.')

        corrs_analysis = dict()
        for corr_name, meas in self.data.items():
            anl = self._process_data_corr(meas, corr_name, fit_order=fit_order)
            corrs_analysis[corr_name] = anl

        corr_names = tuple(corrs_analysis)
        nbpms = len(self.model_bpms_idx)
        ncorrs = len(corr_names)
        respmat_meas = np.zeros((2 * nbpms, ncorrs), dtype=float)

        for col, corr_name in enumerate(corr_names):
            respmat_meas[:, col] = corrs_analysis[corr_name]['respmat_col']

        self.analysis = dict(
            fit_order=fit_order,
            respmat_meas=respmat_meas,
            corr_analysis=corrs_analysis,
        )

    def calc_model_respmat(self, corr_names=None):
        """Calculate selected columns of the model response matrix.

        Parameters
        ----------
        corr_names : sequence of str or None
            Correctors whose response-matrix columns will be calculated.
            The returned columns follow exactly the order given here.

            If None, use the processed correctors stored in:
            self.analysis['corr_names']

            If no analysis is available, use: self.params.corrs2measure

        Returns
        -------
        respmat_model : numpy.ndarray
            Model response matrix with shape: (2 * nr_bpms, len(corr_names))
        """
        model = self.model
        bpm_idcs = self.model_bpms_idx
        nr_bpms = len(bpm_idcs)
        dkick = 50e-6

        if corr_names is None:
            if self.analysis:
                corr_names = tuple(self.analysis['corr_analysis'])
            else:
                corr_names = tuple(self.params.corrs2measure)

        corr_map = self._get_model_corrector_map()
        unknown_corrs = set(corr_names) - set(corr_map)

        if unknown_corrs:
            raise ValueError(
                'The following correctors were not found in the model: '
                f'{sorted(unknown_corrs)}.'
            )

        respmat_model = np.zeros((2 * nr_bpms, len(corr_names)), dtype=float)

        for col, corr_name in enumerate(corr_names):
            corr_info = corr_map[corr_name]
            corr_type = corr_info['type']
            elem_idcs = corr_info['indices']

            attr = 'vkick_polynom' if corr_type == 'CV' else 'hkick_polynom'

            nr_segments = len(elem_idcs)

            if nr_segments == 0:
                raise RuntimeError(
                    'No model segments were found for corrector '
                    f'{corr_name!r}.'
                )

            kicks_0 = [getattr(model[idx], attr) for idx in elem_idcs]

            try:
                # Positive kick.
                for idx, kick0 in zip(elem_idcs, kicks_0):
                    new_kick = kick0 + dkick / (2 * nr_segments)
                    self._apply_kick(idx, attr, new_kick)

                coordp, *_ = pa.tracking.line_pass(
                    model, particles=np.zeros(6), indices=bpm_idcs
                )

                # Negative kick.
                for idx, kick0 in zip(elem_idcs, kicks_0):
                    new_kick = kick0 - dkick / (2 * nr_segments)
                    self._apply_kick(idx, attr, new_kick)

                coordn, *_ = pa.tracking.line_pass(
                    model, particles=np.zeros(6), indices=bpm_idcs
                )

            finally:
                # Always restore the original model, even if tracking fails.
                for idx, kick0 in zip(elem_idcs, kicks_0):
                    self._apply_kick(idx, attr, kick0)

            respmat_model[:nr_bpms, col] = (coordp[0] - coordn[0]) / dkick
            respmat_model[nr_bpms:, col] = (coordp[2] - coordn[2]) / dkick

        return respmat_model

    # -------------- Fit Septum Forces -----------------------

    def set_septum_gradient(self, kxl, kyl, ksxl, ksyl):
        """."""
        idcs = pa.lattice.find_indices(self.model, 'fam_name', 'InjSeptM66')
        nr_segments = len(idcs)

        for idx in idcs:
            elem = self.model[idx]
            elem.KxL = kxl / nr_segments
            elem.KyL = kyl / nr_segments
            elem.KsxL = ksxl / nr_segments
            elem.KsyL = ksyl / nr_segments

    def get_septum_gradient(self):
        """."""
        idcs = pa.lattice.find_indices(self.model, 'fam_name', 'InjSeptM66')

        kxl, kyl, ksxl, ksyl = 0, 0, 0, 0

        for idx in idcs:
            elem = self.model[idx]
            kxl += elem.KxL
            kyl += elem.KyL
            ksxl += elem.KsxL
            ksyl += elem.KsyL

        return np.array([kxl, kyl, ksxl, ksyl])

    def reset_septum_gradient(self):
        """."""
        grads = np.zeros(shape=(4), dtype=float)
        return self.set_septum_gradient(*grads)

    def fit_septum_gradients_respmat(
        self,
        x0=None,
        bounds=None,
        fixed_gradients=None,
        corr_names=None,
        planes='xy',
        rel_residue_threshold=None,
        normalization='none',
        normalization_floor=1e-12,
        verbose=2,
        **least_squares_kwargs,
    ):
        """Fit septum integrated gradients using the response matrix.

        Parameters
        ----------
        x0 : array_like or None
            Initial values [KxL, KyL, KsxL, KsyL]. If None, use the
            current values from the model.

        bounds : 2-tuple or None
            Lower and upper bounds accepted by scipy.optimize.least_squares.

        corr_names : sequence of str or None
            Corrector columns included in the fit. If None, use the
            correctors that were measured and processed.

        planes : str
            Planes included in the fit: 'x', 'y', 'xy', 'h', 'v', 'hv'.

        rel_residue_threshold : float or None
            Reject individual response-matrix elements associated with
            poor polynomial trajectory fits.

        normalization : str
            Response-matrix normalization strategy. Available options are:
            'none', 'column_rms', 'plane_rms', 'block_rms'.

        normalization_floor : float
            Minimum normalization scale. This prevents groups with very
            small RMS values from receiving excessively large weights.

        verbose : int
            Optimizer verbosity.

        **least_squares_kwargs
            Additional arguments passed to scipy.optimize.least_squares.

        Returns
        -------
        result : scipy.optimize.OptimizeResult
            Result returned by least_squares.
        """
        if not self.analysis:
            raise ValueError(
                'No analysis available. Run process_data() first.'
            )

        target = self.analysis['respmat_meas']
        matrix_corr_names = tuple(self.analysis['corr_analysis'])

        if corr_names is None:
            fit_corr_names = matrix_corr_names
        else:
            fit_corr_names = tuple(corr_names)

        unknown_corrs = set(fit_corr_names) - set(matrix_corr_names)

        if unknown_corrs:
            raise ValueError(
                'The following correctors were not measured or processed: '
                f'{sorted(unknown_corrs)}.'
            )

        mask = self._build_respmat_fit_mask(
            corr_names=fit_corr_names,
            planes=planes,
            rel_residue_threshold=rel_residue_threshold,
        )

        if x0 is None:
            x0 = self.get_septum_gradient()

        if fixed_gradients is None:
            fixed_gradients = [None] * 4

        if len(fixed_gradients) != 4:
            raise ValueError(
                'fixed_gradients must contain four values in the order '
                '[KxL, KyL, KsxL, KsyL]. '
                'Use None for free parameters.'
            )

        fixed_gradients = np.array(fixed_gradients, dtype=object)
        free_mask = np.array(
            [value is None for value in fixed_gradients], dtype=bool
        )

        if not np.any(free_mask):
            raise ValueError('At least one septum gradient must remain free.')

        reference_gradients = x0.copy()

        for idx, value in enumerate(fixed_gradients):
            if value is not None:
                reference_gradients[idx] = float(value)

        x0_free = reference_gradients[free_mask]

        weights, normalization_scales = self._build_respmat_fit_weights(
            target=target,
            mask=mask,
            matrix_corr_names=matrix_corr_names,
            normalization=normalization,
            normalization_floor=normalization_floor,
        )

        self._respmat_fit_target = target
        self._respmat_fit_mask = mask
        self._respmat_fit_weights = weights
        self._respmat_fit_matrix_corr_names = matrix_corr_names
        self._respmat_fit_corr_names = fit_corr_names

        self._respmat_fit_reference_gradients = reference_gradients
        self._respmat_fit_free_mask = free_mask

        if bounds is None:
            bounds_free = (-np.inf, np.inf)
        else:
            lower, upper = bounds

            lower = np.broadcast_to(np.asarray(lower, dtype=float), (4,))
            upper = np.broadcast_to(np.asarray(upper, dtype=float), (4,))

            bounds_free = (lower[free_mask], upper[free_mask])

        kwargs = dict(
            fun=self._err_func_respmat,
            x0=x0_free,
            bounds=bounds_free,
            method='trf',
            x_scale='jac',
            verbose=verbose,
        )
        kwargs.update(least_squares_kwargs)

        result = least_squares(**kwargs)

        fitted_gradients = self._expand_free_septum_gradients(result.x)
        parameter_names = np.array(['KxL', 'KyL', 'KsxL', 'KsyL'])

        result.free_gradients = result.x.copy()
        result.full_gradients = fitted_gradients.copy()
        result.free_gradient_mask = free_mask.copy()
        result.free_gradient_names = parameter_names[free_mask].copy()
        result.fixed_gradient_names = parameter_names[~free_mask].copy()
        result.initial_gradients = reference_gradients.copy()
        result.fixed_gradients = fixed_gradients.copy()
        result.matrix_corr_names = matrix_corr_names
        result.fit_corr_names = fit_corr_names
        result.fit_mask = mask.copy()
        result.fit_weights = weights.copy()
        result.normalization = normalization.lower()
        result.normalization_scales = normalization_scales
        result.final_respmat = self.calc_model_respmat(
            corr_names=matrix_corr_names
        )

        return result

    @staticmethod
    def calc_fitting_error(fit_result, rcond=None):
        """Estimate parameter uncertainties from least_squares result."""
        jac = fit_result.jac
        nr_res, nr_params = jac.shape
        _, s_vals, vh = np.linalg.svd(jac, full_matrices=False)

        if rcond is None:
            rcond = np.finfo(float).eps * max(jac.shape)

        cutoff = rcond * s_vals[0]
        keep = s_vals > cutoff

        if not np.any(keep):
            raise ValueError('Jacobian has zero numerical rank.')

        vh = vh[keep]
        s_vals = s_vals[keep]
        cov = (vh.T / (s_vals * s_vals)) @ vh

        dof = nr_res - nr_params

        if dof > 0:
            res_var = 2 * fit_result.cost / dof
            cov *= res_var
        else:
            cov[:] = np.nan

        return np.sqrt(np.diag(cov))

    # ---------------- Plot methods --------------------------

    def plot_respmat_col(self, corr_name):
        """Plot a response-matrix column.

        Parameters
        ----------
        corr_name : str
            Name of a processed corrector.

        respmat_model : numpy.ndarray or None
            Optional model response matrix with the same column ordering
            as self.analysis['corr_analysis'].keys().
        """
        if not self.analysis:
            raise ValueError(
                'No analysis available. Run process_data() first.'
            )

        nr_bpms = len(self.model_bpms_idx)
        corr_names = tuple(self.analysis['corr_analysis'])

        if corr_name not in corr_names:
            raise ValueError(f'Corrector {corr_name!r} was not processed.')

        col = corr_names.index(corr_name)
        col_meas = self.analysis['respmat_meas'][:, col]
        col_model = self.calc_model_respmat()[:, col]

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))

        axs[0].plot(col_model[:nr_bpms], '-o', color='tab:blue', label='model')
        axs[1].plot(col_model[nr_bpms:], '-o', color='tab:red', label='model')

        axs[0].plot(
            col_meas[:nr_bpms], 'o--', color='b', alpha=0.75, label='meas'
        )
        axs[1].plot(
            col_meas[nr_bpms:], 'o--', color='C1', alpha=0.75, label='meas'
        )

        axs[0].axvline(6 - 1 / 2, ls='--', color='k')
        axs[1].axvline(6 - 1 / 2, ls='--', color='k')

        axs[0].set_ylabel(r'$R_x$ [m/rad]')
        axs[1].set_ylabel(r'$R_y$ [m/rad]')
        axs[1].set_xlabel('BPM index')
        axs[0].set_title(f'Response-matrix column: {corr_name}')

        for ax in axs:
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.5, ls='--', lw=0.5, color='k')

        fig.tight_layout()

        return fig, axs

    def plot_traj_fitting_relative_residue(self, corr_name, order=1):
        """."""
        fig, ax = plt.subplots(figsize=(10, 5))
        ratio = self.analysis['corr_analysis'][corr_name]['fit_rel_residue'][
            order + 1
        ]
        nbpm = len(self.model_bpms_idx)

        ax.plot(ratio[:nbpm], '-o', label='Horizontal')
        ax.plot(ratio[nbpm:], '-o', label='Vertical')

        ax.legend(loc='best', ncol=2, fontsize='small')
        ax.set_title(f'Relative Residue Fit Order N={order} by Order 0.')
        ax.set_xlabel('BPM Index')
        ax.set_ylabel(
            r'Relative residue $\chi^2_{y=P_N(x)}/\chi^2_{y=P_0(x)}$'
        )
        ax.grid(True, ls='--', alpha=0.4, color='k', lw=0.5)
        ax.set_ylim(None, 1.15)
        fig.tight_layout()
        return fig, ax

    def plot_traj_fit_at_bpm(self, corr_name, bpm_idx=0, plane='h'):
        """."""
        ish = plane.lower().startswith(('h', 'x'))
        idx = bpm_idx
        if not ish:
            idx += len(self.model_bpms_idx)
        analysis = self.analysis['corr_analysis'][corr_name]
        ratio = analysis['fit_rel_residue']
        corr_strn = analysis['corr_strn']
        xfit = analysis['fit_x']
        traj_points = analysis['traj_xy'][:, idx]
        coefs = analysis['fit_coefs'][:, idx]
        traj_fit = np.polynomial.polynomial.polyval(xfit, coefs)

        fig, ax = plt.subplots(figsize=(8, 5))

        stg = f'BPM {bpm_idx:d}, '
        stg += f'{"Horizontal" if ish else "Vertical":s} Plane\n'
        stg += 'coefs = ['
        stg += ', '.join([f'{r:.2g}' for r in coefs])
        stg += ']    ratios = ['
        stg += ', '.join([f'{r:.2g}' for r in ratio[2:, idx]])
        stg += ']'
        ax.set_title(stg, fontsize='small')

        ax.plot(corr_strn, traj_points, 'o', label='Data')
        ax.plot(corr_strn, traj_fit, label='Fit')
        ax.legend(loc='best')
        ax.set_xlabel('Corrector Strengths [urad]')
        ax.set_ylabel('Trajectory [um]')
        ax.grid(True, ls='--', alpha=0.4, color='k', lw=0.5)

        fig.tight_layout()
        return fig, ax

    # ---------------- Helper methods ------------------------

    def _process_data_corr(self, data, corr, fit_order=1):
        """."""
        trajs = []
        corr_strn = []
        for datum in data:
            trajs.extend(datum['traj_xy'])
            corr_strn.extend(datum['corr_strn'])
        trajs = np.array(trajs)
        corr_strn = np.array(corr_strn)
        if ('InjSept' in corr) or ('InjKckr' in corr):
            corr_strn = corr_strn * 1e3
        xfit = corr_strn - corr_strn.mean()
        coefs, _ = np.polynomial.polynomial.polyfit(
            xfit, trajs, deg=fit_order, full=True
        )

        ress = [(trajs**2).sum(axis=0)]
        for i in range(1, fit_order + 2):
            fit = np.polynomial.polynomial.polyval(xfit, coefs[:i])
            ress.append(((trajs - fit.T) ** 2).sum(axis=0))
        ress = np.array(ress)
        ratio = ress / ress[1][None, :]

        return dict(
            fit_x=xfit,
            fit_coefs=coefs,
            fit_residue_order=ress,
            fit_rel_residue=ratio,
            respmat_col=coefs[1],
            traj_xy=trajs,
            corr_strn=corr_strn,
        )

    def _wait_new_traj(self, traj_xy_0=None, timeout_orb=None):
        """."""
        timeout_orb = timeout_orb or self.params.timeout_orb
        if traj_xy_0 is None:
            traj_xy_0 = self.trajxy
        for _ in range(50):
            traj_xy = self.trajxy
            if not np.any(np.isclose(traj_xy_0, traj_xy)):
                return True
            _time.sleep(timeout_orb / 50)
        return False

    def _create_devices(self):
        """."""
        self.devices = dict(
            evg=_EVG(),
            tb_sofb=_SOFB(_SOFB.DEVICES.TB),
            bo_sofb=_SOFB(_SOFB.DEVICES.BO),
            bo_dcct=_DCCT(_DCCT.DEVICES.BO),
        )
        for corr_name in self.params.ALL_CORRS:
            self.devices[corr_name] = _PowerSupply(corr_name)

    def _set_device_corrector(self, devname, value):
        dev = self.devices[devname]
        return dev.set_strength(value, tol=0.2, wait_mon=False)

    def _find_model_corrs_idcs(self):
        model = self._model
        len_li = self._len_li
        len_tb = self._len_tb

        ch_idcs = []
        cv_idcs = []

        idx = 0
        for elem in model[:len_li]:
            name = elem.fam_name
            if name.startswith('CH'):
                ch_idcs.append([idx])
            elif name.startswith('CV'):
                cv_idcs.append([idx])
            idx += 1

        for elem in model[len_li : len_li + len_tb]:
            name = elem.fam_name
            if name.startswith('CHV') or name.startswith('QS'):
                ch_idcs.append([idx])
                cv_idcs.append([idx])
            idx += 1

        sept_idcs = [pa.lattice.find_indices(model, 'fam_name', 'InjSept')]
        kckr_idcs = [pa.lattice.find_indices(model, 'fam_name', 'InjKckr')]

        corr_idcs = dict(
            CH=ch_idcs, InjSept=sept_idcs, InjKckr=kckr_idcs, CV=cv_idcs
        )
        return corr_idcs

    def _get_model_corrector_entries(self):
        """Return model correctors in the same order as Params.ALL_CORRS."""
        entries = []

        for corr_type, corr_idcs_list in self.model_corrs_idx.items():
            for elem_idcs in corr_idcs_list:
                entries.append((corr_type, np.asarray(elem_idcs, dtype=int)))

        if len(entries) != len(self.params.ALL_CORRS):
            raise RuntimeError(
                'The number of model correctors does not match '
                'Params.ALL_CORRS: '
                f'model={len(entries)}, '
                f'ALL_CORRS={len(self.params.ALL_CORRS)}.'
            )

        named_entries = []

        for corr_name, entry in zip(self.params.ALL_CORRS, entries):
            corr_type, elem_idcs = entry

            if elem_idcs.size == 0:
                raise RuntimeError(
                    'No model elements were found for corrector '
                    f'{corr_name!r}.'
                )

            named_entries.append((corr_name, corr_type, elem_idcs))

        return tuple(named_entries)

    def _get_model_corrector_map(self):
        """Map power-supply names to model corrector information."""
        info = dict()
        for c_name, c_type, el_idcs in self._get_model_corrector_entries():
            info[c_name] = {'type': c_type, 'indices': el_idcs}
        return info

    def _apply_kick(self, idx, attr, kick):
        elem = self.model[idx]

        try:
            setattr(elem, attr, kick)
        except ZeroDivisionError:
            fallback = attr.replace('_polynom', '')
            setattr(elem, fallback, kick)

    def _build_respmat_fit_mask(
        self, corr_names=None, planes='xy', rel_residue_threshold=None
    ):
        """Build a Boolean mask selecting response-matrix elements.

        Parameters
        ----------
        corr_names : sequence of str or None
            Processed correctors included in the fit. If None, use every
            corrector present in self.analysis['corr_analysis'].keys().

        planes : str
            Planes included in the fit. Accepted values are:
            'x', 'y', 'xy', 'h', 'v' and 'hv'.

        rel_residue_threshold : float or None
            If not None, reject BPM/corrector elements for which the
            relative residual of the linear trajectory fit is above
            this threshold.

        Returns
        -------
        mask : numpy.ndarray
            Boolean matrix with the same shape as respmat_meas.
        """
        respmat_meas = self.analysis['respmat_meas']
        measured_corr_names = tuple(self.analysis['corr_analysis'])

        mask = np.zeros_like(respmat_meas, dtype=bool)

        if corr_names is None:
            corr_names = measured_corr_names
        else:
            corr_names = tuple(corr_names)

        unknown_corrs = set(corr_names) - set(measured_corr_names)

        if unknown_corrs:
            raise ValueError(
                'The following correctors were not measured or processed: '
                f'{sorted(unknown_corrs)}.'
            )

        nbpms = len(self.model_bpms_idx)
        plane = planes.lower()

        use_x = ('x' in plane) or ('h' in plane)
        use_y = ('y' in plane) or ('v' in plane)

        if not use_x and not use_y:
            raise ValueError(
                f'Invalid planes={planes!r}. Use "x", "y" or "xy".'
            )

        for corr_name in corr_names:
            col = measured_corr_names.index(corr_name)

            if use_x:
                mask[:nbpms, col] = True

            if use_y:
                mask[nbpms:, col] = True

            if rel_residue_threshold is not None:
                corr_analysis = self.analysis['corr_analysis'][corr_name]

                # Residual of the linear fit divided by the residual
                # of the constant fit.
                ratio = corr_analysis['fit_rel_residue'][2]

                good_fit = np.isfinite(ratio) & (
                    ratio <= rel_residue_threshold
                )

                mask[:, col] &= good_fit

        mask &= np.isfinite(respmat_meas)

        return mask

    def _build_respmat_fit_weights(
        self,
        target,
        mask,
        matrix_corr_names,
        normalization='none',
        normalization_floor=1e-12,
    ):
        """Build deterministic weights for response-matrix fitting.

        Parameters
        ----------
        target : numpy.ndarray
            Measured response matrix.

        mask : numpy.ndarray
            Boolean matrix selecting response-matrix elements.

        matrix_corr_names : sequence of str
            Corrector names associated with the columns of target and mask.

        normalization : str
            Normalization strategy. Available options are:
            'none', 'column_rms', 'plane_rms' and 'block_rms'.

        normalization_floor : float
            Minimum allowed normalization scale.

        Returns
        -------
        weights : numpy.ndarray
            Deterministic fitting weights with the same shape as target.

        scales : dict
            Effective normalization scales used for each group.
        """
        matrix_corr_names = tuple(matrix_corr_names)

        if normalization_floor <= 0:
            raise ValueError('normalization_floor must be positive.')

        normalization = normalization.lower()
        weights = np.ones_like(target, dtype=float)
        scales = dict()

        def apply_rms(group_mask, group_name):
            """Apply RMS normalization to one selected group."""
            group_mask = np.asarray(group_mask, dtype=bool)
            group_mask &= mask

            if not np.any(group_mask):
                scales[group_name] = np.nan
                return

            values = target[group_mask]
            scale = np.sqrt(np.mean(values**2))

            if not np.isfinite(scale):
                raise ValueError(
                    f'Non-finite normalization scale for group {group_name!r}.'
                )

            effective_scale = max(scale, normalization_floor)

            weights[group_mask] = 1.0 / effective_scale
            scales[group_name] = effective_scale

        if normalization == 'none':
            scales['global'] = 1.0

        elif normalization == 'column_rms':
            selected_cols = np.nonzero(np.any(mask, axis=0))[0]

            for col in selected_cols:
                col_mask = np.zeros_like(mask, dtype=bool)
                col_mask[:, col] = True

                corr_name = matrix_corr_names[col]

                apply_rms(
                    group_mask=col_mask, group_name=f'column:{corr_name}'
                )

        elif normalization == 'plane_rms':
            nbpms = len(self.model_bpms_idx)

            x_mask = np.zeros_like(mask, dtype=bool)
            y_mask = np.zeros_like(mask, dtype=bool)

            x_mask[:nbpms, :] = True
            y_mask[nbpms:, :] = True

            apply_rms(x_mask, 'plane:x')
            apply_rms(y_mask, 'plane:y')

        elif normalization == 'block_rms':
            nbpms = len(self.model_bpms_idx)

            horizontal_cols = []
            vertical_cols = []

            for col, corr_name in enumerate(matrix_corr_names):
                if 'CV' in corr_name:
                    vertical_cols.append(col)
                else:
                    horizontal_cols.append(col)

            xh_mask = np.zeros_like(mask, dtype=bool)
            yh_mask = np.zeros_like(mask, dtype=bool)
            xv_mask = np.zeros_like(mask, dtype=bool)
            yv_mask = np.zeros_like(mask, dtype=bool)

            xh_mask[:nbpms, horizontal_cols] = True
            yh_mask[nbpms:, horizontal_cols] = True
            xv_mask[:nbpms, vertical_cols] = True
            yv_mask[nbpms:, vertical_cols] = True

            apply_rms(xh_mask, 'block:x_from_h')
            apply_rms(yh_mask, 'block:y_from_h')
            apply_rms(xv_mask, 'block:x_from_v')
            apply_rms(yv_mask, 'block:y_from_v')

        else:
            valid = ('none', 'column_rms', 'plane_rms', 'block_rms')

            raise ValueError(
                f'Unknown normalization strategy: {normalization!r}. '
                f'Valid options are {valid}.'
            )

        return weights, scales

    def _err_func_respmat(self, free_gradients):
        """Residual vector used in response-matrix fitting."""
        gradients = self._expand_free_septum_gradients(free_gradients)
        self.set_septum_gradient(*gradients)

        respmat_model = self.calc_model_respmat(
            corr_names=self._respmat_fit_matrix_corr_names
        )

        target = self._respmat_fit_target
        mask = self._respmat_fit_mask
        weights = self._respmat_fit_weights

        residual = (respmat_model - target) * weights

        return residual[mask].ravel()

    def _expand_free_septum_gradients(self, free_gradients):
        """Reconstruct the complete septum-gradient vector."""
        gradients = self._respmat_fit_reference_gradients.copy()
        gradients[self._respmat_fit_free_mask] = free_gradients
        return gradients
