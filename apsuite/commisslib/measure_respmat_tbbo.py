"""."""

import time as _time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

import pyaccel as pa
from pymodels import li, tb, bo

from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.devices import PowerSupply as _PowerSupply
from siriuspy.devices import EVG as _EVG
from siriuspy.devices import SOFB as _SOFB
from siriuspy.search import PSSearch as _PSSearch

from ..optimization import SimulAnneal
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
        + [0.1]  # InjSept [urad]
        + [0.1]  # InjKckr [urad]
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
        for i in range(self.params.nr_points):
            traj_xy_0 = self.trajxy
            evg.cmd_turn_on_injection()

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
        """."""
        if not self.data:
            raise ValueError('No data to process. Run measurement first.')

        corrs_analysis = dict()
        for corr, meas in self.data.items():
            anl = self._process_data_corr(meas, fit_order=fit_order)
            corrs_analysis[corr] = anl

        nbpms = len(self.model_bpms_idx)
        ncorrs = len(self.params.ALL_CORRS)
        respmat_meas = np.zeros((2 * nbpms, ncorrs), dtype=float)
        for corr, anl in corrs_analysis.items():
            idx = self.params.ALL_CORRS.index(corr)
            respmat_meas[:, idx] = anl['respmat_col']

        self.analysis = dict(
            fit_order=fit_order,
            respmat_meas=respmat_meas,
            corr_analysis=corrs_analysis,
        )

    def calc_model_respmat(self):
        """."""
        model = self.model
        dkick = 50e-6
        bpm_idcs = self.model_bpms_idx
        corrs = self.model_corrs_idx
        nr_bpms = len(self.model_bpms_idx)
        nr_corrs = sum([len(v) for v in self.model_corrs_idx.values()])
        respmat_model = np.zeros((2 * nr_bpms, nr_corrs))

        col = 0
        for corr_type, corr_idcs in corrs.items():
            attr = 'hkick_polynom' if corr_type != 'CV' else 'vkick_polynom'

            for elem_idcs in corr_idcs:
                nr_segments = len(elem_idcs)
                kicks_0 = [getattr(model[idx], attr) for idx in elem_idcs]

                # Positive kick
                for idx, kick0 in zip(elem_idcs, kicks_0):
                    new_kick = kick0 + dkick / 2 / nr_segments
                    self._apply_kick(idx, attr, new_kick)

                coordp, *_ = pa.tracking.line_pass(
                    model, particles=np.zeros(6), indices=bpm_idcs
                )

                # Negative kick
                for idx, kick0 in zip(elem_idcs, kicks_0):
                    new_kick = kick0 - dkick / 2 / nr_segments
                    self._apply_kick(idx, attr, new_kick)

                coordn, *_ = pa.tracking.line_pass(
                    model, particles=np.zeros(6), indices=bpm_idcs
                )

                # Restore original kicks
                for idx, kick0 in zip(elem_idcs, kicks_0):
                    self._apply_kick(idx, attr, kick0)

                # Response
                respmat_model[:nr_bpms, col] = (coordp[0] - coordn[0]) / dkick
                respmat_model[nr_bpms:, col] = (coordp[2] - coordn[2]) / dkick

                col += 1

        return respmat_model

    # ---------------- Plot methods --------------------------

    def plot_respmat_col(self, corr_name, nr_bpms=None):
        """."""
        if nr_bpms is None:
            nr_bpms = len(self.model_bpms_idx)

        idx = self.params.ALL_CORRS.index(corr_name)
        col_meas = self.analysis['respmat_meas'][:, idx]

        fig, axs = plt.subplots(2, 1, figsize=(10, 6))
        # axs[0].plot(
        #     col_model[:nr_bpms], '-o', color='tab:blue', label='model'
        # )
        axs[0].plot(
            col_meas[:nr_bpms], 'o--', color='b', alpha=0.75, label='meas'
        )

        # axs[1].plot(col_model[nr_bpms:], '-o', color='tab:red', label='model')
        axs[1].plot(
            col_meas[nr_bpms:], 'o--', color='C1', alpha=0.75, label='meas'
        )

        axs[0].axvline(6 - 1 / 2, ls='--', color='k')
        axs[1].axvline(6 - 1 / 2, ls='--', color='k')

        axs[0].set_ylabel(r'$col_x$ [um/urad]')
        axs[1].set_ylabel(r'$col_y$ [um/urad]')
        axs[1].set_xlabel('BPM idx')
        axs[0].set_title(f'respmat col - {corr_name}')

        axs[0].legend(fontsize=10)
        axs[1].legend(fontsize=10)
        axs[0].grid(True, alpha=0.5, ls='--', lw=0.5, color='k')
        axs[1].grid(True, alpha=0.5, ls='--', lw=0.5, color='k')

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

    def _process_data_corr(self, data, fit_order=1):
        """."""
        trajs = []
        corr_strn = []
        for datum in data:
            trajs.extend(datum['traj_xy'])
            corr_strn.extend(datum['corr_strn'])
        trajs = np.array(trajs)
        corr_strn = np.array(corr_strn)
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

    def _apply_kick(self, idx, attr, kick):
        elem = self.model[idx]

        try:
            setattr(elem, attr, kick)
        except ZeroDivisionError:
            fallback = attr.replace('_polynom', '')
            setattr(elem, fallback, kick)


# def calc_model_respmatTBBO(
#     tb_mod, model, corr_names, elems, meth='middle', ishor=True
# ):
#     """."""
#     bpms = np.array(pa.lattice.find_indices(model, 'fam_name', 'BPM'))[1:]
#     _, cumulmat = pa.tracking.find_m44(
#         model, indices='open', fixed_point=[0, 0, 0, 0]
#     )

#     matrix = np.zeros((len(corr_names), 2 * bpms.size))
#     for idx, corr in enumerate(corr_names):
#         elem = elems[corr]
#         indcs = np.array(elem.model_indices)
#         if corr.sec == 'BO':
#             print('Booster ', corr)
#             indcs += len(tb_mod)
#         cortype = elem.magnet_type
#         kxl = kyl = ksxl = ksyl = 0
#         if corr.dev == 'InjSept':
#             # kxl = tb_mod[indcs[0][1]].KxL
#             # kyl = tb_mod[indcs[0][1]].KyL
#             # ksxl = tb_mod[indcs[0][1]].KsxL
#             # ksyl = tb_mod[indcs[0][1]].KsyL
#             midx = pa.lattice.find_indices(tb_mod, 'fam_name', 'InjSeptM66')
#             for m in midx:
#                 kxl += tb_mod[m].KxL
#                 kyl += tb_mod[m].KyL
#                 ksxl += tb_mod[m].KsxL
#                 ksyl += tb_mod[m].KsyL
#         if not ishor and corr.dev in {'InjSept', 'InjKckr'}:
#             cortype = 'vertical'
#         matrix[idx, :] = _get_respmat_line(
#             cumulmat,
#             indcs,
#             bpms,
#             length=elem.model_length,
#             kxl=kxl,
#             kyl=kyl,
#             ksxl=ksxl,
#             ksyl=ksyl,
#             cortype=cortype,
#             meth=meth,
#         )
#     return matrix


# def _get_respmat_line(
#     cumul_mat,
#     indcs,
#     bpms,
#     length,
#     kxl=0,
#     kyl=0,
#     ksxl=0,
#     ksyl=0,
#     cortype='vertical',
#     meth='middle',
# ):

#     idx = 3 if cortype.startswith('vertical') else 1
#     cor = indcs[0]
#     if meth.lower().startswith('end'):
#         cor = indcs[-1] + 1
#     elif meth.lower().startswith('mid'):
#         # create a symplectic integrator of second order
#         # for the last half of the element:
#         drift = np.eye(4, dtype=float)
#         drift[0, 1] = length / 2 / 2
#         drift[2, 3] = length / 2 / 2
#         quad = np.eye(4, dtype=float)
#         quad[1, 0] = -kxl / 2
#         quad[3, 2] = -kyl / 2
#         quad[1, 2] = -ksxl / 2
#         quad[3, 0] = -ksyl / 2
#         half_cor = np.dot(np.dot(drift, quad), drift)

#     m0c = cumul_mat[cor]
#     if meth.lower().startswith('mid'):
#         m0c = np.linalg.solve(half_cor, m0c)
#     mat = np.linalg.solve(m0c.T, cumul_mat[bpms].transpose((0, 2, 1)))
#     mat = mat.transpose(0, 2, 1)
#     # if meth.lower().startswith('mid'):
#     #     mat = np.dot(mat, half_cor)
#     respx = mat[:, 0, idx]
#     respy = mat[:, 2, idx]
#     respx[bpms < indcs[0]] = 0
#     respy[bpms < indcs[0]] = 0
#     return np.hstack([respx, respy])


# class FindSeptQuad(SimulAnneal):
#     """."""

#     def __init__(
#         self,
#         tb_model,
#         bo_model,
#         corr_names,
#         elems,
#         respmat,
#         nturns=5,
#         save=False,
#         in_sept=True,
#     ):
#         """."""
#         super().__init__(save=save)
#         self.tb_model = tb_model
#         self.bo_model = bo_model
#         self.corr_names = corr_names
#         self.elems = elems
#         self.nturns = nturns
#         self.respmat = respmat
#         self.in_sept = in_sept

#     def initialization(self):
#         """."""
#         return

#     def calc_obj_fun(self):
#         """."""
#         if self.in_sept:
#             sept_idx = pa.lattice.find_indices(
#                 self.tb_model, 'fam_name', 'InjSept'
#             )
#         else:
#             sept_idx = self.elems['TB-04:MA-CV-2'].model_indices
#         k, ks = self._position
#         pa.lattice.set_attribute(self.tb_model, 'K', sept_idx, k)
#         pa.lattice.set_attribute(self.tb_model, 'Ks', sept_idx, ks)
#         respmat = calc_model_respmatTBBO(
#             self.tb_model, self.bo_model, self.corr_names, self.elems
#         )
#         respmat -= self.respmat
#         return np.sqrt(np.mean(respmat * respmat))
