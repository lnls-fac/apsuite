"""Main module."""

import datetime as _datetime
import time as _time
from copy import deepcopy as _dcopy

import numpy as _np
import pyaccel as _pyacc
from mathphys.functions import get_namedtuple as _get_namedtuple
from pymodels import si as _si
from siriuspy.devices import (
    CurrInfoSI as _CurrInfoSI,
    PowerSupply as _PowerSupply,
    SOFB as _SOFB,
)
from siriuspy.namesys import SiriusPVName as _PVName

from ..orbcorr.orbit_correction import OrbitCorr as _OrbitCorr
from ..utils import (
    ParamsBaseClass as _ParamsBaseClass,
    ThreadedMeasBaseClass as _BaseClass,
)

from .measure_bba import BBAParams as _BBAParams


class ParallelBBAParams(_ParamsBaseClass):
    """."""

    BPMNAMES = _BBAParams.BPMNAMES
    QUADNAMES = _BBAParams.QUADNAMES

    def __init__(self):
        """."""
        super().__init__()

        self.quad_deltakl = 0.01  # [1/m]

        self.wait_correctors = 0.3  # [s]
        self.wait_quadrupole = 0.3  # [s]
        self.timeout_wait_orbit = 3  # [s]

        self.corr_max_nr_iters = 8
        self.ios_conv_threshold = 0.1

        self.sofb_nrpoints = 20
        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 5  # [um]

        self.cycling_nr_steps = 2

    def __str__(self):
        """."""
        stg = ''
        stg += f'quad_deltakl       = {self.quad_deltakl:.3f}\n'
        stg += f'wait_correctors    = {self.wait_correctors:.3f}\n'
        stg += f'wait_quadrupole    = {self.wait_quadrupole:.3f}\n'
        stg += f'timeout_wait_orbit = {self.timeout_wait_orbit:.3f}\n'
        stg += f'corr_nr_iters      = {self.corr_max_nr_iters:.3f}\n'
        stg += f'ios_conv_threshold = {self.ios_conv_threshold:.2e}\n'
        stg += f'sofb_nrpoints      = {self.sofb_nrpoints:.3f}\n'
        stg += f'sofb_maxcorriter   = {self.sofb_maxcorriter:.3f}\n'
        stg += f'sofb_maxorberr     = {self.sofb_maxorberr:.3f}\n'
        stg += f'cycling_nr_steps   = {self.cycling_nr_steps:.3f}\n'
        return stg

    @staticmethod
    def get_default_groups(ngroups=8):
        """."""
        if ngroups == 2:
            group_class = [
                [('M2', ''), ('C3', '1'), ('C1', '2'), ('C4', '')],
                [('C1', '1'), ('C3', '2'), ('C2', ''), ('M1', '')],
            ]
        elif ngroups in {8, 16}:
            group_class = [
                [('M2', '')],
                [('C3', '1')],
                [('C1', '1')],
                [('C3', '2')],
                [('C1', '2')],
                [('C4', '')],
                [('C2', '')],
                [('M1', '')],
            ]
        else:
            group_class = [
                [('M2', ''), ('C3', '1')],
                [('C1', '1'), ('C3', '2')],
                [('C1', '2'), ('C4', '')],
                [('C2', ''), ('M1', '')],
            ]

        groups = [
            sorted([
                e
                for e in ParallelBBAParams.BPMNAMES
                if (e.sub[2:], e.idx) in k
            ])
            for k in group_class
        ]

        if ngroups == 16:
            groups_ = []
            for grp in groups:
                groups_.append(grp[::2])
                groups_.append(grp[1::2])
            groups = groups_

        groups = [
            sorted(
                group,
                key=lambda x: {
                    'Q4': 0,
                    'Q1': 1,
                    'QDB2': 2,
                    'QDP2': 3,
                    'QS': 4,
                }.get(
                    ParallelBBAParams.QUADNAMES[
                        ParallelBBAParams.BPMNAMES.index(x)
                    ].dev,
                    999,
                ),
            )
            for group in groups
        ]
        return groups

    def get_default_dkl(self, ngroups=8, groups=None):
        """."""
        groups = self.get_default_groups(ngroups) if groups is None else groups
        dkl = [_np.ones(len(g)) * self.quad_deltakl for g in groups]
        for d in dkl:
            d[::2] *= -1
        return dkl


class DoParallelBBA(_BaseClass):
    """."""

    STATUS = _get_namedtuple('Status', ['Fail', 'Success'])

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=ParallelBBAParams(), target=self._do_pbba, isonline=isonline
        )
        self.data['bpmnames'] = list(ParallelBBAParams.BPMNAMES)
        self.data['quadnames'] = list(ParallelBBAParams.QUADNAMES)
        self.data['measure'] = list()
        self.data['groups2dopbba'] = ParallelBBAParams.get_default_groups()
        self.data['delta_kl'] = self.params.get_default_dkl(
            groups=self.data['groups2dopbba']
        )
        self.data['log'] = list()
        self.data['jacobians'] = list()
        self._model = None
        self._fam_data = None

        if self.isonline:
            self.devices['sofb'] = _SOFB(_SOFB.DEVICES.SI)
            self.devices['currinfosi'] = _CurrInfoSI()
            self.connect_to_quadrupoles()

    def __str__(self):
        """."""
        stn = 'Params\n'
        stp = self.params.__str__()
        stp = '    ' + stp.replace('\n', '\n    ')
        stn += stp + '\n'
        connected = str(self.connected and len(self.devices.keys()) > 0)
        stn += 'Connected?  ' + connected + '\n\n'
        stn += '     {:^20s} {:^20s} {:^7s}\n'.format('BPM', 'Quad', 'dKL')
        tmplt = '{:03d}: {:^20s} {:^20s} {:+.3f}\n'
        dta = self.data
        for group_id, group in enumerate(self.data['groups2dopbba']):
            stn += f'> Group {group_id:03d}\n'
            for j, bpm in enumerate(group):
                idx = dta['bpmnames'].index(bpm)
                stn += tmplt.format(
                    idx,
                    dta['bpmnames'][idx],
                    dta['quadnames'][idx],
                    dta['delta_kl'][group_id][j],
                )
            stn += '\n'
        return stn

    @property
    def havebeam(self):
        """."""
        haveb = self.devices['currinfosi']
        return haveb.connected and haveb.storedbeam

    @property
    def measuredbpms(self):
        """."""
        mesured = []
        for group in self.data['measure']:
            mesured.extend(group['bpms'])
        return sorted(mesured)

    # #### pbba groups and deltas #####
    @property
    def groups2dopbba(self):
        """."""
        return _dcopy(self.data['groups2dopbba'])

    @groups2dopbba.setter
    def groups2dopbba(self, groups):
        self.data['groups2dopbba'] = [
            [_PVName(bpm) for bpm in group if isinstance(bpm, str)]
            for group in groups
            if isinstance(group, (list, tuple, _np.ndarray))
        ]

    @property
    def delta_kl(self):
        """."""
        return _dcopy(self.data['delta_kl'])

    @delta_kl.setter
    def delta_kl(self, value):
        _max = self.params.quad_deltakl
        for i, group in enumerate(self.data['groups2dopbba']):
            if len(value[i]) != len(group):
                raise ValueError(
                    f'size mismatch between group {i} and given delta_kl'
                )
            if any([abs(v) > _max for v in value[i]]):
                raise ValueError(f"values for delta kl can't exceed {_max}")
        self.data['delta_kl'] = _dcopy(value)

    # #### model utils #####
    @property
    def jacobians(self):
        """."""
        return _dcopy(self.data['jacobians'])

    @jacobians.setter
    def jacobians(self, jacs):
        """."""
        if len(jacs) != len(self.data['groups2dopbba']):
            raise ValueError('Size not compatible.')
        self.data['jacobians'] = _dcopy(jacs)

    @property
    def model(self):
        """."""
        if self._model is None:
            self._log_print('\n     Undefined model... setting a default one')
            self._model = _si.create_accelerator()
            self._model.cavity_on = True
            self._model.radiation_on = 1
            self._model = _si.fitted_models.vertical_dispersion_and_coupling(
                self._model
            )
            self._fam_data = _si.families.get_family_data(self._model)
        return self._model

    @model.setter
    def model(self, value):
        if not value.cavity_on and value.radiation_on != 1:
            raise ValueError(
                'cavity_on must be True and radiation_on must be 1'
            )
        self._model = value
        self._fam_data = _si.families.get_family_data(self._model)

    @property
    def fam_data(self):
        """."""
        return self._fam_data

    # ### quadrupole connection #####
    def connect_to_quadrupoles(self):
        """."""
        for qname in self.data['quadnames']:
            if qname in self.devices:
                continue
            self.devices[qname] = _PowerSupply(
                qname,
                props2init=('PwrState-Sts', 'KL-SP', 'KL-RB', 'KLRef-Mon'),
            )

    # #### sofb utils #####
    def get_orbit(self):
        """."""
        if not self.havebeam:
            return
        sofb = self.devices['sofb']
        nrpts = sofb.nr_points
        sofb.nr_points = self.params.sofb_nrpoints

        sofb.cmd_reset()
        sofb.wait_buffer(self.params.timeout_wait_orbit)

        sofb.nr_points = nrpts

        return _np.hstack([sofb.orbx, sofb.orby])

    def correct_orbit(self):
        """."""
        if not self.havebeam:
            return
        sofb = self.devices['sofb']
        sofb.correct_orbit_manually(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr,
        )

    def get_kicks(self):
        """."""
        sofb = self.devices['sofb']
        return _np.r_[sofb.kickch, sofb.kickcv, sofb.kickrf]

    def set_delta_kicks(self, dkicks):
        """."""
        sofb = self.devices['sofb']
        nch, ncv, nrf = sofb._data.nr_ch, sofb._data.nr_cv, 1
        if len(dkicks) != nch + ncv + nrf:
            raise ValueError(
                f'invalid dim for dkicks, must have shape=({nch + ncv + nrf},)'
            )
        dch, dcv, drf = dkicks[:nch], dkicks[nch : nch + ncv], dkicks[-1]

        sofb.deltakickch, sofb.deltakickcv, sofb.deltakickrf = dch, dcv, drf
        sofb.cmd_applycorr_all()

    @property
    def enbllistbpm(self):
        """."""
        sofb = self.devices['sofb']
        enblx = sofb.bpmxenbl.copy()
        enbly = sofb.bpmyenbl.copy()
        return _np.array(_np.hstack([enblx, enbly]), dtype=bool)

    @enbllistbpm.setter
    def enbllistbpm(self, value):
        sofb = self.devices['sofb']
        nbpms = sofb._data.nr_bpms
        if len(value) != 2 * nbpms:
            raise ValueError(f'Invalid size! Must be {2 * nbpms}.')
        if all(v in [0, 1, True, False] for v in value):
            value = _np.array(value, dtype=bool)
        else:
            raise ValueError('Values must be boolean (0 / 1 or True / False).')
        sofb.bpmxenbl = value[:nbpms]
        sofb.bpmyenbl = value[nbpms:]

    # #### pbba utils #####

    def set_quad_strengths(self, group_id, strengths, ignore_timeout=False):
        """."""
        bpms = self.data['groups2dopbba'][group_id]
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']
        for strength, bpmname in zip(strengths, bpms):  # noqa: B905
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            quad.strength = strength

        if ignore_timeout:
            return DoParallelBBA.STATUS.Success

        for strength, bpmname in zip(strengths, bpms):  # noqa: B905
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            if not quad.wait_float(
                'KLRef-Mon',
                strength,
                rel_tol=0.0,
                abs_tol=0.05 * self.params.quad_deltakl,
                timeout=self.params.wait_quadrupole,
            ):
                return DoParallelBBA.STATUS.Fail
        return DoParallelBBA.STATUS.Success

    def get_quad_strengths(self, group_id):
        """."""
        bpms = self.data['groups2dopbba'][group_id]
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        strengths = []
        for bpmname in bpms:
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            strengths.append(quad.strength)
        return _np.array(strengths)

    def get_quad_strength_limits(self, group_id, margin=0.0005):
        """."""
        bpms = self.data['groups2dopbba'][group_id]
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        limits = []
        for bpmname in bpms:
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            pv = quad.pv_object('KL-SP')
            upp = pv.upper_disp_limit
            low = pv.lower_disp_limit
            # Limits are interchanged in some quads:
            lolim = min(upp, low) + margin
            hilim = max(upp, low) - margin
            limits.append([lolim, hilim])
        return _np.array(limits, dtype=float)

    def check_isvalid_dkl(self, group_id, init_strengths=None, margin=0.0005):
        """."""
        bpms = self.data['groups2dopbba'][group_id]
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        quadlims = self.get_quad_strength_limits(group_id, margin=margin)
        delta_kl = self.data['delta_kl'][group_id]

        if init_strengths is None:
            strengths = self.get_quad_strengths(group_id)
        else:
            strengths = init_strengths

        ok = True
        for idx, bpmname in enumerate(bpms):
            quadname = quad_names[bpm_names.index(bpmname)]
            stren = strengths[idx]
            dkl = delta_kl[idx]
            lolim, hilim = quadlims[idx]
            low = min(stren + dkl / 2, stren - dkl / 2)
            upp = max(stren + dkl / 2, stren - dkl / 2)
            if upp > hilim or low < lolim:
                max_delta_kl = min(hilim - stren, stren - lolim)
                msg = f'WARN: {quadname} KL = {stren:.2g}, dKL = {abs(dkl):.2g}. '
                msg += f'Limits: ({lolim:.2g}, {hilim:.2g}). Max. dKL = {max_delta_kl * 2:.2g}.'
                self._log_print(msg)
                ok = False
        return ok

    def meas_ios(self, group_id, init_strengths=None):
        """."""
        delta_strens = self.data['delta_kl'][group_id]

        if init_strengths is None:
            strens_orig = self.get_quad_strengths(group_id)
            _time.sleep(self.params.wait_quadrupole)
        else:
            strens_orig = init_strengths

        if not self.set_quad_strengths(
            group_id, strens_orig + delta_strens / 2
        ):
            return None, DoParallelBBA.STATUS.Fail

        orb_pos = self.get_orbit()

        if not self.set_quad_strengths(
            group_id, strens_orig - delta_strens / 2
        ):
            return None, DoParallelBBA.STATUS.Fail

        orb_neg = self.get_orbit()

        if not self.set_quad_strengths(group_id, strens_orig):
            return None, DoParallelBBA.STATUS.Fail

        return orb_pos - orb_neg, DoParallelBBA.STATUS.Success

    def calc_ios_jacobians(self, groups_to_calc=None):  # noqa: C901
        """Calculate the IOS Response Matrices for all groups."""
        model = self.model
        _orbcorr = _OrbitCorr(
            model=model, acc='SI', corr_system='SOFB', use6dtrack=True
        )
        quadindices = self._get_quads_indices_in_model(self.data['quadnames'])

        def _get_or_set_kl(bname, value=None):
            _do = getattr if value is None else setattr
            bidx = self.data['bpmnames'].index(bname)
            qidx = quadindices[bidx]
            if 'QS' == self.data['quadnames'][bidx].dev:
                return _do(model[qidx], 'KsL', value)
            else:
                return _do(model[qidx], 'KL', value)

        def _get_quad_strengths(group):
            strens = []
            for bname in group:
                strens.append(_get_or_set_kl(bname))
            return _np.array(strens)

        def _set_quad_strengths(group, strengths):
            for strength, bname in zip(strengths, group):  # noqa: B905
                _get_or_set_kl(bname, strength)

        jacobians = []
        groups_to_calc = (
            _np.arange(len(self.data['groups2dopbba']))
            if groups_to_calc is None
            else groups_to_calc
        )
        for group_id in groups_to_calc:
            group = self.data['groups2dopbba'][group_id]
            try:
                delta_strens = self.data['delta_kl'][group_id]
            except Exception as e:
                str_msg = 'undefined or empty "delta_kl"'
                str_msg += f' of group {group_id}'
                raise IndexError(str_msg) from e
            strens_orig = _get_quad_strengths(group)

            _set_quad_strengths(group, strens_orig + delta_strens / 2)
            try:
                jac_pos = _orbcorr.get_jacobian_matrix()
            except Exception as err:
                _set_quad_strengths(group, strens_orig)
                raise err

            _set_quad_strengths(group, strens_orig - delta_strens / 2)
            try:
                jac_neg = _orbcorr.get_jacobian_matrix()
            except Exception as err:
                _set_quad_strengths(group, strens_orig)
                raise err

            _set_quad_strengths(group, strens_orig)
            jac = jac_pos - jac_neg
            jac[:, -1] *= 1e6  # rescale: [m/Hz] -> [um/Hz]
            jacobians.append(jac)
        return jacobians

    def analyze_groups(self, analyze_coupling=False):
        """Helper function to analyze the groups' properties."""
        if not self.data['jacobians']:
            raise ValueError('Please calculate and set jacobians first.')

        anl = []
        for group_id in range(len(self.data['groups2dopbba'])):
            print(f'Analyzing group: {group_id:d}')
            anl.append(self.analyze_group(group_id, analyze_coupling))
        return anl

    def analyze_group(self, group_id, analyze_coupling=False):
        """Helper function to analyze group's properties."""
        jacobian = self.data['jacobians'][group_id]
        u_mat, svals, vt_mat = _np.linalg.svd(jacobian)

        model = self.model
        quadindices = self._get_quads_indices_in_model(self.data['quadnames'])
        delta_strens = self.data['delta_kl'][group_id]
        group = self.data['groups2dopbba'][group_id]

        tune_variation = [_pyacc.optics.get_frac_tunes(model)[:2]]

        if analyze_coupling:

            def _get_coupling_parameters():
                rad_on, cav_on = model.radiation_on, model.cavity_on
                model.radiation_on = 0
                model.cavity_on = False
                ed = _pyacc.optics.calc_edwards_teng(model)[0]
                mtsp, ratio = _pyacc.optics.estimate_coupling_parameters(ed)
                model.radiation_on = rad_on
                model.cavity_on = cav_on
                return mtsp, _np.std(ratio)

            min_tunesep, std_ratio = _get_coupling_parameters()
            min_tunesep_variation = [min_tunesep]
            std_ratio_variation = [std_ratio]

        for fac in [1, -2, 1]:
            for dkl, bpm in zip(delta_strens, group):  # noqa: B905
                _id = self.data['bpmnames'].index(bpm)
                qname = self.data['quadnames'][_id]
                qidx = quadindices[_id]
                if 'QS' in qname:
                    model[qidx].KsL += fac * dkl / 2
                else:
                    model[qidx].KL += fac * dkl / 2
                tune_variation.append(_pyacc.optics.get_frac_tunes(model)[:2])
                if analyze_coupling:
                    min_tunesep, std_ratio = _get_coupling_parameters()
                    min_tunesep_variation.append(min_tunesep)
                    std_ratio_variation.append(std_ratio)

        ret = {
            'u_matrix': u_mat,
            'vt_matrix': vt_mat,
            'svals': svals,
            'tune_variation': _np.array(tune_variation),
        }
        if analyze_coupling:
            ret['min_tunesep_variation'] = _np.array(min_tunesep_variation)
            ret['std_ratio_variation'] = _np.array(std_ratio_variation)
        return ret

    def process_data(self):
        """."""
        for group_id in range(len(self.data['groups2dopbba'])):
            self.process_data_single_group(group_id)

    def process_data_single_group(self, group_id):
        """."""
        meas_data = self.data['measure'][group_id]
        bpmnames = self.data['bpmnames']
        nbpms = len(bpmnames)
        orbit = meas_data['orbit_end']

        # #### error estimation ? #####
        # ios_iter = meas_data['ios_iter']
        # ios_init = ios_iter[0]
        # iosx_init, iosy_init = ios_init[:nbpms], ios_init[nbpms:]
        # ios_end = ios_iter[-1]
        # iosx_end, iosy_end = ios_end[:nbpms], ios_end[nbpms:]
        stdx0 = 0.0
        stdy0 = 0.0

        for bpm in meas_data['bpms']:
            bpm_idx = bpmnames.index(bpm)
            self.analysis[bpm] = {
                'x0': orbit[bpm_idx],
                'y0': orbit[bpm_idx + nbpms],
                'stdx0': stdx0,
                'stdy0': stdy0,
            }

    def get_pbba_results(self, error=False):
        """."""
        bpms = self.data['bpmnames']
        bbax = _np.zeros(len(bpms))
        bbay = _np.zeros(len(bpms))
        if error:
            bbaxerr = _np.zeros(len(bpms))
            bbayerr = _np.zeros(len(bpms))
        for idx, bpm in enumerate(bpms):
            res = self.analysis.get(bpm)
            if not res:
                continue
            bbax[idx] = res['x0']
            bbay[idx] = res['y0']
            if error and 'stdx0' in res:
                bbaxerr[idx] = res['stdx0']
                bbayerr[idx] = res['stdy0']
        if error:
            return bbax, bbay, bbaxerr, bbayerr
        return bbax, bbay

    # #### private methods ####
    def _get_quads_indices_in_model(self, quadnames):
        """."""
        fam_data = self.fam_data
        quadindices = []
        for qname in quadnames:
            key = qname.dev
            idx = fam_data[key]['devnames'].index(qname)
            qindex = fam_data[key]['index'][idx]
            qindex = qindex[0] if len(qindex) == 1 else qindex
            quadindices.append(qindex)
        return quadindices

    def _do_pbba(self):
        tini = _datetime.datetime.fromtimestamp(_time.time())
        self._log_print(
            'Starting measurement at {:s}'.format(
                tini.strftime('%Y-%m-%d %Hh%Mm%Ss')
            )
        )

        groups = self.data['groups2dopbba']
        if not all([self.check_isvalid_dkl(g) for g, _ in enumerate(groups)]):
            self._log_print('Adjust quad strength or change dKL first.')
            return

        self.data['jacobians'] = self.calc_ios_jacobians()
        self.data['measure'] = list()

        sofb = self.devices['sofb']
        if sofb.autocorrsts:
            self._log_print(
                '\nSOFB feedback is enabled. Please desable it first.'
            )
            return

        for gid, _ in enumerate(groups):
            if self._stopevt.is_set():
                self._log_print('\nStopped!')
                break
            if not self.havebeam:
                self._log_print('\nBeam was Lost')
                break
            self._log_print('\nCorrecting Orbit... ', end='')
            self.correct_orbit()
            self._log_print('Ok!')
            if not self._dopbba_single_group(gid):
                break

        self._log_print('\nCorrecting Orbit... ', end='')
        self.correct_orbit()
        self._log_print('Ok!')

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split('.')[0]
        self._log_print('\nFinished! Elapsed time {:s}'.format(dtime))

    def _dopbba_single_group(self, group_id):
        """."""
        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime('%Hh%Mm%Ss')
        self._log_print(f'{strtini:s}: Doing PBBA for Group {group_id:d}')

        enblbpm = self.enbllistbpm  # cut jacobian with only enabled bpms
        jac = (self.data['jacobians'][group_id])[enblbpm, :]
        inv_jac = self._calc_inverse_jacobian(jac, group_id)

        group_data = {
            'bpms': self.data['groups2dopbba'][group_id],
            'strengths_init': self.get_quad_strengths(group_id),
            'orbit_init': self.get_orbit(),
            'kicks_init': self.get_kicks(),
            'enbllistbpm': enblbpm.copy(),
        }

        self._log_print('    Cycling:')
        msg, sts = self._do_cycling(group_id, group_data['strengths_init'])
        if not sts:
            self._restore_init_conditions(
                group_id,
                group_data['strengths_init'],
                extra_info_before_message=msg,
            )
            nr_iters = 0
        else:  # proceed to IOS correction
            self._log_print('    Correcting IOS:')
            nr_iters = self.params.corr_max_nr_iters

        ios_iter, dkicks_iter = [], []
        sts = self.STATUS.Fail

        converged = False
        increased = False

        def _reduction(a, a0):
            rms_a0 = _np.std(a0)
            if rms_a0 == 0:
                return _np.inf
            return _np.std(a) / rms_a0

        for i in range(nr_iters):
            self._log_print(
                '        {:02d}/{:02d} --> '.format(i + 1, nr_iters), end=''
            )
            if self._stopevt.is_set():
                self._restore_init_conditions(
                    group_id,
                    group_data['strengths_init'],
                    extra_info_before_message='Measurement stopped. ',
                )
                break
            if not self.havebeam:
                self._restore_init_conditions(
                    group_id,
                    group_data['strengths_init'],
                    extra_info_before_message='Error: beam is off. ',
                )
                break
            ios, sts = self.meas_ios(group_id, group_data['strengths_init'])
            if not sts:
                self._restore_init_conditions(
                    group_id,
                    group_data['strengths_init'],
                    extra_info_before_message='Fail while measuring IOS. ',
                )
                break

            ios_iter.append(ios)  # save ios (all bpms)
            ios = ios[enblbpm]  # use only enabled bpms for correction
            self._log_print(' IOS (rms):', _np.std(ios), '--> ', end='')

            if i > 0:
                dios_p = _reduction(ios, ios_iter[-2][enblbpm])
                if dios_p >= 1.0:  # deactivated
                    prev_dkicks = dkicks_iter[-1]
                    self.set_delta_kicks(-prev_dkicks)
                    self._log_print('Done.', end=' ')
                    increased = True
                    break
                dios_i = _reduction(ios, ios_iter[0][enblbpm])
                if dios_i < self.params.ios_conv_threshold:
                    self._log_print('Done.', end=' ')
                    converged = True
                    break

            dkicks = list(-1 * _np.dot(inv_jac, ios))
            dkicks_iter.append(dkicks)
            self.set_delta_kicks(dkicks)
            self._log_print('Done.')

        if sts and converged:
            self._log_print(f'IOS converged ({i:d} iterations).')

        elif sts and increased:
            msg = f'IOS increased ({i:d} iterations). '
            msg += 'Kicks were restored to the last valid values.'
            self._log_print(msg)

        elif sts and not converged and not increased:
            ios, sts = self.meas_ios(group_id, group_data['strengths_init'])
            if not sts:
                self._restore_init_conditions(
                    group_id,
                    group_data['strengths_init'],
                    extra_info_before_message='Fail while measuring IOS. ',
                )
            else:
                ios_iter.append(ios)
                dios_i = _reduction(ios[enblbpm], ios_iter[0][enblbpm])
                dios_p = _reduction(ios[enblbpm], ios_iter[-2][enblbpm])
                msg = f'Max iterations reached ({i + 1:d})'
                if dios_i < self.params.ios_conv_threshold:
                    msg += ', but IOS converged'
                elif dios_p >= 1.0:
                    msg += ', and IOS increased'
                self._log_print(msg + '.')

        group_data['kicks_end'] = self.get_kicks()
        group_data['ios_iter'] = ios_iter
        group_data['dkicks_iter'] = dkicks_iter
        group_data['orbit_end'] = self.get_orbit()
        group_data['delta_kl'] = self.data['delta_kl'][group_id]
        self.data['measure'].append(group_data)

        self.correct_orbit()

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split('.')[0]
        msg = '    Finished. Status: '
        if sts:
            self._log_print(msg + 'OK! Elapsed time: {:s}'.format(dtime))
        else:
            self._log_print(msg + 'Fail! Elapsed time: {:s}'.format(dtime))
        return sts

    def _calc_inverse_jacobian(self, jacobian, group_id):
        """."""
        u, s, vt = _np.linalg.svd(jacobian, full_matrices=False)
        nr_svals = 2 * len(self.groups2dopbba[group_id])
        i_s = _np.zeros_like(s)
        i_s[:nr_svals] = 1.0 / s[:nr_svals]

        return vt.T @ _np.diag(i_s) @ u.T

    def _do_cycling(self, group_id, init_strengths):
        delta_strengths = self.data['delta_kl'][group_id]
        nr_cycles = self.params.cycling_nr_steps
        for i in range(nr_cycles):
            self._log_print(
                '        {:02d}/{:02d} --> '.format(i + 1, nr_cycles), end=''
            )
            if self._stopevt.is_set():
                return 'Event stopped! ', DoParallelBBA.STATUS.Fail
            if not self.havebeam:
                return 'No beam! ', DoParallelBBA.STATUS.Fail
            if not self.set_quad_strengths(
                group_id, init_strengths + delta_strengths / 2
            ):
                return 'Fail! ', DoParallelBBA.STATUS.Fail
            if not self.set_quad_strengths(
                group_id, init_strengths - delta_strengths / 2
            ):
                return 'Fail! ', DoParallelBBA.STATUS.Fail
            if not self.set_quad_strengths(group_id, init_strengths):
                return 'Fail! ', DoParallelBBA.STATUS.Fail
            self._log_print('Ok')
        return '', DoParallelBBA.STATUS.Success

    def _restore_init_conditions(
        self,
        group_id,
        init_strengths,
        message='Restoring initial conditions and exiting...',
        correct_orbit=True,
        extra_info_before_message='',
    ):
        """."""
        self._log_print(extra_info_before_message + message)

        self.set_quad_strengths(group_id, init_strengths, ignore_timeout=True)

        bpms = self.data['groups2dopbba'][group_id]
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        for strength, bpmname in zip(init_strengths, bpms):  # noqa: B905
            qname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[qname]
            if not quad.wait_float(
                'KLRef-Mon',
                strength,
                rel_tol=0.0,
                abs_tol=0.05 * self.params.quad_deltakl,
                timeout=self.params.wait_quadrupole,
            ):
                self._log_print(
                    f'    {qname}: Could not be restored to initial strength'
                )

        if correct_orbit:
            self.correct_orbit()

    def _log_print(self, msg, *args, **kw):
        """."""
        self.data['log'].append((_time.time(), msg))
        print(msg, *args, **kw)
