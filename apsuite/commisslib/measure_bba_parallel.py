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
    METHODS = _get_namedtuple("Method", ["rms", "ptp"])

    def __init__(self):
        """."""
        super().__init__()

        self.quad_deltakl = 0.01  # [1/m]

        self.wait_correctors = 1.0  # [s]
        self.wait_quadrupole = 2.0  # [s]

        self.corr_max_nr_iters = 8
        self.ios_conv_tol = 0.5  # [um]
        self._conv_method = self.METHODS.rms
        self.ios_raise_cnt_limit = 2

        self.sofb_nrpoints = 80
        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 5  # [um]
        self.timeout_wait_orbit = 12  # [s]

        self.cycling_nr_steps = 2

    def __str__(self):
        """."""
        stg = ''
        stg += f'quad_deltakl       = {self.quad_deltakl:.3f}\n'
        stg += f'wait_correctors    = {self.wait_correctors:.3f}\n'
        stg += f'wait_quadrupole    = {self.wait_quadrupole:.3f}\n'
        stg += f'timeout_wait_orbit = {self.timeout_wait_orbit:.3f}\n'
        stg += f'corr_max_nr_iters  = {self.corr_max_nr_iters:d}\n'
        stg += f'ios_conv_tol       = {self.ios_conv_tol:.2e}\n'
        stg += 'conv_method        = ' + \
            f'{self.METHODS._fields[self._conv_method]:s} (RMS or PTP)\n'
        stg += f'sofb_nrpoints      = {self.sofb_nrpoints:d}\n'
        stg += f'sofb_maxcorriter   = {self.sofb_maxcorriter:d}\n'
        stg += f'sofb_maxorberr     = {self.sofb_maxorberr:.3f}\n'
        stg += f'cycling_nr_steps   = {self.cycling_nr_steps:d}\n'
        return stg

    @property
    def conv_method(self):
        """."""
        return self._conv_method

    @property
    def conv_method_str(self):
        """."""
        return self.METHODS._fields[self._conv_method]

    @conv_method_str.setter
    def conv_method_str(self, value):
        """."""
        self.conv_method = value

    @conv_method.setter
    def conv_method(self, value):
        if isinstance(value, str) and value.lower() in self.METHODS._fields:
            self._conv_method = self.METHODS._fields.index(value.lower())
        elif 0 <= value < len(self.METHODS._fields):
            self._conv_method = int(value)
        else:
            raise ValueError(
                'Invalid method! Select: (int) 0 or 1 | (str) "rms" or "ptp"'
            )

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
        self.params = ParallelBBAParams()
        super().__init__(
            params=self.params, target=self._do_pbba, isonline=isonline
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
        cinfo = self.devices['currinfosi']
        return cinfo.connected and cinfo.storedbeam

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
        self.data['delta_kl'] = _np.array(_dcopy(value))

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
            self._log('\n     Undefined model... setting a default one')
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
        if not value.cavity_on or value.radiation_on != 1:
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
        orb = _np.hstack([sofb.orbx, sofb.orby])
        sofb.nr_points = nrpts

        return orb

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
        _time.sleep(self.params.wait_correctors)

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
        value = _np.array(value, dtype=bool)
        sofb.bpmxenbl = value[:nbpms]
        sofb.bpmyenbl = value[nbpms:]

    # #### pbba utils #####

    def set_quad_strengths(self, group_id, strengths, wait_refmon=True):
        """."""
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        if group_id is None or group_id == 'All':
            bpms = bpm_names
        else:
            bpms = self.data['groups2dopbba'][group_id]

        if len(bpms) != len(strengths):
            msg = 'Size mismatch between the group and strengths: '
            msg += f'{len(bpms)} != {len(strengths)}!'
            raise ValueError(msg)

        for strength, bpmname in zip(strengths, bpms):  # noqa: B905
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            quad.strength = strength

        if not wait_refmon:
            return DoParallelBBA.STATUS.Success

        t0_ = _time.time()
        for strength, bpmname in zip(strengths, bpms):  # noqa: B905
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            dt_ = self.params.wait_quadrupole - (_time.time() - t0_)
            if dt_ <= 0 or not quad.wait_float(
                'KLRef-Mon',
                strength,
                rel_tol=0.0,
                abs_tol=0.05 * self.params.quad_deltakl,
                timeout=dt_,
            ):
                return DoParallelBBA.STATUS.Fail
        return DoParallelBBA.STATUS.Success

    def get_quad_strengths(self, group_id):
        """."""
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        if group_id is None or group_id == 'All':
            bpms = bpm_names
        else:
            bpms = self.data['groups2dopbba'][group_id]

        strengths = []
        for bpmname in bpms:
            quadname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[quadname]
            strengths.append(quad.strength)

        return _np.array(strengths, dtype=float)

    def get_quad_strength_limits(self, group_id, margin=0.0005):
        """."""
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        if group_id == 'All' or group_id is None:
            bpms = bpm_names
        else:
            bpms = self.data['groups2dopbba'][group_id]

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

    def check_isvalid_dkl(
            self,
            group_id,
            init_strengths=None,
            strength_limits=None,
            margin=0.0005,
            return_valid=False,
        ):
        """."""
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        bpms = (
            bpm_names if group_id is None or group_id == 'All'
            else self.data['groups2dopbba'][group_id]
        )

        strengths = (
            self.get_quad_strengths(group_id)
            if init_strengths is None else init_strengths
        )

        lims = (
            self.get_quad_strength_limits(group_id, margin=margin)
            if strength_limits is None else strength_limits
        )

        if len(bpms) != len(strengths):
            msg = 'Size mismatch between the group and init_strengths: '
            msg += f'{len(bpms)} != {len(strengths)}!'
            raise ValueError(msg)

        if len(strengths) != len(lims):
            msg = 'Size mismatch between init_strengths and strength_limits: '
            msg += f'{len(strengths)} != {len(lims)}!'
            raise ValueError(msg)

        ok = True
        valid = strengths.copy()
        for idx, bpm in enumerate(bpms):
            quadname = quad_names[bpm_names.index(bpm)]
            stren = strengths[idx]
            _gid = [
                True if bpm in gp else False
                for gp in self.data['groups2dopbba']
            ].index(True)
            _gp = self.data['groups2dopbba'][_gid]
            dkl = abs(self.data['delta_kl'][_gid][_gp.index(bpm)])
            lolim, hilim = lims[idx]
            clow, chigh = lolim + dkl / 2, hilim - dkl / 2
            if clow > chigh:
                self._log(f'ERR: {quadname}, dKL = {dkl:.3g} too high!')
                valid[idx] = None
            else:
                valid[idx] = _np.clip(stren, clow, chigh)
            if valid[idx] != stren:
                msg = f'WARN: {quadname}, '
                msg += f'KL = {stren:.3g}, '
                msg += f'dKL = {dkl:.3g}, '
                msg += f'limits = ({lolim:.3g}, {hilim:.3g}). '
                msg += f'Change KL to: {valid[idx]}'
                self._log(msg)
                ok = False
        if return_valid:
            return ok, valid
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
            qname = self.data['quadnames'][bidx]
            qidx = quadindices[bidx]
            att = 'KsL' if 'QS' in qname else 'KL'
            return _do(model[qidx], att, value)

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
        u_mat, svals, vt_mat = _np.linalg.svd(jacobian, full_matrices=False)

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
                return mtsp, _np.mean(ratio)

            min_tunesep, emit_ratio = _get_coupling_parameters()
            min_tunesep_variation = [min_tunesep]
            emit_ratio_variation = [emit_ratio]

        for fac in [1, -2, 1]:
            for dkl, bpm in zip(delta_strens, group):  # noqa: B905
                _id = self.data['bpmnames'].index(bpm)
                qname = self.data['quadnames'][_id]
                qidx = quadindices[_id]
                att = 'KsL' if 'QS' in qname else 'KL'
                ele = model[qidx]
                setattr(ele, att, getattr(ele, att) + fac * dkl / 2)
                tune_variation.append(_pyacc.optics.get_frac_tunes(model)[:2])
                if analyze_coupling:
                    min_tunesep, emit_ratio = _get_coupling_parameters()
                    min_tunesep_variation.append(min_tunesep)
                    emit_ratio_variation.append(emit_ratio)

        ret = {
            'u_matrix': u_mat,
            'vt_matrix': vt_mat,
            'svals': svals,
            'tune_variation': _np.array(tune_variation),
        }
        if analyze_coupling:
            ret['min_tunesep_variation'] = _np.array(min_tunesep_variation)
            ret['emit_ratio_variation'] = _np.array(emit_ratio_variation)
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
        msg = 'Starting measurement at '
        self._log(msg + tini.strftime('%Y-%m-%d %Hh%Mm%Ss'))

        groups = self.data['groups2dopbba']
        isvalid = all([
            self.check_isvalid_dkl(gid) for gid, _ in enumerate(groups)
        ])
        if not isvalid:
            self._log('Adjust quad strength or change dKL first.')
            return

        self.data['jacobians'] = self.calc_ios_jacobians()
        self.data['measure'] = list()

        sofb = self.devices['sofb']
        if sofb.autocorrsts:
            self._log('\nSOFB feedback is enabled. Please desable it first.')
            return

        for gid, _ in enumerate(groups):
            if self._stopevt.is_set():
                self._log('\nStopped!')
                break
            if not self.havebeam:
                self._log('\nBeam was Lost')
                break
            self._log('\nCorrecting Orbit... ', end='')
            self.correct_orbit()
            self._log('Ok!')
            if not self._dopbba_single_group(gid):
                break

        self._log('\nCorrecting Orbit... ', end='')
        self.correct_orbit()
        self._log('Ok!')

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini)
        dtime = dtime.split('.')[0]
        self._log('\nFinished! Elapsed time {:s}'.format(dtime))

    def _dopbba_single_group(self, group_id):
        tini = _datetime.datetime.fromtimestamp(_time.time())
        strtini = tini.strftime('%Hh%Mm%Ss')
        self._log(f'{strtini:s}: Doing PBBA for Group {group_id:d}')

        enblbpm = self.enbllistbpm  # cut jacobian with only enabled bpms
        jac = (self.data['jacobians'][group_id])[enblbpm, :]
        inv_jac = self._calc_inverse_jacobian(jac, group_id)

        strengths_init = self.get_quad_strengths(group_id)
        group_data = {
            'bpms': self.data['groups2dopbba'][group_id],
            'strengths_init': strengths_init,
            'orbit_init': self.get_orbit(),
            'kicks_init': self.get_kicks(),
            'enbllistbpm': enblbpm.copy(),
        }

        sts = self._do_cycling(group_id, strengths_init)
        nr_iters = self.params.corr_max_nr_iters
        if not sts:
            nr_iters = 0
            self._restore_conditions(
                group_id, strengths_init, 'Error: Failed during cycling'
            )
        else:  # proceed to IOS correction
            self._log('Correcting IOS:', tab=1)

        ios_iter, dkicks_iter, residue_iter = [], [], []

        converged = False
        increased = 0
        tolerance = self.params.ios_conv_tol
        comp_func = _np.ptp \
            if self.params.conv_method == self.params.METHODS.ptp else _np.std

        for i in range(nr_iters):
            self._log(f'{i + 1:02d}/{nr_iters:02d} --> ', tab=2, end='')
            if self._stopevt.is_set():
                self._restore_conditions(
                    group_id, strengths_init, 'Measurement stopped.'
                )
                sts = DoParallelBBA.STATUS.Fail
                break
            if not self.havebeam:
                self._restore_conditions(
                    group_id, strengths_init, 'Error: dont have beam.'
                )
                sts = DoParallelBBA.STATUS.Fail
                break
            ios, sts = self.meas_ios(group_id, strengths_init)
            if not sts:
                self._restore_conditions(
                    group_id, strengths_init, 'Fail while measuring IOS.'
                )
                break

            ios_iter.append(ios)  # save ios (all bpms)
            ios = ios[enblbpm]  # use only enabled bpms for correction
            residue = comp_func(ios)
            residue_iter.append(residue)
            msg = ' IOS (' + ('ptp'
                if self.params.conv_method == self.params.METHODS.ptp
                else 'rms') + '): '
            msg += f'{residue:.3f} [um] --> '
            self._log(msg, end='')

            if residue < tolerance:
                converged = True
                break
            elif i > 0 and residue > residue_iter[-2]:
                increased += 1
                if increased > self.params.ios_raise_cnt_limit:
                    self.set_delta_kicks(-dkicks_iter[-1])
                    break

            dkicks = -1 * _np.dot(inv_jac, ios)
            dkicks_iter.append(dkicks)
            self.set_delta_kicks(dkicks)
            self._log('Done.')

        if sts and converged:
            self._log(f'IOS converged ({i:d} iterations).')

        elif sts and increased:
            self._log(f'IOS increased ({i:d} iterations).')
            self._log('Kicks were restored to the last valid values.')

        elif sts and not converged and not increased:
            ios, sts = self.meas_ios(group_id, strengths_init)
            if not sts:
                self._restore_conditions(
                    group_id, strengths_init, 'Fail while measuring IOS.'
                )
            else:
                ios_iter.append(ios)
                residue = comp_func(ios[enblbpm])
                residue_iter.append(residue)
                msg = f'Max iterations reached ({i + 1:d})'
                if residue < tolerance:
                    msg += ', but IOS converged'
                elif residue > residue_iter[-2]:
                    msg += ', and IOS increased'
                    self.set_delta_kicks(-dkicks_iter[-1])
                self._log(msg + '.')

        group_data['kicks_end'] = self.get_kicks()
        group_data['ios_iter'] = ios_iter
        group_data['dkicks_iter'] = dkicks_iter
        group_data['residue_iter'] = residue_iter
        group_data['orbit_end'] = self.get_orbit()
        group_data['delta_kl'] = self.data['delta_kl'][group_id]
        self.data['measure'].append(group_data)

        self.correct_orbit()

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini).split('.')[0]
        if sts:
            self._log('Finished. Status: OK! ET: ' + dtime)
        else:
            self._log('Finished. Status: Fail! ET: ' + dtime)
        return sts

    def _calc_inverse_jacobian(self, jacobian, group_id):
        u, s, vt = _np.linalg.svd(jacobian, full_matrices=False)
        nr_svals = 2 * len(self.groups2dopbba[group_id])
        i_s = _np.zeros_like(s)
        i_s[:nr_svals] = 1.0 / s[:nr_svals]

        return vt.T @ _np.diag(i_s) @ u.T

    def _do_cycling(self, group_id, init_strengths):
        self._log('Cycling:', tab=1)
        kl = init_strengths
        dkl = self.data['delta_kl'][group_id]
        nr_cycles = self.params.cycling_nr_steps
        for i in range(nr_cycles):
            self._log(f'{i + 1:02d}/{nr_cycles:02d} --> ', tab=2, end='')
            if self._stopevt.is_set():
                self._log('Event stopped!')
                return DoParallelBBA.STATUS.Fail
            if not self.havebeam:
                self._log('Error: dont have beam!')
                return DoParallelBBA.STATUS.Fail
            if not self.set_quad_strengths(group_id, kl + dkl / 2):
                self._log('Fail!')
                return DoParallelBBA.STATUS.Fail
            if not self.set_quad_strengths(group_id, kl - dkl / 2):
                self._log('Fail!')
                return DoParallelBBA.STATUS.Fail
            if not self.set_quad_strengths(group_id, init_strengths):
                self._log('Fail!')
                return DoParallelBBA.STATUS.Fail
            self._log('Ok!')
        return DoParallelBBA.STATUS.Success

    def _restore_conditions(
        self,
        group_id,
        strengths,
        info='',
        message='Restoring conditions and exiting...',
        correct_orbit=True,
    ):
        info += ' ' if info else ''
        self._log(info + message)

        self.set_quad_strengths(group_id, strengths, ignore_timeout=True)

        bpms = self.data['groups2dopbba'][group_id]
        quad_names = self.data['quadnames']
        bpm_names = self.data['bpmnames']

        for strength, bpmname in zip(strengths, bpms):  # noqa: B905
            qname = quad_names[bpm_names.index(bpmname)]
            quad = self.devices[qname]
            if not quad.wait_float(
                'KLRef-Mon',
                strength,
                rel_tol=0.0,
                abs_tol=0.05 * self.params.quad_deltakl,
                timeout=self.params.wait_quadrupole,
            ):
                self._log(f'{qname}: could not restore strength!')

        if correct_orbit:
            self.correct_orbit()

    def _log(self, msg, *args, **kwargs):
        end = kwargs.pop('end', '\n')
        tab = kwargs.pop('tab', 0)
        time = _time.time()
        msg = '    ' * tab + msg + end
        self.data['log'].append((time, msg))
        kwargs['end'] = ''
        print(msg, *args, **kwargs)
