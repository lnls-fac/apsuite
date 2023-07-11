"""."""
import time as _time
import logging as _log
import os as _os
import subprocess as _subprocess

import numpy as _np
import scipy.io as _scyio

from pymodels import si as _si
from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoSI, EVG, \
    Event, EGTriggerPS, ASLLRF, InjCtrl

from ..utils import ParamsBaseClass as _Params, \
    ThreadedMeasBaseClass as _BaseClass
from ..optimization.rcds import RCDS as _RCDS, RCDSParams as _RCDSParams
from ..optics_analysis import ChromCorr


class OptimizeDAParams(_RCDSParams):
    """."""

    SEXT_FAMS = (
        'SDA0', 'SDB0', 'SDP0',
        'SFA0', 'SFB0', 'SFP0',
        'SDA1', 'SDB1', 'SDP1',
        'SDA2', 'SDB2', 'SDP2',
        'SDA3', 'SDB3', 'SDP3',
        'SFA1', 'SFB1', 'SFP1',
        'SFA2', 'SFB2', 'SFP2')
    SEXT_FAMS_ACHROM = SEXT_FAMS[:6]
    SEXT_FAMS_CHROM = SEXT_FAMS[6:]

    def __init__(self):
        """."""
        super().__init__()
        self.cold_injection = False
        self.onaxis_rf_phase = 0  # [°]
        self.offaxis_rf_phase = 0  # [°]
        self.offaxis_weight = 1
        self.onaxis_weight = 1
        self.wait_between_injections = 1  # [s]
        self.onaxis_nrpulses = 5
        self.offaxis_nrpulses = 20
        self.offaxis_inj_with_dpkckr = False
        self.offaxis_dpkckr_strength = -3.50  # [mrad]
        self.use_median = False
        self.names_sexts2corr = [
            'SDA2', 'SDB2', 'SDP2', 'SFA2', 'SFB2', 'SFP2']
        self.names_sexts2use = []
        for sext in self.SEXT_FAMS:
            if sext in self.names_sexts2corr:
                continue
            self.names_sexts2use.append(sext)

    def __str__(self):
        """."""
        stg = '-----  RCDS Parameters  -----\n\n'
        stg += super().__str__()
        stg += '\n\n-----  OptimizeDA Parameters  -----\n\n'
        stg += self._TMPF('onaxis_rf_phase', self.onaxis_rf_phase, '[°]')
        stg += self._TMPF('offaxis_rf_phase', self.offaxis_rf_phase, '[°]')
        stg += self._TMPF('offaxis_weight', self.offaxis_weight, '')
        stg += self._TMPF('onaxis_weight', self.onaxis_weight, '')
        stg += self._TMPS(
            'names_sexts2corr', ', '.join(self.names_sexts2corr), '')
        stg += self._TMPS(
            'names_sexts2use', ', '.join(self.names_sexts2use), '')
        stg += self._TMPF(
            'wait_between_injections', self.wait_between_injections, '[s]')
        stg += self._TMPD('onaxis_nrpulses', self.onaxis_nrpulses, '')
        stg += self._TMPD('offaxis_nrpulses', self.offaxis_nrpulses, '')
        stg += self._TMPF(
            'offaxis_dpkckr_strength', self.offaxis_dpkckr_strength, '[mrad]')
        stg += self._TMPS(
            'offaxis_inj_with_dpkckr',
            str(bool(self.offaxis_inj_with_dpkckr)), '')
        stg += self._TMPS('use_median', str(bool(self.use_median)), '')
        return stg


class OptimizeDA(_RCDS):
    """."""

    def __init__(self, isonline=True, use_thread=True):
        """."""
        _RCDS.__init__(self, isonline=isonline, use_thread=use_thread)
        self.params = OptimizeDAParams()
        self.data['strengths'] = []
        self.data['obj_funcs'] = []
        self.data['onaxis_obj_funcs'] = []
        self.data['offaxis_obj_funcs'] = []

        self.sextupoles = []
        if self.isonline:
            self._create_devices()
        self.chrom_corr = ChromCorr(_si.create_accelerator(), acc='SI')

        self.full_chrom_mat = _np.zeros(
            (2, len(self.params.SEXT_FAMS)), dtype=float)

        idcs = [
            self.params.SEXT_FAMS.index(sx)
            for sx in self.chrom_corr.knobs.all]
        self.full_chrom_mat[:, idcs] = self.chrom_corr.calc_jacobian_matrix()

    def _initialization(self):
        """."""
        if not super()._initialization():
            return False
        self.data['timestamp'] = _time.time()
        self.data['strengths'] = [self.get_strengths_from_machine()]
        self.data['obj_funcs'] = []
        self.data['onaxis_obj_funcs'] = []
        self.data['offaxis_obj_funcs'] = []
        return True

    def objective_function(self, pos=None):
        """."""
        if not self.params.offaxis_weight and not self.params.onaxis_weight:
            raise ValueError('At least one weigth must be nonzero')

        if pos is not None:
            strengths = self.get_isochrom_strengths(pos)
            self.set_strengths_to_machine(strengths)
        else:
            strengths = self.get_strengths_from_machine()
        self.data['strengths'].append(strengths)

        if self._stopevt.is_set():
            return 0.0

        injeff_offaxis = 0.0
        if self.params.offaxis_weight:
            injeff_offaxis = self.inject_beam_offaxis()

        if self._stopevt.is_set():
            return 0.0

        injeff_onaxis = 0.0
        if self.params.onaxis_weight:
            injeff_onaxis = self.inject_beam_onaxis()

        objective = self.params.offaxis_weight * injeff_offaxis
        objective += self.params.onaxis_weight * injeff_onaxis
        objective /= self.params.offaxis_weight + self.params.onaxis_weight
        self.data['obj_funcs'].append(objective)
        self.data['offaxis_obj_funcs'].append(injeff_offaxis)
        self.data['onaxis_obj_funcs'].append(injeff_onaxis)
        return -objective

    def inject_beam_offaxis(self):
        """."""
        injctrl = self.devices['injctrl']
        nr_pulses = self.params.offaxis_nrpulses

        if not self.params.offaxis_inj_with_dpkckr:
            if injctrl.pumode_mon != injctrl.PUModeMon.Optimization:
                injctrl.cmd_change_pumode_to_optimization()
                _time.sleep(1.0)
        else:
            self.devices['pingh'].set_strength(
                self.params.offaxis_dpkckr_strength, tol=0.2, timeout=13,
                wait_mon=True)
            
        if self.params.cold_injection:
            injeffs = self.inject_beam_and_get_injeff_cold_config(
                nrpulses=nr_pulses)
        else:
            injeffs = self.inject_beam_and_get_injeff(nrpulses=nr_pulses)

        self.data['offaxis_obj_funcs'].append(injeffs)
        fun = _np.median if self.params.use_median else _np.mean
        return fun(injeffs)

    def inject_beam_onaxis(self):
        """."""
        injctrl = self.devices['injctrl']
        llrf = self.devices['llrf']
        nr_pulses = self.params.onaxis_nrpulses

        if injctrl.pumode_mon != injctrl.PUModeMon.OnAxis:
            injctrl.cmd_change_pumode_to_onaxis()
            _time.sleep(1.0)

        llrf.set_phase(self.params.onaxis_rf_phase, wait_mon=True)
        _time.sleep(0.5)

        if self.params.cold_injection:
            injeffs = self.inject_beam_and_get_injeff_cold_config(
                nrpulses=nr_pulses)
        else:
            injeffs = self.inject_beam_and_get_injeff(nrpulses=nr_pulses)


        llrf.set_phase(self.params.offaxis_rf_phase, wait_mon=True)
        _time.sleep(0.5)

        self.data['onaxis_obj_funcs'].append(injeffs)
        fun = _np.median if self.params.use_median else _np.mean
        return fun(injeffs)

    def inject_beam_and_get_injeff(self, get_injeff=True, nrpulses=1):
        """Inject beam and get injected current, if desired."""
        inj0 = self.devices['currinfo'].injeff
        self.devices['evg'].set_nrpulses(nrpulses)
        self.devices['evg'].cmd_turn_on_injection(wait_rb=True)
        if not get_injeff:
            self.devices['evg'].wait_injection_finish()
            return

        injeffs = []
        cnt = nrpulses
        for _ in range(5 * nrpulses * 2):
            injn = self.devices['currinfo'].injeff
            if inj0 != injn:
                inj0 = injn
                injeffs.append(injn)
                cnt -= 1
            if cnt == 0:
                break
            _time.sleep(0.1)
        else:
            _log.warning('Timed out waiting injeff to update.')

        self.devices['evg'].wait_injection_finish()
        return injeffs

    def inject_beam_and_get_injeff_cold_config(self, nrpulses=1):
        """Inject beam and get injected current."""
        self.devices['evg'].set_nrpulses(1)
        injeffs = []
        for _ in range(nrpulses):
            self.devices['evg'].cmd_turn_on_injection(wait_rb=True)
            _time.sleep(self.params.wait_between_injections)
            injn = self.devices['currinfo'].injeff
            injeffs.append(injn)
        self.devices['evg'].wait_injection_finish()
        return injeffs

    def get_isochrom_strengths(self, pos):
        """."""
        idcs_corr = [
            self.params.SEXT_FAMS.index(sx)
            for sx in self.params.names_sexts2corr]
        idcs_2use = [
            self.params.SEXT_FAMS.index(sx)
            for sx in self.params.names_sexts2use]

        mat2corr = self.full_chrom_mat[:, idcs_corr]
        imat2corr = _np.linalg.pinv(mat2corr, rcond=1e-15)

        mat2use = self.full_chrom_mat[:, idcs_2use]

        str0 = self.data['strengths'][0].copy()
        pos0 = str0[idcs_2use]
        dpos = pos - pos0
        dchrom = mat2use @ dpos
        dcorr = - imat2corr @ dchrom

        strengths = _np.full(len(self.params.SEXT_FAMS), _np.nan)
        strengths[idcs_2use] = pos
        strengths[idcs_corr] = dcorr + str0[idcs_corr]
        return strengths

    def measure_objective_function_noise(self, nr_evals, pos=None):
        """."""
        obj = []
        for i in range(nr_evals):
            obj.append(self.objective_function(pos))
            _log.info(f'{i+1:02d}/{nr_evals:02d}  --> obj. = {obj[-1]:.3f}')
        noise_level = _np.std(obj)
        _log.info(f'obj. = {_np.mean(obj):.3f} +- {noise_level:.3f}')
        self.data['measured_objfuncs_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level
        return noise_level, obj

    def get_current_position(self, return_limits=False):
        """Return current strengths of sextupoles used by RCDS.

        Returns:
            numpy.ndarray (N, 1): strengths of the N sextupoles used in
                optimization.

        """
        stren, (lower0, upper0) = self.get_strengths_from_machine(
            return_limits=True)
        pos, lower, upper = [], [], []
        for sxt, stg, low, upp in zip(
                self.params.SEXT_FAMS, stren, lower0, upper0):
            if sxt in self.params.names_sexts2use:
                pos.append(stg)
                lower.append(low)
                upper.append(upp)
        if not return_limits:
            return _np.array(pos)
        return _np.array(pos), (_np.array(lower), _np.array(upper))

    def get_strengths_from_machine(self, return_limits=False):
        """."""
        val, lower, upper = [], [], []
        for sxt in self.sextupoles:
            val.append(sxt.strengthref_mon)
            if not return_limits:
                continue
            lims = sxt.pv_object('SL-Mon').get_ctrlvars()
            upper.append(lims['upper_disp_limit'])
            lower.append(lims['lower_disp_limit'])
        if not return_limits:
            return _np.array(val)
        return _np.array(val), (_np.array(lower), _np.array(upper))

    def set_strengths_to_machine(self, strengths):
        """."""
        if len(strengths) != len(self.sextupoles):
            raise ValueError(
                'Length of strengths must match number of sextupole families.')

        for i, stg in enumerate(strengths):
            if stg is None or _np.isnan(stg):
                continue
            self.sextupoles[i].strength = stg

        # NOTE: the loop below waits sextupoles to reach the set current.
        for i, stg in enumerate(strengths):
            if stg is None or _np.isnan(stg):
                continue
            self.sextupoles[i].set_strength(
                stg, tol=1e-3, timeout=10, wait_mon=True)

    def _create_devices(self):
        for fam in self.params.SEXT_FAMS:
            sext = PowerSupply('SI-Fam:PS-'+fam)
            self.devices[fam] = sext
            self.sextupoles.append(sext)

        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['nlk'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_NLKCKR)
        self.devices['pingv'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_PING_V)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['evt_study'] = Event('Study')
        self.devices['egun_trigps'] = EGTriggerPS()
        self.devices['llrf'] = ASLLRF(ASLLRF.DEVICES.SI)
        self.devices['injctrl'] = InjCtrl()


# -----------------------------------------------------------------------------
#         Implements Server  to optimize Dynap with external optimizer
#                        (used during Xiaobiao's visit)
# -----------------------------------------------------------------------------
class DynapServerParams(_Params):
    """."""

    _TMPD = '{:30s}: {:10d} {:s}\n'.format
    _TMPF = '{:30s}: {:10.3f} {:s}\n'.format
    _TMPS = '{:30s}: {:10s} {:s}\n'.format

    SEXT_FAMS = (
        'SDA0', 'SDB0', 'SDP0',
        'SFA0', 'SFB0', 'SFP0',
        'SDA1', 'SDB1', 'SDP1',
        'SDA2', 'SDB2', 'SDP2',
        'SDA3', 'SDB3', 'SDP3',
        'SFA1', 'SFB1', 'SFP1',
        'SFA2', 'SFB2', 'SFP2')
    SEXT_FAMS_ACHROM = SEXT_FAMS[:6]
    SEXT_FAMS_CHROM = SEXT_FAMS[6:]

    def __init__(self):
        """."""
        super().__init__()
        self.client = 'localhost'
        self.folder = '/home/sirius/shared/screens-iocs/data_by_day/'
        self.folder += '2023-02-23-SI_nonlinear_optics_opt_RCDS_matlab/'
        self.folder += 'run1/'
        self.input_fname = 'input.mat'
        self.output_fname = 'output.mat'
        self.onaxis_rf_phase = 0  # [°]
        self.offaxis_rf_phase = 0  # [°]
        self.onaxis_nrpulses = 5
        self.offaxis_nrpulses = 20
        self.offaxis_inj_with_dpkckr = False
        self.offaxis_dpkckr_strength = -3.50  # [mrad]
        self.use_median = False
        self.is_a_toy_run = False

    def __str__(self):
        """."""
        stg = self._TMPS('client', self.client, '')
        stg = self._TMPS('folder', self.folder, '')
        stg += self._TMPS('input_fname', self.input_fname, '')
        stg += self._TMPS('output_fname', self.output_fname, '')
        stg += self._TMPF('onaxis_rf_phase', self.onaxis_rf_phase, '[°]')
        stg += self._TMPF('offaxis_rf_phase', self.offaxis_rf_phase, '[°]')
        stg += self._TMPD('onaxis_nrpulses', self.onaxis_nrpulses, '')
        stg += self._TMPD('offaxis_nrpulses', self.offaxis_nrpulses, '')
        stg += self._TMPF(
            'offaxis_dpkckr_strength', self.offaxis_dpkckr_strength, '[mrad]')
        stg += self._TMPS(
            'offaxis_inj_with_dpkckr',
            str(bool(self.offaxis_inj_with_dpkckr)), '')
        stg += self._TMPS('use_median', str(bool(self.use_median)), '')
        stg += self._TMPS('is_a_toy_run', str(bool(self.is_a_toy_run)), '')
        return stg


class DynapServer(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(
            self, isonline=isonline, target=self.run_server)
        self.params = DynapServerParams()

        self.sextupoles = []
        if self.isonline:
            self._create_devices()

    def _initialization(self):
        """."""
        self.data['timestamp'] = _time.time()
        strn = self.get_strengths_from_machine()
        self.data['strengths'] = [strn]
        self.data['strengths_rel'] = [_np.zeros(strn.size)]
        self.data['strengths_received'] = [_np.zeros(strn.size)]
        self.data['initial_strengths'] = self.get_strengths_from_machine()
        self.data['obj_funcs'] = []
        self.data['onaxis_obj_funcs'] = []
        self.data['offaxis_obj_funcs'] = []
        return True

    def run_server(self):
        """."""
        self._initialization()

        while not self._stopevt.is_set():
            input_fname = _os.path.join(
                self.params.folder, self.params.input_fname)
            input_data = self._read_file_get_inputs(input_fname)

            if self._stopevt.is_set():
                continue

            if self.params.is_a_toy_run:
                output = self.toy_objective_function(**input_data)
            else:
                output = self.objective_function(**input_data)
            out = {'injeff_offaxis': output[0], 'injeff_onaxis': output[1]}

            if self._stopevt.is_set():
                continue
            output_fname = _os.path.join(
                self.params.folder, self.params.output_fname)
            self._write_output_to_file(output_fname, out)

    def toy_objective_function(self, **res):
        """."""
        strn = _np.array(list(res['strengths'].values()))
        self.data['strengths_rel'].append(strn)
        return _np.sum(strn*strn), 0.0

    def objective_function(
            self, strengths=None, offaxis_flag=True, onaxis_flag=True,
            relative=True):
        """."""
        if strengths is not None:
            if relative:
                self.set_relative_strengths_to_machine(strengths)
                self.data['strengths_rel'].append(strengths)
            else:
                self.set_strengths_to_machine(strengths)

            self.data['strengths_received'].append(strengths)
            _time.sleep(1)
            self.data['strengths'].append(self.get_strengths_from_machine())

        if self._stopevt.is_set():
            return 0.0, 0.0

        injeff_offaxis = 0.0
        if offaxis_flag:
            injeff_offaxis = self.inject_beam_offaxis()

        if self._stopevt.is_set():
            return 0.0, 0.0

        injeff_onaxis = 0.0
        if onaxis_flag:
            injeff_onaxis = self.inject_beam_onaxis()

        return injeff_offaxis, injeff_onaxis

    def inject_beam_offaxis(self):
        """."""
        injctrl = self.devices['injctrl']
        nr_pulses = self.params.offaxis_nrpulses

        if not self.params.offaxis_inj_with_dpkckr:
            if injctrl.pumode_mon != injctrl.PUModeMon.Optimization:
                injctrl.cmd_change_pumode_to_optimization()
                _time.sleep(1.0)
        else:
            self.devices['pingh'].set_strength(
                self.params.offaxis_dpkckr_strength, tol=0.2, timeout=13,
                wait_mon=True)

        injeffs = self.inject_beam_and_get_injeff(nrpulses=nr_pulses)

        self.data['offaxis_obj_funcs'].append(injeffs)
        fun = _np.median if self.params.use_median else _np.mean
        return fun(injeffs)

    def inject_beam_onaxis(self):
        """."""
        injctrl = self.devices['injctrl']
        llrf = self.devices['llrf']
        nr_pulses = self.params.onaxis_nrpulses

        if injctrl.pumode_mon != injctrl.PUModeMon.OnAxis:
            injctrl.cmd_change_pumode_to_onaxis()
            _time.sleep(1.0)

        llrf.set_phase(self.params.onaxis_rf_phase, wait_mon=True)
        _time.sleep(0.5)

        injeffs = self.inject_beam_and_get_injeff(nrpulses=nr_pulses)

        llrf.set_phase(self.params.offaxis_rf_phase, wait_mon=True)
        _time.sleep(0.5)

        self.data['onaxis_obj_funcs'].append(injeffs)
        fun = _np.median if self.params.use_median else _np.mean
        return fun(injeffs)

    def inject_beam_and_get_injeff(self, get_injeff=True, nrpulses=1):
        """Inject beam and get injected current, if desired."""
        inj0 = self.devices['currinfo'].injeff
        self.devices['evg'].set_nrpulses(nrpulses)
        self.devices['evg'].cmd_turn_on_injection(wait_rb=True)
        if not get_injeff:
            self.devices['evg'].wait_injection_finish()
            return

        injeffs = []
        cnt = nrpulses
        for _ in range(5 * nrpulses * 2):
            injn = self.devices['currinfo'].injeff
            if inj0 != injn:
                inj0 = injn
                injeffs.append(injn)
                cnt -= 1
            if cnt == 0:
                break
            _time.sleep(0.1)
        else:
            _log.warning('Timed out waiting injeff to update.')

        self.devices['evg'].wait_injection_finish()
        return injeffs

    def measure_objective_function_noise(
            self, nr_evals, onaxis_flag=True, offaxis_flag=True):
        """."""
        self._initialization()
        obj = []
        for i in range(nr_evals):
            if self._stopevt.is_set():
                break
            obj.append(self.objective_function(
                onaxis_flag=onaxis_flag, offaxis_flag=offaxis_flag))
            _log.info(
                f'{i+1:02d}/{nr_evals:02d}  --> '
                f'obj. = ({obj[-1][0]:.3f}, {obj[-1][1]:.3f})')
        noise_level = _np.std(obj, axis=0)
        self.data['measured_objfuncs_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level
        return noise_level, obj

    def get_strengths_from_machine(self, return_limits=False):
        """."""
        val, lower, upper = [], [], []
        for sxt in self.sextupoles:
            val.append(sxt.strengthref_mon)
            if not return_limits:
                continue
            lims = sxt.pv_object('SL-Mon').get_ctrlvars()
            upper.append(lims['upper_disp_limit'])
            lower.append(lims['lower_disp_limit'])
        if not return_limits:
            return _np.array(val)
        return _np.array(val), (_np.array(lower), _np.array(upper))

    def set_relative_strengths_to_machine(self, strengths):
        """."""
        initial = self.data['initial_strengths']
        strengths = initial * (1 + strengths)
        self.set_strengths_to_machine(strengths)

    def set_strengths_to_machine(self, strengths):
        """."""
        for i, fam in enumerate(self.params.SEXT_FAMS):
            stg = strengths.get(fam)
            if stg is None or _np.isnan(stg):
                continue
            self.sextupoles[i].strength = stg

        # NOTE: the loop below waits sextupoles to reach the set current.
        for i, stg in enumerate(strengths):
            if stg is None or _np.isnan(stg):
                continue
            self.sextupoles[i].set_strength(
                stg, tol=1e-3, timeout=10, wait_mon=True)

    def _create_devices(self):
        for fam in self.params.SEXT_FAMS:
            sext = PowerSupply('SI-Fam:PS-'+fam)
            self.devices[fam] = sext
            self.sextupoles.append(sext)

        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['nlk'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_NLKCKR)
        self.devices['pingv'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_PING_V)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['evt_study'] = Event('Study')
        self.devices['egun_trigps'] = EGTriggerPS()
        self.devices['llrf'] = ASLLRF(ASLLRF.DEVICES.SI)
        self.devices['injctrl'] = InjCtrl()

    def _read_file_get_inputs(self, fname, remove=True):
        cnt = 0
        while not self._stopevt.is_set():
            if not self._isfile(fname):
                _time.sleep(0.2)
                cnt += 1
                if not cnt % 100:
                    _log.info(f'Waiting input for {cnt*0.2:.2f} s...')
                continue
            cnt = 0
            _log.info('Input file found.')
            res = self._load_file(fname)
            if remove:
                self._remove_file(fname)
                _log.info('Deleting input file.')
            break
        if self._stopevt.is_set():
            return
        fams = [str(s[0]) for s in res['g_fam'].ravel()]
        data = {}
        data['offaxis_flag'] = bool(res['offaxis_flag'][0][0])
        data['onaxis_flag'] = bool(res['onaxis_flag'][0][0])
        data['relative'] = bool(res.get('relative_flag', [[1]])[0][0])
        data['strengths'] = {f: v for f, v in zip(fams, res['dk_21'].ravel())}
        return data

    # ------- Auxiliary Methods to load and save input and output files -------
    def _write_output_to_file(self, fname, output):
        _log.info('Writing output file...')
        self._save_file(fname, output)
        _log.info('Done!')

    def _listdir(self, folder):
        if self.params.client.startswith('local'):
            return _os.listdir(folder)

        res = _subprocess.run(
            ['ssh', self.params.client, 'ls ' + folder],
            stdout=_subprocess.PIPE)
        stdout = res.stdout.split()
        return stdout

    def _isfile(self, fname):
        if self.params.client.startswith('local'):
            return _os.path.isfile(fname)

        res = _subprocess.run(
            ['ssh', self.params.client, 'file ' + fname],
            stdout=_subprocess.PIPE)
        stdout = res.stdout.split()
        return b'cannot' not in stdout

    def _load_file(self, fname):
        if not self.params.client.startswith('local'):
            _subprocess.run(
                ['scp', self.params.client + ':' + fname, './'],
                stdout=_subprocess.PIPE)
            *_, fname = fname.split('/')
        res = _scyio.loadmat(fname, appendmat=False)
        return res

    def _remove_file(self, fname):
        if self.params.client.startswith('local'):
            _os.remove(fname)
            return
        _subprocess.run(
            ['ssh', self.params.client, 'rm', '-rf', fname],
            stdout=_subprocess.PIPE)

    def _save_file(self, fname, out):
        if self.params.client.startswith('local'):
            _scyio.savemat(fname, out, appendmat=False)
            return

        *_, fname_ = fname.split('/')
        _scyio.savemat(fname_, out, appendmat=False)
        _subprocess.run(
            ['scp', './' + fname_, self.params.client + ':' + fname],
            stdout=_subprocess.PIPE)
