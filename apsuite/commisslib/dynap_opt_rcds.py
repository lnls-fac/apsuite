"""."""
import time as _time
import logging as _log

import numpy as _np

from pymodels import si as _si
from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoSI, EVG, \
    Event, EGTriggerPS, ASLLRF, InjCtrl

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
        self.onaxis_rf_phase = 0  # [°]
        self.offaxis_rf_phase = 0  # [°]
        self.offaxis_weight = 1
        self.onaxis_weight = 1
        self.wait_between_injections = 1  # [s]
        self.onaxis_nrpulses = 5
        self.offaxis_nrpulses = 20
        self.offaxis_nrpulses_discard = 4
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
        stg += self._TMPD(
            'offaxis_nrpulses_discard', self.offaxis_nrpulses_discard, '')
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
        self._prepare_evg()
        return True

    def objective_function(self, pos=None):
        """."""
        if not self.params.offaxis_weight and not self.params.onaxis_weight:
            raise ValueError('At least one weigth must be nonzero')

        if pos is not None:
            strengths = self.get_isochrom_strengths(pos)
            self.set_strengths_to_machine(strengths)
            self.data['strengths'].append(strengths)
            _time.sleep(1)

        injeff_offaxis = 0.0
        if self.params.offaxis_weight:
            injeff_offaxis = self.inject_beam_offaxis()

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
        nr_pulses_discard = self.params.offaxis_nrpulses_discard

        if injctrl.pumode_mon != injctrl.PUModeMon.Optimization:
            injctrl.cmd_change_pumode_to_optimization()
            _time.sleep(1.0)

        injeffs = []
        self.devices['egun_trigps'].cmd_disable_trigger()
        _time.sleep(0.5)
        get_injeff = False
        for i in range(nr_pulses):
            if self._stopevt.is_set():
                break
            if i == nr_pulses_discard:
                get_injeff = True
                self.devices['egun_trigps'].cmd_enable_trigger()
                _time.sleep(0.5)

            injeff = self.inject_beam_and_get_injeff(get_injeff)
            if injeff is not None:
                injeffs.append(injeff)
            _time.sleep(self.params.wait_between_injections)

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

        injeffs = []
        for _ in range(nr_pulses):
            if self._stopevt.is_set():
                break
            self.devices['egun_trigps'].cmd_disable_trigger()
            _time.sleep(1.0)
            self.inject_beam_and_get_injeff(get_injeff=False)
            _time.sleep(1.0)
            self.devices['egun_trigps'].cmd_enable_trigger()
            _time.sleep(1.0)

            injeffs.append(self.inject_beam_and_get_injeff())
            _time.sleep(self.params.wait_between_injections)

        llrf.set_phase(self.params.offaxis_rf_phase, wait_mon=True)
        _time.sleep(0.5)

        self.data['onaxis_obj_funcs'].append(injeffs)
        fun = _np.median if self.params.use_median else _np.mean
        return fun(injeffs)

    def inject_beam_and_get_injeff(self, get_injeff=True):
        """Inject beam and get injected current, if desired."""
        inj0 = self.devices['currinfo'].injeff
        self.devices['evg'].cmd_turn_on_injection(wait_rb=True)
        self.devices['evg'].wait_injection_finish()
        if not get_injeff:
            return

        for _ in range(50):
            if inj0 != self.devices['currinfo'].injeff:
                break
            _time.sleep(0.1)
        else:
            _log.warning('Timed out waiting injeff to update.')
        return self.devices['currinfo'].injeff

    def _prepare_evg(self):
        evg = self.devices['evg']
        # configure to inject on first bucket just once
        evg.bucketlist = [1]
        evg.nrpulses = 1
        evg.cmd_update_events()
        _time.sleep(1)

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
        self.params.noise_level = noise_level
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
        _time.sleep(2)

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
