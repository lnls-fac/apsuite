"""."""
import time as _time
import logging as _log
import os as _os

import numpy as _np
import scipy.io as _scyio

from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoSI, EVG, \
    Event, EGTriggerPS, ASLLRF, InjCtrl

from ..utils import ParamsBaseClass as _Params, \
    ThreadedMeasBaseClass as _BaseClass


class DynapServerParams(_Params):
    """."""

    _TMPD = '{:30s}: {:10d} {:s}\n'
    _TMPF = '{:30s}: {:10.3f} {:s}\n'
    _TMPS = '{:30s}: {:10s} {:s}\n'

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
        self.folder = '/home/sirius/shared/screens-iocs/data_by_day/'
        self.folder += '2023-02-23-SI_nonlinear_optics_opt_RCDS_matlab/'
        self.folder += 'run1/'
        self.input_fname = 'input.mat'
        self.output_fname = 'output.mat'
        self.onaxis_rf_phase = 0  # [°]
        self.offaxis_rf_phase = 0  # [°]
        self.wait_between_injections = 1  # [s]
        self.onaxis_nrpulses = 5
        self.offaxis_nrpulses = 5

    def __str__(self):
        """."""
        stg = self._TMPS.format('folder', self.folder, '')
        stg += self._TMPS.format('input_fname', self.input_fname, '')
        stg += self._TMPS.format('output_fname', self.output_fname, '')
        stg += self._TMPF.format(
            'onaxis_rf_phase', self.onaxis_rf_phase, '[°]')
        stg += self._TMPF.format(
            'offaxis_rf_phase', self.offaxis_rf_phase, '[°]')
        stg += self._TMPF.format(
            'wait_between_injections', self.wait_between_injections, '[s]')
        stg += self._TMPF.format('onaxis_nrpulses', self.onaxis_nrpulses, '')
        stg += self._TMPF.format('offaxis_nrpulses', self.offaxis_nrpulses, '')
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
        self.data['strengths'] = [self.get_strengths_from_machine()]
        self.data['initial_strengths'] = self.get_strengths_from_machine()
        self.data['obj_funcs'] = []
        self.data['onaxis_obj_funcs'] = []
        self.data['offaxis_obj_funcs'] = []
        self._prepare_evg()
        return True

    def _read_file_get_inputs(self, fname, remove=True):
        cnt = 0
        while not self._stopevt.is_set():
            if not _os.path.isfile(fname):
                _time.sleep(0.2)
                cnt += 1
                if not cnt % 100:
                    _log.info(f'Waiting input for {cnt*0.2:.2f} s...')
                continue
            cnt = 0
            _log.info('Input file found.')
            res = _scyio.loadmat(fname, appendmat=False)
            if remove:
                _os.remove(fname)
                _log.info('Deleting input file.')
            break
        fams = [str(s[0]) for s in res['g_fam'][0]]
        data = {}
        data['offaxis_flag'] = bool(res['offaxis_flag'][0][0])
        data['onaxis_flag'] = bool(res['onaxis_flag'][0][0])
        data['strengths'] = {f: v for f, v in zip(fams, res['dk_21'][0])}
        return data

    def _write_output_to_file(self, fname, output):
        _log.info('Writing output file...')
        out = {'injeff_offaxis': output[0], 'injeff_onaxis': output[1]}
        _scyio.savemat(fname, out, appendmat=False)
        _log.info('Done!')

    def run_server(self):
        """."""
        self._initialization()

        while not self._stopevt.is_set():
            input_fname = _os.path.join(
                self.params.folder, self.params.input_fname)
            input_data = self._read_file_get_inputs(input_fname)
            output = self.toy_objective_function(**input_data)
            output_fname = _os.path.join(
                self.params.folder, self.params.output_fname)
            self._write_output_to_file(output_fname, output)

    def toy_objective_function(self, **res):
        """."""
        strn = _np.array(list(res['strengths'].values()))
        return _np.sum(strn*strn), 0.0

    def objective_function(self, **res):
        """."""
        strengths = res.get('strenghts')
        if strengths is not None:
            self.set_relative_strengths_to_machine(strengths)
            self.data['strengths'].append(strengths)
            _time.sleep(1)

        injeff_offaxis = 0.0
        offaxis_flag = res.get('offaxis_flag', True)
        if offaxis_flag:
            injeff_offaxis = self.inject_beam_offaxis()

        injeff_onaxis = 0.0
        onaxis_flag = res.get('onaxis_flag', True)
        if onaxis_flag:
            injeff_onaxis = self.inject_beam_onaxis()

        return injeff_offaxis, injeff_onaxis

    def inject_beam_offaxis(self):
        """."""
        injctrl = self.devices['injctrl']
        nr_pulses = self.params.offaxis_nrpulses

        if injctrl.pumode_mon != injctrl.PUModeMon.Optimization:
            injctrl.cmd_change_pumode_to_optimization()
            _time.sleep(1.0)

        injeffs = []
        for _ in range(nr_pulses):
            if self._stopevt.is_set():
                break
            injeffs.append(self.inject_beam_and_get_injeff())
            _time.sleep(self.params.wait_between_injections)

        self.data['offaxis_obj_funcs'].append(injeffs)
        return injeffs

    def inject_beam_onaxis(self):
        """."""
        injctrl = self.devices['injctrl']
        llrf = self.devices['llrf']
        nr_pulses = self.params.onaxis_nrpulses

        if injctrl.pumode_mon != injctrl.PUModeMon.OnAxis:
            injctrl.cmd_change_pumode_to_onaxis()
            _time.sleep(1.0)

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

            llrf.set_phase(self.params.onaxis_rf_phase, wait_mon=True)
            _time.sleep(0.5)
            injeffs.append(self.inject_beam_and_get_injeff())
            llrf.set_phase(self.params.offaxis_rf_phase, wait_mon=True)

            _time.sleep(self.params.wait_between_injections)

        self.data['onaxis_obj_funcs'].append(injeffs)
        return injeffs

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

    def measure_objective_function_noise(
            self, nr_evals, onaxis_flag=True, offaxis_flag=True):
        """."""
        self._initialization()
        obj = []
        for i in range(nr_evals):
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
        for i, fam in enumerate(self.params.SEXT_FAMS):
            stg = strengths.get(fam)
            if stg is None or _np.isnan(stg):
                continue
            self.sextupoles[i].strength = initial[i]*(1 + stg)
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
