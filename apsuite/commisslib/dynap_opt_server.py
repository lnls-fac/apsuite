"""."""
import time as _time
import logging as _log
import os as _os

import subprocess as _subprocess
import numpy as _np
import scipy.io as _scyio

from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoSI, EVG, \
    Event, EGTriggerPS, ASLLRF, InjCtrl

from ..utils import ParamsBaseClass as _Params, \
    ThreadedMeasBaseClass as _BaseClass


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

        if injctrl.pumode_mon != injctrl.PUModeMon.Optimization:
            injctrl.cmd_change_pumode_to_optimization()
            _time.sleep(1.0)

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
        for i, fam in enumerate(self.params.SEXT_FAMS):
            stg = strengths.get(fam)
            if stg is None or _np.isnan(stg):
                continue
            self.sextupoles[i].strength = initial[i]*(1 + stg)
        _time.sleep(2)

    def set_strengths_to_machine(self, strengths):
        """."""
        initial = self.data['initial_strengths']
        for i, fam in enumerate(self.params.SEXT_FAMS):
            stg = strengths.get(fam)
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
