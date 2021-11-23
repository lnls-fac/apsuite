"""Main module."""
import time as _time
from threading import Event as _Flag

import numpy as _np
import scipy.signal as _scysig
import scipy.integrate as _scyint
import scipy.optimize as _scyopt

import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs

from siriuspy.devices import StrengthConv, PowerSupply, BPM, CurrInfoSI, \
    Trigger, Event, EVG, RFGen
from siriuspy.sofb.csdev import SOFBFactory

from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class ACORMParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.nr_points = 5500
        self.ch_kick = 5  # [urad]
        self.cv_kick = 5  # [urad]
        self.delay_cm = 50e3
        self.frequencies = self.find_primes(30)

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format
        st = ftmp('center_frequency  [Hz]', self.center_frequency, '')
        st += ftmp('bandwidth [Hz]', self.bandwidth, '')
        st += stmp('filter_type', self.filter_type, '[gauss or sinc]')
        st += stmp('acqtype', self.acqtype, '[SRAM or BRAM]')
        return st

    @staticmethod
    def find_primes(n_primes, start=3):
        primes = []
        i = start
        while True:
            for j in range(3, int(_np.sqrt(i))+1, 2):
                if not (i % j):
                    break
            else:
                if (i % 2):
                    primes.append(i)
            i += 1
            if len(primes) >= n_primes:
                break
        return _np.array(primes)


class MeasACORM(_ThreadBaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        params = ACORMParams()
        super().__init__(
            params=params, target=self._do_measure, isonline=isonline)
        self.bpms = dict()
        self._flags = dict()

        if self.isonline:
            self._create_devices()

    def _create_devices(self):
        self.sofb_data = SOFBFactory.create('SI')

        # Create objects to convert kicks to current
        self.devices.update({
            n+':StrengthConv': StrengthConv(n, 'Ref-Mon')
            for n in self.sofb_data.ch_names})
        self.devices.update({
            n+':StrengthConv': StrengthConv(n, 'Ref-Mon')
            for n in self.sofb_data.cv_names})

        # Create objects to interact with correctors
        self.devices.update({
            nme: PowerSupply(nme) for nme in self.sofb_data.ch_names})
        self.devices.update({
            nme: PowerSupply(nme) for nme in self.sofb_data.cv_names})

        # Create object to get stored current
        self.devices['currinfo'] = CurrInfoSI()

        # Create BPMs trigger:
        self.devices['trigbpms'] = Trigger('SI-Fam:TI-BPM')

        # Create Correctors Trigger:
        self.devices['trigcorrs'] = Trigger('SI-Glob:TI-Mags-Corrs')

        # Create event to start data acquisition sinchronously:
        self.devices['evt_study'] = Event('Study')
        self.devices['evg'] = EVG()

        # Create RF generator object
        self.devices['rfgen'] = RFGen()

        # Create BPMs
        self.bpms = dict()
        name = self.sofb_data.bpm_names[0]
        self.bpms[name] = BPM(name)
        self.csbpm = self.bpms[name].csdata

        propties_not_interested = self.bpms[name].auto_monitor_status
        propties_to_keep = ['GEN_XArrayData', 'GEN_YArrayData']
        for ppt in propties_to_keep:
            propties_not_interested.pop(ppt)

        for i, name in enumerate(self.sofb_data.bpm_names):
            if not i:
                bpm = self.bpms[name]
            else:
                bpm = BPM(name)
                self.bpms[name] = bpm
            for propty in propties_not_interested:
                bpm.set_auto_monitor(propty, False)
            for propty in propties_to_keep:
                bpm.set_auto_monitor(propty, True)
                pvo = bpm.pv_object(propty)
                self._flags[pvo.pvname] = _Flag()
                pvo.add_callback(self._set_flag)
        self.devices.update(self.bpms)

    def _set_flag(self, pvname, **kwargs):
        _ = kwargs
        self._flags[pvname].set()

    def _reset_flags(self):
        for flag in self._flags.values():
            flag.clear()

    def wait_bpms_update(self, timeout=10):
        for name, flag in self._flags.items():
            t0 = _time.time()
            if not flag.wait(timeout=timeout):
                print(f'Timed out in PV {name:s}')
                return False
            timeout -= _time.time() - t0
        return True

    def process_data(self):
        """."""
        infos = self.data['infos']
        data = self.data['modes_data']

        analysis = dict(coeffs=[], tim_fits=[], fits=[], modes_filt=[])
        for i, (info, datum) in enumerate(zip(infos, data)):
            print('.', end='')
            if not ((i+1) % 80):
                print()
            tim = info['time']
            freq = info['dft_freq']
            interval = self.estimate_fitting_intervals(
                info, clearance_ini=self.params.fitting_clearance_ini,
                clearance_fin=self.params.fitting_clearance_fin)
            coeff, tim_fit, fit = self.fit_growth_rates(
                tim, datum, interval=interval,
                offset=self.params.fitting_consider_offset,
                full=True)

            mode_filt = self.filter_data(freq, datum, self.params)

            analysis['coeffs'].append(coeff)
            analysis['tim_fits'].append(tim_fit)
            analysis['fits'].append(fit)
            analysis['modes_filt'].append(mode_filt)
        self.analysis = analysis

    def get_orbit(self):
        orbx, orby = [], []
        for num, bpm in enumerate(self.sofb_data.bpm_names):
            orbx.append(bpm.mt_posx)
            orby.append(bpm.mt_posy)
        return _np.array(orbx).T, _np.array(orby).T

    def get_data(self, chs_used, cvs_used):
        orbx, orby = self._get_orbit()
        bpm0 = list(self.bpms.values())[0]
        csbpm = self.csbpm
        data = dict()
        data['timestamp'] = _time.time()
        data['rf_frequency'] = self.devices['rfgen'].frequency
        data['stored_current'] = self.devices['currinfo'].current
        data['orbx'] = orbx
        data['orby'] = orby
        data['bpms_acq_rate'] = csbpm.AcqChan._fields[bpm0.acq_channel]
        data['bpms_nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['bpms_nrsamples_post'] = bpm0.acq_nrsamples_post
        data['bpms_trig_delay_raw'] = self.devices['trigbpm'].delay_raw
        data['bpms_switching_mode'] = csbpm.SwModes._fields[
            bpm0.switching_mode]

        data.update({
            'ch_names': [], 'ch_amplitudes': [], 'ch_offsets': [],
            'ch_kick_amplitudes': [], 'ch_kick_offsets': [],
            'ch_frequency': [], 'ch_num_cycles': [], 'ch_cycle_type': [],
            'cv_names': [], 'cv_amplitudes': [], 'cv_offsets': [],
            'cv_kick_amplitudes': [], 'cv_kick_offsets': [],
            'cv_frequency': [], 'cv_num_cycles': [], 'cv_cycle_type': [],
            })
        for cmn in chs_used:
            data['ch_names'].append(cmn)
            cm = self.devices[cmn]
            conv = self.devices[cmn+':StrengthConv'].conv_current_2_strength
            data['ch_amplitudes'].append(cm.cycle_ampl)
            data['ch_offsets'].append(cm.cycle_offset)
            data['ch_kick_amplitudes'].append(conv(cm.cycle_ampl))
            data['ch_kick_offsets'].append(conv(cm.cycle_offset))
            data['ch_frequency'].append(cm.cycle_freq)
            data['ch_num_cycles'].append(cm.cycle_num_cycles)
            data['ch_cycle_type'].append(cm.cycle_type_str)

        for cmn in cvs_used:
            data['cv_names'].append(cmn)
            cm = self.devices[cmn]
            conv = self.devices[cmn+':StrengthConv'].conv_current_2_strength
            data['cv_amplitudes'].append(cm.cycle_ampl)
            data['cv_offsets'].append(cm.cycle_offset)
            data['cv_kick_amplitudes'].append(conv(cm.cycle_ampl))
            data['cv_kick_offsets'].append(conv(cm.cycle_offset))
            data['cv_frequency'].append(cm.cycle_freq)
            data['cv_num_cycles'].append(cm.cycle_num_cycles)
            data['cv_cycle_type'].append(cm.cycle_type_str)

        data['corrs_trig_delay_raw'] = self.devices['trigcorr'].delay_raw
        return data

    def _config_bpms(self, nr_points, acq_rate=None):
        acq_rate = self.csbpm.AcqChan.Monit1 if acq_rate is None else acq_rate

        print('Aborting...')
        self._cmd_bpms_acq_abort()
        print('Aborted! Restarting...')

        for bpm in self.bpms.values():
            bpm.acq_repeat = self.csbpm.AcqRepeat.Repetitive
            bpm.acq_channel = acq_rate
            bpm.acq_trigger = self.csbpm.AcqTrigTyp.External
            bpm.acq_nrsamples_pre = 0
            bpm.acq_nrsamples_post = nr_points

        self._cmd_bpms_acq_start()
        print('Done')

    def _cmd_bpms_acq_abort(self, bpms):
        for bpm in bpms:
            bpm.acq_ctrl = self.csbpm.AcqEvents.Abort
        states = {self.csbpm.AcqStates.Aborted, self.csbpm.AcqStates.Idle}
        for bpm in bpms:
            boo = bpm._wait('ACQStatus-Sts', states, comp=lambda x, y: x in y)
            if not boo:
                print('Timed out waiting abort.')

    def _cmd_bpms_acq_start(self, bpms):
        for bpm in bpms:
            bpm.acq_ctrl = self.csbpm.AcqEvents.Start
        for bpm in bpms:
            boo = bpm._wait(
                'ACQStatus-Sts', self.csbpm.AcqStates.External_Trig)
            if not boo:
                print('Timed out waiting start.')

    def _config_timing(self, cm_delay):
        trigbpm = self.devices['trigbpms']
        trigcorr = self.devices['trigcorrs']
        evt_study = self.devices['evt_study']
        evg = self.devices['evg']

        trigbpm.delay = 0.0
        trigbpm.nr_pulses = 1
        trigbpm.source = 'Study'

        trigcorr.delay = cm_delay
        trigcorr.nr_pulses = 1
        trigcorr.source = 'Study'

        # configure event Study to be in External mode
        evt_study.delay = 0
        evt_study.mode = 'External'

        # update event configurations in EVG
        evg.cmd_update_events()

    def config_correctors(self, corr_names, kick, freq_vector, time_duration):
        for i, cmn in enumerate(corr_names):
            cm = self.devices[cmn]
            conv = self.device[cmn+':StrengthConv']
            cm.cycle_type = cm.CYCLETYPE.Sine
            cm.cycle_freq = freq_vector[i]
            cm.cycle_ampl = conv.conv_strength_2_current(kick)
            cm.cycle_offset = cm.currentref_mon
            cm.cycle_theta_begin = 0
            cm.cycle_theta_end = 0
            cm.cycle_num_cycles = int(time_duration * freq_vector[i])

    def change_corrs_opmode(self, mode, corr_names=None, timeout=10):
        if corr_names is None:
            corr_names = self.sofb_data.ch_names + self.sofb_data.cv_names

        for cmn in corr_names:
            cm = self.devices[cmn]
            opm = cm.OPMODE_SEL
            cm.opmode = opm.Cycle if mode == 'cycle' else opm.SlowRef

        for _ in range(int(timeout)):
            ok = True
            for cmn in corr_names:
                cm = self.devices[cmn]
                opm = cm.OPMODE_SEL
                mode = opm.Cycle if mode == 'cycle' else opm.SlowRef
                ok &= cm.opmode == mode
            if ok:
                return True
            _time.sleep(0.2)
        return False

    @staticmethod
    def shift_list(lst, num):
        return lst[-num:] + lst[:-num]

    def _do_measure(self):
        elt0 = _time.time()

        self._config_bpms(self.params.nr_points)
        self._config_timing(self.params.delay_cm)

        # Shift correctors so first corrector is 01M1
        chs_shifted = self.shift_list(self.sofb_data.ch_names, 1)
        cvs_shifted = self.shift_list(self.sofb_data.cv_names, 1)

        # set operation mode to slowref
        if not self.change_corrs_opmode('slowref'):
            print('Problem: Correctors not in SlowRef mode.')
            return

        exc_duration = 1e-3 * self.params.nr_points * 0.95
        ch_kick = self.params.ch_kick
        cv_kick = self.params.cv_kick
        freqh = self.params.ch_freqs
        freqv = self.params.cv_freqs

        self.data = []
        for sector in self.params.sectors_to_measure:
            elt = _time.time()

            print(f'Sector {sector:02d}...', end='')
            slch = slice((sector-1)*6, sector*6)
            slcv = slice((sector-1)*8, sector*8)
            chs_slc = chs_shifted[slch]
            cvs_slc = cvs_shifted[slcv]

            # configure correctors
            self.config_correctors(chs_slc, ch_kick, freqh, exc_duration)
            self.config_correctors(cvs_slc, cv_kick, freqv, exc_duration)

            # set operation mode to cycle
            if not self.change_corrs_opmode('cycle', chs_slc + cvs_slc):
                print('Problem: Correctors not in Cycle mode.')
                break

            # send event through timing system to start acquisitions
            self._reset_flags()
            self.devices['evt_study'].cmd_external_trigger()

            if not self.wait_bpms_update(timeout=self.params.timeout):
                print('Problem: timed out waiting BPMs update.')
                break

            # get data
            self.data.append(self.get_data(chs=chs_slc, cvs=cvs_slc))

            # set operation mode to slowref
            if not self.change_corrs_opmode('slowref', chs_slc + cvs_slc):
                print('Problem: Correctors not in SlowRef mode.')
                break

            elt -= _time.time()
            elt *= -1
            print(f',      ET: {elt:.2f}s')
            if self._stopevt.is_set():
                print('Stopping...')
                break
        elt0 -= _time.time()
        elt0 *= -1
        print(f'Finished!!  ET: {elt0/60:.2f}min')
