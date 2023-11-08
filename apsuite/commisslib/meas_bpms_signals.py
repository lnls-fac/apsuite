"""."""
import numpy as _np
import scipy.fft as _sp_fft
import time as _time

from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from siriuspy.devices import Tune, CurrInfoSI, \
    Trigger, Event, EVG, RFGen, FamBPMs


class AcqBPMsSignalsParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        self.trigbpm_delay = 0.0
        self.trigbpm_nrpulses = 1
        self.do_pulse_evg = True
        self.timing_event = 'Study'
        self.event_delay = 0.0
        self.event_mode = 'External'
        self.timeout = 40
        self.nrpoints_before = 0
        self.nrpoints_after = 20000
        self.acq_rate = 'FAcq'
        self.acq_repeat = False
        self.signals2acq = 'XY'

    def __str__(self):
        """."""
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:26s} = {1:9}  {2:s}\n'.format
        stg = ''
        stg += ftmp('trigbpm_delay', self.trigbpm_delay, '[us]')
        stg += dtmp('trigbpm_nrpulses', self.trigbpm_nrpulses, '')
        stg += stmp('do_pulse_evg', str(self.do_pulse_evg), '')
        stg += stmp('timing_event', self.timing_event, '')
        stg += ftmp('event_delay', self.event_delay, '[us]')
        stg += stmp('event_mode', self.event_mode, '')
        stg += ftmp('timeout', self.timeout, '[s]')
        stg += dtmp('nrpoints_before', self.nrpoints_before, '')
        stg += dtmp('nrpoints_after', self.nrpoints_after, '')
        stg += stmp('acq_rate', self.acq_rate, '')
        stg += dtmp('acq_repeat', self.acq_repeat, '')
        stg += stmp('signals2acq', str(self.signals2acq), '')
        return stg

    def from_dict(self, params_dict):
        """."""
        dic = dict()
        for key, val in params_dict.items():
            if key.startswith('orbit_'):  # compatibility with old data
                key = key.replace('orbit_', '')
            dic[key] = val
        return super().from_dict(dic)


class AcqBPMsSignals(_BaseClass):
    """."""

    BPM_TRIGGER = 'SI-Fam:TI-BPM'
    PSM_TRIGGER = 'SI-Fam:TI-BPM-PsMtm'

    def __init__(self, isonline=True, ispost_mortem=False):
        """."""
        super().__init__(params=AcqBPMsSignalsParams(), isonline=isonline)
        self._ispost_mortem = ispost_mortem

        if self.isonline:
            self.create_devices()

    def load_and_apply(self, fname: str):
        """Load and apply `data` and `params` from pickle or HDF5 file.

        Args:
            fname (str): name of the pickle file. If extension is not provided,
                '.pickle' will be added and a pickle file will be assumed.
                If provided, must be '.pickle' for pickle files or
                {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.

        """
        ret = super().load_and_apply(fname)
        data = dict()
        for key, val in self.data.items():
            if key.startswith('bpms_'):  # compatibility with old data
                key = key.replace('bpms_', '')
            data[key] = val
        self.data = data
        return ret

    def create_devices(self):
        """."""
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['fambpms'] = FamBPMs(
            devname=FamBPMs.DEVICES.SI, ispost_mortem=self._ispost_mortem,
            props2init='acq')
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        trigname = self.BPM_TRIGGER
        if self._ispost_mortem:
            trigname = self.PSM_TRIGGER
        self.devices['trigbpm'] = Trigger(trigname)
        self.devices['evt_study'] = Event('Study')
        self.devices['evg'] = EVG()
        self.devices['rfgen'] = RFGen()

    def get_timing_state(self):
        """."""
        trigbpm = self.devices['trigbpm']
        evt_study = self.devices['evt_study']
        state = dict()
        state['trigbpm_source'] = trigbpm.source
        state['trigbpm_nrpulses'] = trigbpm.nr_pulses
        state['trigbpm_delay'] = trigbpm.delay
        state['evt_study_delay'] = evt_study.delay
        state['evt_study_mode'] = evt_study.mode
        return state

    def recover_timing_state(self, state):
        """."""
        self.prepare_timing(state)

    def prepare_timing(self, state=None):
        """."""
        state = dict() if state is None else state

        trigbpm = self.devices['trigbpm']
        trigbpm.delay = state.get('trigbpm_delay', self.params.trigbpm_delay)
        trigbpm.nr_pulses = state.get(
            'trigbpm_nrpulses', self.params.trigbpm_nrpulses)
        src = state.get('trigbpm_source', self.params.timing_event)
        trigbpm.source = src

        if src != 'Study':
            return
        evt_study = self.devices['evt_study']
        evt_study.delay = state.get('evt_study_delay', self.params.event_delay)
        evt_study.mode = state.get('evt_study_mode', self.params.event_mode)

        # Update event configurations in EVG
        self.devices['evg'].cmd_update_events()

    def trigger_timing_signal(self):
        """."""
        if not self.params.do_pulse_evg:
            return
        if self.params.timing_event == 'Study':
            self.devices['evt_study'].cmd_external_trigger()
        else:
            self.devices['evg'].cmd_turn_on_injection()

    def prepare_bpms_acquisition(self):
        """."""
        fambpms = self.devices['fambpms']
        prms = self.params
        fambpms.mturn_signals2acq = self.params.signals2acq
        return fambpms.mturn_config_acquisition(
            nr_points_after=prms.nrpoints_after,
            nr_points_before=prms.nrpoints_before,
            acq_rate=prms.acq_rate, repeat=prms.acq_repeat)

    def acquire_data(self):
        """."""
        fambpms = self.devices['fambpms']
        ret = self.prepare_bpms_acquisition()
        tag = self._bpm_tag(idx=abs(ret)-1)
        if ret < 0:
            print(tag + ' did not finish last acquisition.')
        elif ret > 0:
            print(tag + ' is not ready for acquisition.')

        fambpms.mturn_reset_flags_and_update_initial_timestamps()
        self.trigger_timing_signal()

        time0 = _time.time()
        ret = fambpms.mturn_wait_update(timeout=self.params.timeout)
        print(f'it took {_time.time()-time0:02f}s to update bpms')
        if ret != 0:
            print(f'There was a problem with acquisition')
            if ret > 0:
                tag = self._bpm_tag(idx=ret-1)
                print('This BPM did not update: ' + tag)
            elif ret == -1:
                print('Initial timestamps were not defined')
            return
        self.data = self.get_data()

    def get_data(self):
        """Get Orbit and auxiliary data."""
        fbpms = self.devices['fambpms']
        mturn_orbit = fbpms.get_mturn_signals()

        data = dict()
        data['ispost_mortem'] = self._ispost_mortem
        data['timestamp'] = _time.time()
        rf_freq = self.devices['rfgen'].frequency
        data['rf_frequency'] = rf_freq
        data['stored_current'] = self.devices['currinfo'].current

        if list(self.params.signals2acq) != list(fbpms.mturn_signals2acq):
            raise ValueError('signals2acq was not configured properly.')
        elif len(mturn_orbit) != len(fbpms.mturn_signals2acq):
            raise ValueError(
                'Lenght of signals2acq does not match signals acquired.')
        for i, sig in enumerate(self.params.signals2acq):
            if sig.upper() == 'X':
                data['orbx'] = mturn_orbit[i]
            elif sig.upper() == 'Y':
                data['orby'] = mturn_orbit[i]
            elif sig.upper() == 'S':
                data['sumdata'] = mturn_orbit[i]
            elif sig.upper() == 'Q':
                data['posq'] = mturn_orbit[i]
            else:
                data['ampl'+sig.lower()] = mturn_orbit[i]

        tune = self.devices['tune']
        data['tunex'], data['tuney'] = tune.tunex, tune.tuney
        bpm0 = fbpms.devices[0]
        data['acq_rate'] = bpm0.acq_channel_str
        data['sampling_frequency'] = fbpms.get_sampling_frequency(rf_freq)
        data['nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['nrsamples_post'] = bpm0.acq_nrsamples_post
        data['trig_delay_raw'] = self.devices['trigbpm'].delay_raw
        data['switching_mode'] = bpm0.switching_mode_str
        data['switching_frequency'] = fbpms.get_switching_frequency(rf_freq)
        data['tunex_enable'] = tune.enablex
        data['tuney_enable'] = tune.enabley
        return data

    @staticmethod
    def filter_data_frequencies(
            orb, fmin, fmax, fsampling, keep_within_range=True):
        """Filter acquisition matrix considering a frequency range.

        Args:
            matrix (numpy.array): 2d-array with timesamples along rows and
            BPMs indices along columns.
            fmin (float): minimum frequency in range.
            fmax (float): maximum frequency in range.
            fsampling (float): sampling frequency on matrix
            keep_within_range (bool, optional): Defaults to True.

        Returns:
            filtered matrix (numpy.array): same structure as matrix.

        """
        dft = _sp_fft.rfft(orb, axis=0)
        freq = _sp_fft.rfftfreq(orb.shape[0], d=1/fsampling)
        if keep_within_range:
            idcs = (freq < fmin) | (freq > fmax)
            dft[idcs] = 0
        else:
            idcs = (freq > fmin) & (freq < fmax)
            dft[idcs] = 0
        return _sp_fft.irfft(dft, axis=0)

    @staticmethod
    def filter_switching_cycles(orb, freq_sampling, freq_switching):
        """
        Filter out the switching frequency from the TbT data.

        Parameters:
            orb (numpy.ndarray): Input signal of shape (Nsamples, Nbpms).
            freq_sampling (float): Sampling frequency of the input signal.
            freq_switching (float): Switching frequency to be filtered out.

        Returns:
            numpy.ndarray: Signal with the switching frequency removed, same
            shape as the input.

        """
        # Calculate the number of samples per switching cycle
        sw_sample_size = round(freq_sampling/freq_switching)
        osiz = orb.shape[0]
        nr_sws = osiz // sw_sample_size
        siz = nr_sws * sw_sample_size

        # Divide data into 3D array with switching cycles
        orb_reshape = orb[:siz].T.reshape(orb.shape[1], -1, sw_sample_size)

        # Average to get the switching signature
        sw_sig = orb_reshape.mean(axis=1)

        # Replicate the switching signature to match the size of original data
        sw_pert = _np.tile(sw_sig, (1, nr_sws))
        if osiz > siz:
            sw_pert = _np.hstack([sw_pert, sw_sig[:, :osiz-siz]])
        # Subtract the replicated switching signature from the original data
        return orb - sw_pert.T

    @staticmethod
    def calc_spectrum(data, fs=1):
        """."""
        spec = _sp_fft.rfft(data, axis=0)/data.shape[0]
        freq = _sp_fft.rfftfreq(data.shape[0], d=1/fs)
        return spec, freq

    @staticmethod
    def calc_svd(data):
        """."""
        return _np.linalg.svd(data, full_matrices=False)

    def _bpm_tag(self, idx):
        names = self.devices['fambpms'].bpm_names
        return f'{names[idx]:s} (idx={idx:d})'
