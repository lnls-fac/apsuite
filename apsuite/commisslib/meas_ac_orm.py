"""Main module."""
import time as _time
from threading import Thread as _Thread
from copy import deepcopy as _dcopy
from functools import reduce as _red
import operator as _opr

import numpy as _np
import scipy.optimize as _scyopt
import scipy.stats as _scystat

from mathphys.functions import save_pickle as _save_pickle

from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient
from siriuspy.devices import StrengthConv, PowerSupply, CurrInfoSI, \
    Trigger, Event, EVG, RFGen, Tune, FamBPMs, ASLLRF
from siriuspy.sofb.csdev import SOFBFactory
from siriuspy.search import LLTimeSearch as _LLTime
# import pyaccel as _pyaccel
from mathphys.functions import get_namedtuple as _get_namedtuple
from .. import asparams as _asparams
from ..utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass

from .meas_bpms_signals import AcqBPMsSignals as _AcqBPMsSignals
from .measure_orbit_stability import OrbitAnalysis as _OrbitAnalysis


class ACORMParams(_ParamsBaseClass):
    """."""
    RFModes = _get_namedtuple('RFModes', ('Step', 'Phase'))

    def __init__(self):
        """."""
        super().__init__()
        self.timeout_bpms = 60  # [s]
        self.ref_respmat_name = 'ref_respmat'
        self.meas_bpms_noise = True
        self.meas_rf_line = True

        self.corrs_ch2meas = 'all'
        self.corrs_cv2meas = 'all'
        self.corrs_nrruns_per_acq = 1
        self.corrs_excit_time = 4  # [s]
        self.corrs_delay = 5e-3  # [s]
        self.corrs_norm_kicks = False
        self.corrs_ch_kick = 5  # [urad]
        self.corrs_cv_kick = 5  # [urad]
        self.corrs_dorb1ch = 20  # [um]
        self.corrs_dorb1cv = 20  # [um]
        freqs = self.find_primes(16, start=3)
        self.corrs_ch_freqs = freqs[1::2][:6]
        self.corrs_cv_freqs = freqs[::2][:8]

        self.rf_excit_time = 4  # [s]
        self._rf_mode = self.RFModes.Phase
        self.rf_step_kick = 5  # [Hz]
        self.rf_step_delay = 200e-3  # [s]
        self.rf_phase_amp = 2  # [°]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += ftmp('timeout_bpms', self.timeout_bpms, '[s]')
        stg += stmp('ref_respmat_name', self.ref_respmat_name, '')
        stg += stmp('meas_bpms_noise', str(self.meas_bpms_noise), '')
        stg += stmp('meas_rf_line', str(self.meas_rf_line), '')

        stg += stmp('corrs_ch2meas', str(self.corrs_ch2meas), '')
        stg += stmp('corrs_cv2meas', str(self.corrs_cv2meas), '')
        stg += dtmp('corrs_nrruns_per_acq', self.corrs_nrruns_per_acq, '')
        stg += ftmp('corrs_excit_time', self.corrs_excit_time, '[s]')
        stg += ftmp('corrs_delay', self.corrs_delay, '[s]')
        stg += stmp('corrs_norm_kicks', str(self.corrs_norm_kicks), '')
        stg += ftmp(
            'corrs_ch_kick', self.corrs_ch_kick,
            '[urad] (only used if corrs_norm_kicks == False)')
        stg += ftmp(
            'corrs_cv_kick', self.corrs_cv_kick,
            '[urad] (only used if corrs_norm_kicks == False)')
        stg += ftmp(
            'corrs_dorb1ch', self.corrs_dorb1ch,
            '[um] (only used if corrs_norm_kicks == True)')
        stg += ftmp(
            'corrs_dorb1cv', self.corrs_dorb1cv,
            '[um] (only used if corrs_norm_kicks == True)')
        stg += stmp('corrs_ch_freqs', str(self.corrs_ch_freqs), '[Hz]')
        stg += stmp('corrs_cv_freqs', str(self.corrs_cv_freqs), '[Hz]')

        stg += ftmp('rf_excit_time', self.rf_excit_time, '[s]')
        stg += stmp('rf_mode', self.rf_mode_str, str(self.RFModes))
        stg += ftmp('rf_step_kick', self.rf_step_kick, '[Hz]')
        stg += ftmp('rf_step_delay', self.rf_step_delay, '[s]')
        stg += ftmp('rf_phase_amp', self.rf_phase_amp, '[°]')
        return stg

    @property
    def rf_mode_str(self):
        """."""
        return self.RFModes._fields[self._rf_mode]

    @property
    def rf_mode(self):
        """."""
        return self._rf_mode

    @rf_mode.setter
    def rf_mode(self, value):
        """."""
        if value in self.RFModes._fields:
            self._rf_mode = self.RFModes._fields.index(value)
        elif int(value) in self.RFModes:
            self._rf_mode = int(value)

    @staticmethod
    def find_primes(n_primes, start=3):
        """."""
        primes = []
        i = start
        while True:
            for j in range(3, int(_np.sqrt(i))+1, 2):
                if not i % j:
                    break
            else:
                if i % 2:
                    primes.append(i)
            i += 1
            if len(primes) >= n_primes:
                break
        return _np.array(primes)


# TODO: Ideas for analisys:
# - try to use naff with prior filtering of data,  and making an algorithm to
#   find the frequencies closest to the desired ones
# -
class MeasACORM(_ThreadBaseClass):
    """."""

    RF_CONDITIONING_FREQ = 1/12/4e6  # in units of RF frequency
    RF_CONDITIONING_VOLTAGEMIN = 60  # mV
    RF_CONDITIONING_DUTY = 55  # %

    def __init__(self, isonline=True):
        """."""
        self.params = ACORMParams()
        super().__init__(
            params=self.params, target=self._do_measure, isonline=isonline)
        self.bpms = None

        self.sofb_data = SOFBFactory.create('SI')
        self.configdb = _ConfigDBClient(config_type='si_orbcorr_respm')
        if self.isonline:
            self._create_devices()

    def load_and_apply(self, fname: str):
        """Load and apply `data` and `params` from pickle or HDF5 file.

        Args:
            fname (str): name of the pickle file. If extension is not provided,
                '.pickle' will be added and a pickle file will be assumed.
                If provided, must be '.pickle' for pickle files or
                {'.h5', '.hdf5', '.hdf', '.hd5'} for HDF5 files.

        """
        super().load_and_apply(fname)
        if isinstance(self.data, list):
            self.data = {'magnets': self.data}

    @staticmethod
    def fitting_matrix(tim, freqs, num_cycles=None):
        """Create the matrix used for fitting of fourier components.

        The ordering of the matrix is the following:
           mat[i, 2*j] = cos(2*pi*freqs[j]*tim[i])
           mat[i, 2*j+1] = sin(2*pi*freqs[j]*tim[i])

        Args:
            tim (numpy.ndarray): array with times
            freqs (numpy.ndarray): array with frequencies to fit.
            num_cycles (numpy.ndarray, optional): number of cycles of each
                frequency. If not provided, all data range will be considered.

        Returns:
            numpy.ndarray: fitting matrix (len(tim), 2*len(freqs))

        """
        mat = _np.zeros((tim.size, 2*freqs.size))
        arg = 2*_np.pi*freqs[None, :]*tim[:, None]
        cos = _np.cos(arg)
        sin = _np.sin(arg)

        if num_cycles is not None:
            cond = arg > 2*_np.pi*num_cycles[None, :]
            cos[cond] = 0
            sin[cond] = 0
        mat[:, ::2] = cos
        mat[:, 1::2] = sin
        return mat

    @classmethod
    def fit_fourier_components(cls, data, freqs, dtim, num_cycles=None):
        """Fit Fourier components in signal for the given frequencies.

        Args:
            data (numpy.ndarray, NxM): signal to be fitted consisting of M
                columns of data.
            freqs (numpy.ndarray, K): K frequencies to fit Fourier components.
            dtim (numpy.ndarray, N): time vector for data columns.
            num_cycles (num.ndarray, K): number of cycles of each frequency.
                If not provided, all data range will be considered.

        Returns:
            numpy.ndarray, KxM: Fourier amplitudes.
            numpy.ndarray, KxM: Fourier phases (phase==0 means pure sine).
            numpy.ndarray, KxM: Fourier cosine coefficients.
            numpy.ndarray, KxM: Fourier sine coefficients.

        """
        tim = _np.arange(data.shape[0]) * dtim

        mat = cls.fitting_matrix(tim, freqs, num_cycles)
        coeffs, *_ = _np.linalg.lstsq(mat, data, rcond=None)

        cos = coeffs[::2]
        sin = coeffs[1::2]
        amps = _np.sqrt(cos**2 + sin**2)
        phases = _np.arctan2(cos, sin)
        return amps, phases, cos, sin

    @staticmethod
    def fit_fourier_components_naff(data, freqs0, dtim):
        """Not implemented properly yet. Please don't use this."""
        freqs = _np.zeros((len(freqs0), data.shape[1]))
        fourier = _np.zeros((len(freqs0), data.shape[1]), dtype=complex)

        for i in range(data.shape[1]):
            datum = data[:, i]
            fre, four = _pyaccel.naff.naff_general(
                datum, nr_ff=len(freqs0), is_real=True)
            fre = _np.abs(fre)
            idx = _np.argsort(fre)
            fre = fre[idx]
            four = four[idx]
            freqs[:, i] = fre/dtim
            fourier[:, i] = four

        amps = _np.abs(fourier)
        phases = _np.angle(fourier)
        cos = fourier.real
        sin = fourier.imag

        return amps, phases, cos, sin

    @staticmethod
    def calc_correlation(arr1, arr2):
        """Return the linear correlation between respective columns.

        Args:
            arr1 (numpy.ndarray, NxM): first array
            arr2 (numpy.ndarray, NxM): second array

        Returns:
            numpy.ndarray, M: correlation of the ith column of arr1 with ith
                column of arr2.

        """
        corr = (arr1 * arr2).sum(axis=0)
        corr /= _np.linalg.norm(arr1, axis=0)
        corr /= _np.linalg.norm(arr2, axis=0)
        return _np.abs(corr)

    def save_loco_input_data(self, respmat_name: str, overwrite=False):
        """Save in `.pickle` format a dictionary with all LOCO input data.

        Args:
            respmat_name (str): Name of the response matrix. A matrix with
                this name will be saved in configdb.
            overwrite (bool, optional): whether to overwrite existing files.
                Defaults to False.

        """
        bpms_data = self.data['bpms_noise']
        bpms_anly = self.analysis['bpms_noise']
        data = dict()
        data['timestamp'] = bpms_data['timestamp']
        data['rf_frequency'] = bpms_data['rf_frequency']
        data['tunex'] = bpms_data['tunex']
        data['tuney'] = bpms_data['tuney']
        data['stored_current'] = bpms_data['stored_current']
        data['bpm_variation'] = bpms_anly['bpm_variation']
        data['orbmat_name'] = respmat_name
        mat = self.save_respmat_to_configdb(respmat_name)
        data['respmat'] = mat

        _save_pickle(data, f'loco_input_{respmat_name:s}', overwrite=overwrite)

    def save_respmat_to_configdb(self, name: str):
        """Save response matrix to ConfigDb Server.

        Args:
            name (str): name of the matrix to saved.

        Returns:
            numpy.ndarray, 320x281: response matrix. Missing data is filled
                with zeros.

        """
        mat = self.build_respmat()
        self.configdb.insert_config(name, mat)
        return mat

    def build_respmat(self):
        """Build response matrix from previously analysed data.

        Returns:
            numpy.ndarray, 320x281: response matrix. Missing data is filled
                with zeros.

        """
        sofb = self.sofb_data
        mat = _np.zeros((2*sofb.nr_bpms, sofb.nr_corrs), dtype=float)
        anls = self.analysis['magnets']
        for anly in anls:
            mat[:sofb.nr_bpms, anly['mat_idcs']] = anly['mat_colsx']
            mat[sofb.nr_bpms:, anly['mat_idcs']] = anly['mat_colsy']
        if 'rf' in self.analysis:
            anl = self.analysis['rf']
            mat[:sofb.nr_bpms, -1] = anl['mat_colx']
            mat[sofb.nr_bpms:, -1] = anl['mat_coly']
        return mat

    def process_data(
            self, mag_idx_ini=None, mag_min_freq=None, mag_max_freq=None,
            rf_step_trans_len=10, rf_phase_central_freq=None,
            rf_phase_window=10):
        """Process measured data.

        Args:
            mag_idx_ini ([type], optional): initial index of orbit waveform
                where magnets excitation start. Defaults to None.
            mag_min_freq ([type], optional): Frequencies below this value will
                be filtered out before fitting of magnets excitation.`None`
                means no high pass filter will be applied. Defaults to `None`.
            mag_max_freq ([type], optional): Frequencies above this value will
                be filtered out before fitting of magnets excitation. `None`
                means no low pass filter will be applied. Defaults to `None`.
            rf_transition_length (int, optional): Number of indices to ignore
                right before or after RF frequency changes in RF line
                measurements. Defaults to 10.

        """
        self.analysis['magnets'] = self._process_magnets(
            self.data['magnets'], mag_idx_ini, mag_min_freq, mag_max_freq)

        rf_d = self.data.get('rf')
        if rf_d is not None:
            if 'mode' in rf_d and rf_d['mode'] == self.params.RFModes.Phase:
                anly = self._process_rf_phase(
                    rf_d, rf_phase_window, central_freq=rf_phase_central_freq)
            else:
                anly = self._process_rf_step(rf_d, rf_step_trans_len)
            self.analysis['rf'] = anly
        bpms_data = self.data.get('bpms_noise')
        if bpms_data is not None:
            self.analysis['bpms_noise'] = self._process_bpms_noise(bpms_data)

    def get_magnets_data(self, chs_used, cvs_used):
        """Get magnet related data.

        Args:
            chs_used (list, tuple): CHs used in excitation.
            cvs_used (list, tuple): CVs used in excitation.

        Returns:
            dict: `chs_used` and `cvs_used` related data.

        """
        data = {
            'ch_names': [], 'ch_amplitudes': [], 'ch_offsets': [],
            'ch_kick_amplitudes': [], 'ch_kick_offsets': [],
            'ch_frequency': [], 'ch_num_cycles': [], 'ch_cycle_type': [],
            'cv_names': [], 'cv_amplitudes': [], 'cv_offsets': [],
            'cv_kick_amplitudes': [], 'cv_kick_offsets': [],
            'cv_frequency': [], 'cv_num_cycles': [], 'cv_cycle_type': [],
            }
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

        trig = self.devices['trigcorrs']
        data['corrs_trig_delay_raw'] = trig.delay_raw
        data['corrs_trig_delta_delay_raw'] = trig.delta_delay_raw
        return data

    def get_bpms_data(self):
        """Get all BPM related data relevant for the measurements.

        Returns:
            dict: BPMs data.

        """
        orbx, orby = self.bpms.get_mturn_signals()
        bpm0 = self.bpms.devices[0]
        rf_freq = self.devices['rfgen'].frequency

        data = dict()
        data['orbx'] = orbx
        data['orby'] = orby
        data['rf_frequency'] = rf_freq
        data['acq_rate'] = bpm0.acq_channel_str
        data['sampling_frequency'] = self.bpms.get_sampling_frequency(
            rf_freq)
        data['nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['nrsamples_post'] = bpm0.acq_nrsamples_post
        data['trig_delay_raw'] = self.devices['trigbpms'].delay_raw
        data['switching_mode'] = bpm0.switching_mode_str
        data['switching_frequency'] = self.bpms.get_switching_frequency(
            rf_freq)
        return data

    def get_general_data(self):
        """Get general purpose data.

        Returns:
            dict: general purpose data.

        """
        data = dict()
        data['timestamp'] = _time.time()
        data['stored_current'] = self.devices['currinfo'].current
        data['tunex'] = self.devices['tune'].tunex
        data['tuney'] = self.devices['tune'].tuney
        return data

    # ------------------ Auxiliary Methods ------------------

    def _create_devices(self):
        # Create objects to convert kicks to current
        t00 = _time.time()
        print('Creating kick converters  -> ', end='')
        self.devices.update({
            n+':StrengthConv': StrengthConv(n, 'Ref-Mon')
            for n in self.sofb_data.ch_names})
        self.devices.update({
            n+':StrengthConv': StrengthConv(n, 'Ref-Mon')
            for n in self.sofb_data.cv_names})
        print(f'ET: = {_time.time()-t00:.2f}s')

        # Create objects to interact with correctors
        t00 = _time.time()
        print('Creating correctors       -> ', end='')
        self.devices.update({
            nme: PowerSupply(nme) for nme in self.sofb_data.ch_names})
        self.devices.update({
            nme: PowerSupply(nme) for nme in self.sofb_data.cv_names})
        print(f'ET: = {_time.time()-t00:.2f}s')

        # Create object to get stored current
        t00 = _time.time()
        print('Creating General Devices  -> ', end='')
        self.devices['currinfo'] = CurrInfoSI()
        # Create RF generator object
        self.devices['rfgen'] = RFGen(
            props2init=('GeneralFreq-SP', 'GeneralFreq-RB'))
        self.devices['llrf'] = ASLLRF(ASLLRF.DEVICES.SI)
        # Create Tune object:
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        print(f'ET: = {_time.time()-t00:.2f}s')

        # Create BPMs trigger:
        t00 = _time.time()
        print('Creating Timing           -> ', end='')
        self.devices['trigbpms'] = Trigger('SI-Fam:TI-BPM')
        # Create Correctors Trigger:
        self.devices['trigcorrs'] = Trigger('SI-Glob:TI-Mags-Corrs')
        # Create event to start data acquisition sinchronously:
        self.devices['evt_study'] = Event('Study')
        self.devices['evg'] = EVG()
        print(f'ET: = {_time.time()-t00:.2f}s')

        # Create BPMs
        t00 = _time.time()
        print('Creating BPMs             -> ', end='')
        self.bpms = FamBPMs(mturn_signals2acq='XY', props2init='acq')
        self.devices['fambpms'] = self.bpms
        print(f'ET: = {_time.time()-t00:.2f}s')

    # ---------------- Measurement Methods ----------------

    def _do_measure(self):
        tim_state = self._get_timing_state()
        if self.params.meas_bpms_noise:
            self.data['bpms_noise'] = self._do_measure_bpms_noise()

        if self._stopevt.is_set():
            print('Stopping...')
            self._set_timing_state(tim_state)
            return

        if self.params.meas_rf_line:
            self.data['rf'] = self._do_measure_rf_line()

        if self._stopevt.is_set():
            print('Stopping...')
            self._set_timing_state(tim_state)
            return

        self.data['magnets'] = self._do_measure_magnets()
        self._set_timing_state(tim_state)

    def _do_measure_bpms_noise(self):
        elt = _time.time()
        par = self.params

        print('Measuring BPMs Noise:')

        t00 = _time.time()
        print('    Configuring BPMs...', end='')
        rf_freq = self.devices['rfgen'].frequency
        nr_points = par.corrs_excit_time + par.corrs_delay*2
        nr_points *= self.bpms.get_sampling_frequency(rf_freq, acq_rate='FAcq')
        nr_points = int(_np.ceil(nr_points))
        ret = self._config_bpms(nr_points, rate='FAcq')
        if ret < 0:
            print(f'BPM {-ret-1:d} did not finish last acquisition.')
        elif ret > 0:
            print(f'BPM {ret-1:d} is not ready for acquisition.')
        self._config_timing()
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        t00 = _time.time()
        print('    Sending Trigger signal...', end='')
        self.bpms.mturn_reset_flags_and_update_initial_timestamps()
        self.devices['evt_study'].cmd_external_trigger()
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        # Wait BPMs PV to update with new data
        t00 = _time.time()
        print('    Waiting BPMs to update...', end='')
        ret = self.bpms.mturn_wait_update(timeout=par.timeout_bpms)
        if ret:
            print(
                'Problem: timed out waiting BPMs update. '
                f'Error code: {ret:d}')
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        # get data
        data = self.get_general_data()
        data.update(self.get_bpms_data())
        data['ch_freqs'] = par.corrs_ch_freqs
        data['cv_freqs'] = par.corrs_cv_freqs

        elt -= _time.time()
        elt *= -1
        print(f'    Elapsed Time: {elt:.2f}s')
        return data

    def _do_measure_rf_line(self):
        elt = _time.time()
        par = self.params
        print('Measuring RF Line:')

        rate = 'FOFB' if par.rf_mode == par.RFModes.Phase else 'FAcq'

        t00 = _time.time()
        print('    Configuring BPMs...', end='')
        rf_freq = self.devices['rfgen'].frequency
        nr_points = par.rf_excit_time + par.rf_step_delay*2
        nr_points *= self.bpms.get_sampling_frequency(rf_freq, acq_rate=rate)
        nr_points = int(_np.ceil(nr_points))
        ret = self._config_bpms(nr_points, rate=rate)
        if ret < 0:
            print(f'BPM {-ret-1:d} did not finish last acquisition.')
        elif ret > 0:
            print(f'BPM {ret-1:d} is not ready for acquisition.')
        self._config_timing()
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        if par.rf_mode == par.RFModes.Phase:
            t00 = _time.time()
            print('    Turning RF Conditioning On...')
            self._turn_on_llrf_conditioning()
            print(f'Done! ET: {_time.time()-t00:.2f}s')

        t00 = _time.time()
        print('    Sending Trigger signal...', end='')
        self.bpms.mturn_reset_flags_and_update_initial_timestamps()
        self.devices['evt_study'].cmd_external_trigger()

        if par.rf_mode == par.RFModes.Step:
            t00 = _time.time()
            print('    Sweep RF...', end='')
            thr = _Thread(target=self._sweep_rf, daemon=True)
            _time.sleep(par.rf_step_delay)
            thr.start()
            print(f'Done! ET: {_time.time()-t00:.2f}s')

        # Wait BPMs PV to update with new data
        t00 = _time.time()
        print('    Waiting BPMs to update...', end='')
        ret = self.bpms.mturn_wait_update(timeout=par.timeout_bpms)
        if ret:
            print(
                'Problem: timed out waiting BPMs update. '
                f'Error code: {ret:d}')
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        llrf = self.devices['llrf']
        data['llrf_cond_voltage_min'] = llrf.voltage_refmin_sp
        data['llrf_cond_phase_min'] = llrf.phase_refmin_sp
        data['llrf_cond_voltage'] = llrf.voltage_sp
        data['llrf_cond_phase'] = llrf.phase_sp
        data['llrf_cond_duty_cycle'] = llrf.conditioning_duty_cycle
        data['llrf_cond_state'] = llrf.conditioning_state

        if par.rf_mode == par.RFModes.Phase:
            t00 = _time.time()
            print('    Turning RF Conditioning Off...')
            self._turn_off_llrf_conditioning()
            print(f'Done! ET: {_time.time()-t00:.2f}s')

        # get data
        data = self.get_general_data()
        data.update(self.get_bpms_data())
        data['mode'] = par.rf_mode
        data['step_kick'] = par.rf_step_kick
        data['excit_time'] = par.rf_excit_time
        data['step_delay'] = par.rf_step_delay
        data['mom_compac'] = _asparams.SI_MOM_COMPACT
        data['llrf_cond_freq'] = data['rf_frequency']*self.RF_CONDITIONING_FREQ

        elt -= _time.time()
        elt *= -1
        print(f'    Elapsed Time: {elt:.2f}s')
        return data

    def _sweep_rf(self):
        kick = self.params.rf_step_kick
        excit_time = self.params.rf_excit_time
        rfgen = self.devices['rfgen']
        freq0 = rfgen.frequency
        rfgen.frequency = freq0 - kick
        _time.sleep(excit_time/2)
        rfgen.frequency = freq0 + kick
        _time.sleep(excit_time/2)
        rfgen.frequency = freq0

    def _turn_on_llrf_conditioning(self):
        llrf = self.devices['llrf']
        llrf.voltage_refmin = llrf.voltage_sp
        llrf.phase_refmin = llrf.phase_sp - self.params.rf_phase_amp
        llrf.set_duty_cycle(self.RF_CONDITIONING_DUTY)
        llrf.cmd_turn_on_conditioning()

    def _turn_off_llrf_conditioning(self):
        llrf = self.devices['llrf']
        llrf.cmd_turn_off_conditioning()
        llrf.voltage_refmin = self.RF_CONDITIONING_VOLTAGEMIN
        llrf.phase_refmin = llrf.phase_sp

    def _do_measure_magnets(self):
        elt0 = _time.time()
        data_mags = []

        print('Measuring Magnets:')

        # Shift correctors so first corrector is 01M1
        ch_names = self.sofb_data.ch_names
        cv_names = self.sofb_data.cv_names
        chs_shifted = self._shift_list(ch_names, 1)
        cvs_shifted = self._shift_list(cv_names, 1)

        # set operation mode to slowref
        if not self._change_corrs_opmode('slowref'):
            print('Problem: Correctors not in SlowRef mode.')
            return data_mags

        excit_time = self.params.corrs_excit_time
        freqh = self.params.corrs_ch_freqs
        freqv = self.params.corrs_cv_freqs

        rf_freq = self.devices['rfgen'].frequency
        nr_points = excit_time + self.params.corrs_delay*2
        nr_points *= self.bpms.get_sampling_frequency(rf_freq, acq_rate='FAcq')
        nr_points = int(_np.ceil(nr_points))

        ch2meas = self.params.corrs_ch2meas
        cv2meas = self.params.corrs_cv2meas
        if isinstance(ch2meas, str) and ch2meas.startswith('all'):
            ch2meas = chs_shifted
        if isinstance(cv2meas, str) and ch2meas.startswith('all'):
            cv2meas = cvs_shifted

        ch_kicks = _np.full(len(ch2meas), self.params.corrs_ch_kick)
        cv_kicks = _np.full(len(ch2meas), self.params.corrs_cv_kick)
        if self.params.corrs_norm_kicks:
            orm = _np.array(self.configdb.get_config_value(
                self.params.ref_respmat_name))

            ch_idx = _np.array([ch_names.index(n) for n in ch2meas])
            cv_idx = _np.array([cv_names.index(n) for n in cv2meas])
            cv_idx += len(chs_shifted)
            ch_kicks = self.params.corrs_dorb1ch/orm[:160, ch_idx].std(axis=0)
            cv_kicks = self.params.corrs_dorb1cv/orm[160:, cv_idx].std(axis=0)

        nr1acq = self.params.corrs_nrruns_per_acq
        nfh = freqh.size
        nfv = freqv.size
        nrh = len(ch2meas) // nfh
        nrv = len(cv2meas) // nfv
        nrh += bool(len(ch2meas) % nfh)
        nrv += bool(len(cv2meas) % nfv)
        nruns = max(nrh, nrv)
        nacqs = nruns // nr1acq
        nacqs += bool(nruns % nr1acq)
        for itr in range(nacqs):
            elt = _time.time()

            chs_slc, cvs_slc = [], []
            ch_kick, cv_kick = [], []
            freqhe, freqve = [], []
            off = itr*nr1acq
            for run in range(nr1acq):
                slch = slice((off + run-1)*nfh, (off + run)*nfh)
                slcv = slice((off + run-1)*nfv, (off + run)*nfv)
                chs = ch2meas[slch]
                cvs = cv2meas[slcv]
                if chs or cvs:
                    ch_kick.extend(ch_kicks[slch])
                    cv_kick.extend(cv_kicks[slcv])
                    chs_slc.append(chs)
                    cvs_slc.append(cvs)
                    freqhe.extend(freqh)
                    freqve.extend(freqv)

            t00 = _time.time()
            print('    Configuring BPMs and timing...', end='')
            ret = self._config_bpms(nr_points * len(chs_slc), rate='FAcq')
            if ret < 0:
                print(f'BPM {-ret-1:d} did not finish last acquisition.')
            elif ret > 0:
                print(f'BPM {ret-1:d} is not ready for acquisition.')
            self._config_timing(
                self.params.corrs_delay, chs_slc, cvs_slc, nr_points=nr_points)
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # configure correctors
            t00 = _time.time()
            chs_f = _red(_opr.add, chs_slc)
            cvs_f = _red(_opr.add, cvs_slc)
            print('    Configuring Correctors...', end='')
            self._config_correctors(chs_f, ch_kick, freqhe, excit_time)
            self._config_correctors(cvs_f, cv_kick, freqve, excit_time)
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # set operation mode to cycle
            t00 = _time.time()
            print('    Changing Correctors to Cycle...', end='')
            if not self._change_corrs_opmode('cycle', chs_f + cvs_f):
                print('Problem: Correctors not in Cycle mode.')
                break
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # send event through timing system to start acquisitions
            t00 = _time.time()
            print('    Sending Timing signal...', end='')
            self.bpms.mturn_reset_flags_and_update_initial_timestamps()
            self.devices['evt_study'].cmd_external_trigger()
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # Wait BPMs PV to update with new data
            t00 = _time.time()
            print('    Waiting BPMs to update...', end='')
            ret = self.bpms.mturn_wait_update(timeout=self.params.timeout_bpms)
            if ret:
                print(
                    'Problem: timed out waiting BPMs update. '
                    f'Error code: {ret:d}')
                break
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # get data for each sector separately:
            data = self.get_general_data()
            data.update(self.get_bpms_data())
            orbx = data.pop('orbx')
            orby = data.pop('orby')
            for i, (chs, cvs) in enumerate(zip(chs_slc, cvs_slc)):
                datas = _dcopy(data)
                datas['orbx'] = orbx[i*nr_points:(i+1)*nr_points]
                datas['orby'] = orby[i*nr_points:(i+1)*nr_points]
                datas.update(self.get_magnets_data(chs_used=chs, cvs_used=cvs))
                data_mags.append(datas)

            # set operation mode to slowref
            t00 = _time.time()
            print('    Changing Correctors to SlowRef...', end='')
            if not self._wait_cycle_to_finish(chs_f + cvs_f):
                print('Problem: Cycle still not finished.')
                break
            if not self._change_corrs_opmode('slowref', chs_f + cvs_f):
                print('Problem: Correctors not in SlowRef mode.')
                break
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            elt -= _time.time()
            elt *= -1
            print(f'    Elapsed Time: {elt:.2f}s')
            if self._stopevt.is_set():
                print('Stopping...')
                break

        # set operation mode to slowref
        if not self._change_corrs_opmode('slowref'):
            print('Problem: Correctors not in SlowRef mode.')
            return data_mags

        elt0 -= _time.time()
        elt0 *= -1
        print(f'Finished!!  ET: {elt0/60:.2f}min')
        return data_mags

    # ---------------- Data Processing methods ----------------

    def _process_rf_step(self, rf_data, transition_length=10):
        def _fitting_model(tim, *params):
            idx1, idx2, idx3, amp1, amp2 = params
            y = _np.zeros(tim.size)
            y[int(idx1):int(idx2)] = amp1
            y[int(idx2):int(idx3)] = amp2
            return y

        anly = dict()

        fsamp = rf_data['sampling_frequency']
        dtim = 1/fsamp
        anly['fsamp'] = fsamp
        anly['dtim'] = dtim

        orbx = rf_data['orbx'].copy()
        orby = rf_data['orby'].copy()
        orbx -= orbx.mean(axis=0)
        orby -= orby.mean(axis=0)

        tim = _np.arange(orbx.shape[0]) * dtim
        anly['time'] = tim

        # fit average horizontal orbit to find transition indices
        excit_time = rf_data['excit_time']
        dly = rf_data['step_delay']
        kick = rf_data['step_kick']
        idx1 = (tim > dly).nonzero()[0][0]
        idx2 = (tim > (excit_time/2 + dly)).nonzero()[0][0]
        idx3 = (tim > (excit_time + dly)).nonzero()[0][0]
        etax_avg = 0.033e6  # average dispersion function, in [um]
        rf_freq = rf_data['rf_frequency']
        mom_compac = rf_data.get('mom_compac', _asparams.SI_MOM_COMPACT)

        amp1 = -kick * etax_avg / mom_compac / rf_freq
        amp2 = kick * etax_avg / mom_compac / rf_freq
        par0 = [idx1, idx2, idx3, amp1, amp2]
        par, _ = _scyopt.curve_fit(
            _fitting_model, tim, orbx.mean(axis=1), p0=par0)
        idx1, idx2, idx3, amp1, amp2 = par
        idx1 = int(idx1)
        idx2 = int(idx2)
        idx3 = int(idx3)
        par = idx1, idx2, idx3, amp1, amp2
        anly['params_fit_init'] = par0
        anly['params_fit'] = par
        anly['idx1'] = idx1
        anly['idx2'] = idx2
        anly['idx3'] = idx3
        anly['amp1'] = amp1
        anly['amp2'] = amp2
        anly['transition_length'] = transition_length

        sec1_ini = idx1 + transition_length
        sec1_fin = idx2 - transition_length
        sec2_ini = idx2 + transition_length
        sec2_fin = idx3 - transition_length
        anly['sec1_ini'] = sec1_ini
        anly['sec1_fin'] = sec1_fin
        anly['sec2_ini'] = sec2_ini
        anly['sec2_fin'] = sec2_fin

        orbx_neg = orbx[sec1_ini:sec1_fin].mean(axis=0)
        orby_neg = orby[sec1_ini:sec1_fin].mean(axis=0)
        orbx_pos = orbx[sec2_ini:sec2_fin].mean(axis=0)
        orby_pos = orby[sec2_ini:sec2_fin].mean(axis=0)
        anly['orbx_neg'] = orbx_neg
        anly['orby_neg'] = orby_neg
        anly['orbx_pos'] = orbx_pos
        anly['orby_pos'] = orby_pos
        anly['mat_colx'] = (orbx_pos - orbx_neg) / kick / 2
        anly['mat_coly'] = (orby_pos - orby_neg) / kick / 2
        return anly

    def _process_rf_phase(
            self, rf_data, window=10, central_freq=None, orm_name=None):
        anly = dict()

        fsamp = rf_data['sampling_frequency']
        fswitch = rf_data['switching_frequency']  # bug
        sw_mode = rf_data['switching_mode']
        rf_freq = rf_data['rf_frequency']
        dtim = 1/fsamp
        anly['fsamp'] = fsamp
        anly['dtim'] = dtim
        anly['fswitch'] = fswitch

        orbx = rf_data['orbx'].copy()
        orby = rf_data['orby'].copy()
        orbx -= orbx.mean(axis=0)
        orby -= orby.mean(axis=0)
        if fsamp / fswitch > 1 and sw_mode == 'switching':
            orbx = _AcqBPMsSignals.filter_switching_cycles(
                orbx, fsamp, freq_switching=fswitch)
            orby = _AcqBPMsSignals.filter_switching_cycles(
                orby, fsamp, freq_switching=fswitch)

        f_cond = rf_freq * self.RF_CONDITIONING_FREQ

        # Find peak with maximum amplitude to filter data
        if central_freq is None:
            # Possible range to find peak
            harm_min = int(1700/f_cond)
            harm_max = int(2300/f_cond)
            freqs = _np.arange(harm_min, harm_max) * f_cond
            freqx = _np.fft.rfftfreq(orbx.shape[0], d=dtim)

            # find indices of data frequencies close to excited frequencies
            idcs = _np.searchsorted(freqx, freqs)
            idsm1 = idcs - 1
            cond = _np.abs(freqx[idcs]-freqs) < _np.abs(freqx[idsm1]-freqs)
            idcs_min = _np.where(cond, idcs, idsm1)

            dftx = _np.abs(_np.fft.rfft(orbx, axis=0))[idcs_min]
            dfty = _np.abs(_np.fft.rfft(orby, axis=0))[idcs_min]
            amaxx = dftx.argmax(axis=0)
            amaxy = dfty.argmax(axis=0)
            res = _scystat.mode(_np.r_[amaxx, amaxy], keepdims=False)
            # res = _scystat.mode(amaxy, keepdims=False)
            central_freq = freqx[idcs_min][res.mode]
            print(
                f'Filtering data around {central_freq:.2f}Hz '
                f'({res.count:03d}).')

        central_freq = int(round(central_freq/f_cond)) * f_cond
        fmin = central_freq - window/2
        fmax = central_freq + window/2

        orbx = _AcqBPMsSignals.filter_data_frequencies(
            orbx, fmin=fmin, fmax=fmax, fsampling=fsamp)
        orby = _AcqBPMsSignals.filter_data_frequencies(
            orby, fmin=fmin, fmax=fmax, fsampling=fsamp)
        mom_compac = rf_data.get('mom_compac', _asparams.SI_MOM_COMPACT)

        etax, etay = self._get_reference_dispersion(
            rf_freq, mom_compac, orm_name)
        eta_meas = _OrbitAnalysis.calculate_eta_meas(orbx, orby, etax, etay)
        anly['mat_colx'] = - eta_meas[:orbx.shape[-1]] / mom_compac / rf_freq
        anly['mat_coly'] = - eta_meas[orbx.shape[-1]:] / mom_compac / rf_freq
        return anly

    def _get_reference_dispersion(self, rf_freq, mom_compac, orm_name=None):
        """."""
        orm_name = orm_name or 'ref_respmat'
        orm = _np.array(self.configdb.get_config_value(orm_name))
        etaxy = orm[:, -1]
        nrbpms = etaxy.size // 2
        etaxy *= -mom_compac * rf_freq  # units of [um]
        etax, etay = etaxy[:nrbpms], etaxy[nrbpms:]
        return etax, etay

    def _process_bpms_noise(self, bpms_data):
        sofb = self.sofb_data
        anly = dict()

        ch_freqs = bpms_data['ch_freqs']
        cv_freqs = bpms_data['cv_freqs']

        fsamp = bpms_data['sampling_frequency']
        dtim = 1/fsamp
        freqs0 = _np.r_[ch_freqs, cv_freqs]
        anly['fsamp'] = fsamp
        anly['dtim'] = dtim
        anly['freqs0'] = freqs0

        orbx = bpms_data['orbx'].copy()
        orby = bpms_data['orby'].copy()
        orbx -= orbx.mean(axis=0)
        orby -= orby.mean(axis=0)

        tim = _np.arange(orbx.shape[0]) * dtim
        anly['time'] = tim

        ampx, *_ = self.fit_fourier_components(orbx, freqs0, dtim)
        ampy, *_ = self.fit_fourier_components(orby, freqs0, dtim)
        anly['noisex'] = ampx
        anly['noisey'] = ampy

        sch = len(ch_freqs)
        scv = len(cv_freqs)

        ampx_ch = ampx[:sch].T
        ampy_ch = ampy[:sch].T
        ampx_cv = ampx[sch:].T
        ampy_cv = ampy[sch:].T

        anly['noisex_ch'] = ampx_ch
        anly['noisey_ch'] = ampy_ch
        anly['noisex_cv'] = ampx_cv
        anly['noisey_cv'] = ampy_cv

        match = _np.zeros((2*sofb.nr_bpms, sch))
        matcv = _np.zeros((2*sofb.nr_bpms, scv))
        match[:sofb.nr_bpms] = ampx_ch
        match[sofb.nr_bpms:] = ampy_ch
        matcv[:sofb.nr_bpms] = ampx_cv
        matcv[sofb.nr_bpms:] = ampy_cv

        match = _np.roll(_np.tile(match, 20), -1)
        matcv = _np.roll(_np.tile(matcv, 20), -1)
        mat = _np.zeros((2*sofb.nr_bpms, sofb.nr_corrs))
        mat[:, :sofb.nr_ch] = match
        mat[:, sofb.nr_ch:sofb.nr_chcv] = matcv

        ampx_rf = orbx.std(axis=0)
        ampy_rf = orby.std(axis=0)
        anly['noisex_rf'] = ampx_rf
        anly['noisey_rf'] = ampy_rf
        mat[:sofb.nr_bpms, -1] = ampx_rf
        mat[sofb.nr_bpms:, -1] = ampy_rf

        anly['bpm_variation'] = mat
        return anly

    def _process_magnets(
            self, magnets_data, idx_ini=None, min_freq=None, max_freq=None):
        """."""
        sofb = self.sofb_data
        corr2idx = {name: i for i, name in enumerate(sofb.ch_names)}
        corr2idx.update({name: 120+i for i, name in enumerate(sofb.cv_names)})

        analysis = []
        for data in magnets_data:
            anly = dict()
            fsamp = data['sampling_frequency']
            dtim = 1/fsamp
            ch_freqs = _np.array(data['ch_frequency'])
            cv_freqs = _np.array(data['cv_frequency'])
            freqs0 = _np.r_[ch_freqs, cv_freqs]
            anly['fsamp'] = fsamp
            anly['dtim'] = dtim
            anly['freqs0'] = freqs0

            ch_ncycles = _np.array(data['ch_num_cycles'])
            cv_ncycles = _np.array(data['cv_num_cycles'])
            num_cycles = _np.r_[ch_ncycles, cv_ncycles]
            anly['num_cycles'] = num_cycles

            ch_amps = _np.array(data['ch_kick_amplitudes'])
            cv_amps = _np.array(data['cv_kick_amplitudes'])
            kicks = _np.r_[ch_amps, cv_amps]
            anly['kicks'] = kicks

            ch_idcs = _np.array([corr2idx[name] for name in data['ch_names']])
            cv_idcs = _np.array([corr2idx[name] for name in data['cv_names']])
            idcs = _np.r_[ch_idcs, cv_idcs]
            anly['mat_idcs'] = idcs

            orbx = data['orbx'].copy()
            orby = data['orby'].copy()
            orbx -= orbx.mean(axis=0)
            orby -= orby.mean(axis=0)

            tim = _np.arange(orbx.shape[0]) * dtim
            if idx_ini is None:
                # TODO: This logic is broken when more than one sector is
                # excited per BPM acquisition, because in this case the
                # deltaDelay variable is set, while delayRaw is kept as zero.
                delay = 4/data['rf_frequency']
                delay *= data['corrs_trig_delay_raw']
                idx_ini = (tim >= delay).nonzero()[0][0]
            anly['time'] = tim
            anly['idx_ini'] = idx_ini

            orbx = orbx[idx_ini:]
            orby = orby[idx_ini:]

            if min_freq is not None and max_freq is not None:
                min_freq = min_freq or 0
                max_freq = max_freq or fsamp/2
                dftx = _np.fft.rfft(orbx, axis=0)
                dfty = _np.fft.rfft(orby, axis=0)
                freq = _np.fft.rfftfreq(orbx.shape[0], d=dtim)
                idcs = (freq < min_freq) | (freq > max_freq)
                dftx[idcs] = 0
                orbx = _np.fft.irfft(dftx, axis=0)
                orby = _np.fft.irfft(dfty, axis=0)
            anly['min_freq'] = min_freq
            anly['max_freq'] = max_freq

            ampx, phasex, cosx, sinx = self.fit_fourier_components(
                orbx, freqs0, dtim, num_cycles)
            ampy, phasey, cosy, siny = self.fit_fourier_components(
                orby, freqs0, dtim, num_cycles)

            signx = _np.ones(ampx.shape)
            signx[_np.abs(phasex) > (_np.pi/2)] = -1
            signy = _np.ones(ampy.shape)
            signy[_np.abs(phasey) > (_np.pi/2)] = -1

            anly['ampx'] = ampx
            anly['ampy'] = ampy
            anly['phasex'] = phasex
            anly['phasey'] = phasey
            anly['cosx'] = cosx
            anly['cosy'] = cosy
            anly['sinx'] = sinx
            anly['siny'] = siny
            anly['signx'] = signx
            anly['signy'] = signy
            anly['mat_colsx'] = (signx * ampx / kicks[:, None]).T
            anly['mat_colsy'] = (signy * ampy / kicks[:, None]).T
            analysis.append(anly)
        return analysis

    # ----------------- BPMs related methods -----------------------

    def _config_bpms(self, nr_points, rate='FAcq'):
        return self.bpms.mturn_config_acquisition(
            acq_rate=rate, nr_points_before=0, nr_points_after=nr_points,
            repeat=False, external=True)

    # ----------------- Timing related methods -----------------------

    def _get_timing_state(self):
        trigbpm = self.devices['trigbpms']
        trigcorr = self.devices['trigcorrs']
        evt_study = self.devices['evt_study']
        return {
            'trigbpm_source': trigbpm.source,
            'trigbpm_nr_pulses': trigbpm.nr_pulses,
            'trigbpm_delay': trigbpm.delay,
            'trigcorr_source': trigcorr.source,
            'trigcorr_nr_pulses': trigcorr.nr_pulses,
            'trigcorr_delay_raw': trigcorr.delay_raw,
            'trigcorr_delta_delay_raw': trigcorr.delta_delay_raw,
            'evt_study_mode': evt_study.mode,
            'evt_study_delay': evt_study.delay,
            }

    def _set_timing_state(self, state):
        trigbpm = self.devices['trigbpms']
        trigcorr = self.devices['trigcorrs']
        evt_study = self.devices['evt_study']
        evg = self.devices['evg']

        if 'trigbpm_source' in state:
            trigbpm.source = state['trigbpm_source']
        if 'trigbpm_nr_pulses' in state:
            trigbpm.nr_pulses = state['trigbpm_nr_pulses']
        if 'trigbpm_delay' in state:
            trigbpm.delay = state['trigbpm_delay']
        if 'trigcorr_source' in state:
            trigcorr.source = state['trigcorr_source']
        if 'trigcorr_nr_pulses' in state:
            trigcorr.nr_pulses = state['trigcorr_nr_pulses']
        if 'trigcorr_delay_raw' in state:
            trigcorr.delay_raw = state['trigcorr_delay_raw']
        if 'trigcorr_delta_delay_raw' in state:
            trigcorr.delta_delay_raw = state['trigcorr_delta_delay_raw']
        if 'evt_study_mode' in state:
            evt_study.mode = state['evt_study_mode']
        if 'evt_study_delay' in state:
            evt_study.delay = state['evt_study_delay']
        _time.sleep(0.1)
        evg.cmd_update_events()

    def _config_timing(self, cm_dly=None, chs=None, cvs=None, nr_points=None):
        """Configure timing.

        Args:
            cm_dly (float, optional): General Delay of correctors;
            chs (list, optional): List of lists of CH names. Each list
                represent a different run in the same BPM acquisition.
                Defaults to None.
            cvs (list, optional): List of lists of CV names. Each list
                represent a different run in the same BPM acquisition.
                Defaults to None.
            nr_points (int, optional): number of points of each run.
                Defaults to None.

        Raises:
            ValueError: Impossible trigger configuration.
            ValueError: Invalid trigger name.

        """
        state = dict()
        state['trigbpm_source'] = 'Study'
        state['trigbpm_nr_pulses'] = 1
        state['trigbpm_delay'] = 0.0

        state['evt_study_mode'] = 'External'
        state['evt_study_delay'] = 0

        state['trigcorr_source'] = 'Study'
        state['trigcorr_nr_pulses'] = 1

        rf_freq = self.devices['rfgen'].frequency
        ftim = rf_freq / 4  # timing base frequency
        dly = int(cm_dly * ftim)
        if chs is None or cvs is None or nr_points is None:
            state['trigcorr_delay_raw'] = dly
            self._set_timing_state(state)
            return

        state['trigcorr_delay_raw'] = 0
        nr_runs = len(chs)
        # Calculate delta_delay for correctors to be as close as possible to a
        # multiple of the the sampling period to ensure repeatability of
        # experiment along runs excited during single acquisition:
        fsamp = self.bpms.get_sampling_frequency(rf_freq, 'FAcq')
        runs_delta_dly = _np.arange(len(nr_runs), dtype=float)
        runs_delta_dly *= nr_points / fsamp
        runs_delta_dlyr = _np.round(runs_delta_dly * ftim)

        # get low level trigger names to be configured in each run of the
        # acquisition:
        ll_trigs = []
        for ch, cv in zip(chs, cvs):
            ll_trigs.append(
                {_LLTime.get_trigger_name(c+':BCKPLN') for c in ch+cv})

        # check if correctors controlled by the same trigger are requested to
        # be triggered in different times during the same acquisition
        if len(_red(_opr.or_, ll_trigs)) != _red(_opr.add, map(len, ll_trigs)):
            raise ValueError('Impossible trigger configuration requested.')

        trigcorr = self.devices['trigcorrs']
        delta_delay_raw = _np.zeros(trigcorr.delta_delay_raw.size)
        low_level = trigcorr.low_level_triggers
        for llts, ddlyr in zip(ll_trigs, runs_delta_dlyr):
            # Find all low level triggers of this sector and set their delay:
            for llt in llts:
                if llt not in low_level:
                    raise ValueError(f'Trigger {llt:s} is not valid.')
                delta_delay_raw[low_level.index(llt)] = ddlyr + dly
        state['trigcorr_delta_delay_raw'] = delta_delay_raw
        self._set_timing_state(state)

    # ----------------- Correctors related methods -----------------------

    def _config_correctors(self, corr_names, kicks, freqs, excit_time):
        """."""
        for i, cmn in enumerate(corr_names):
            cmo = self.devices[cmn]
            conv = self.devices[cmn+':StrengthConv'].conv_strength_2_current
            cmo.cycle_type = cmo.CYCLETYPE.Sine
            cmo.cycle_freq = freqs[i]
            cmo.cycle_ampl = conv(kicks[i])
            cmo.cycle_offset = cmo.currentref_mon
            cmo.cycle_theta_begin = 0
            cmo.cycle_theta_end = 0
            cmo.cycle_num_cycles = int(excit_time * freqs[i])

    def _change_corrs_opmode(self, mode, corr_names=None, timeout=10):
        """."""
        opm_sel = PowerSupply.OPMODE_SEL
        opm_sts = PowerSupply.OPMODE_STS
        mode_sel = opm_sel.Cycle if mode == 'cycle' else opm_sel.SlowRef
        mode_sts = opm_sts.Cycle if mode == 'cycle' else opm_sts.SlowRef

        if corr_names is None:
            corr_names = self.sofb_data.ch_names + self.sofb_data.cv_names

        for cmn in corr_names:
            cmo = self.devices[cmn]
            cmo.opmode = mode_sel

        interval = 0.2
        for _ in range(int(timeout/interval)):
            okk = True
            corrname = ''
            for cmn in corr_names:
                cmo = self.devices[cmn]
                oki = cmo.opmode == mode_sts
                if not oki:
                    corrname = cmn
                okk &= oki
            if okk:
                return True
            _time.sleep(interval)

        print(corrname)
        return False

    def _wait_cycle_to_finish(self, corr_names=None, timeout=10):
        if corr_names is None:
            corr_names = self.sofb_data.ch_names + self.sofb_data.cv_names

        for cmn in corr_names:
            cmo = self.devices[cmn]
            if not cmo.wait_cycle_to_finish(timeout=timeout):
                return False
        return True

    @staticmethod
    def _shift_list(lst, num):
        """."""
        return lst[-num:] + lst[:-num]
