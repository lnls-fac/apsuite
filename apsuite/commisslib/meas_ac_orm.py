"""Main module."""
import time as _time
from threading import Thread as _Thread
from copy import deepcopy as _dcopy

import numpy as _np
import scipy.optimize as _scyopt

from mathphys.functions import save_pickle as _save_pickle

from siriuspy.namesys import SiriusPVName as _PVName
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient
from siriuspy.devices import StrengthConv, PowerSupply, CurrInfoSI, \
    Trigger, Event, EVG, RFGen, Tune, FamBPMs
from siriuspy.sofb.csdev import SOFBFactory
import pyaccel as _pyaccel

from ..utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class ACORMParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.nr_points = 5500
        self.acq_rate = 'Monit1'
        self.timeout_bpms = 10  # [s]
        self.ch_kick = 5  # [urad]
        self.cv_kick = 5  # [urad]
        self.rf_kick = 5  # [Hz]
        self.delay_corrs = 50e-3  # [s]
        self.delay_rf = 200e-3  # [s]
        self.exc_duration = 5  # [s]
        self.exc_rf = 4  # [s]

        freqs = self.find_primes(16, start=120)
        self.ch_freqs = freqs[1::2][:6]
        self.cv_freqs = freqs[::2][:8]
        self.nr_sectors_per_acq = 1
        self.sectors_to_measure = _np.arange(1, 21)
        self.meas_bpms_noise = True
        self.meas_rf_line = True

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += dtmp('nr_points', self.nr_points, '')
        stg += stmp('acq_rate', self.acq_rate, '')
        stg += ftmp('timeout_bpms', self.timeout_bpms, '[s]')
        stg += ftmp('ch_kick', self.ch_kick, '[urad]')
        stg += ftmp('cv_kick', self.cv_kick, '[urad]')
        stg += ftmp('rf_kick', self.rf_kick, '[Hz]')
        stg += ftmp('delay_corrs', self.delay_corrs, '[s]')
        stg += ftmp('delay_rf', self.delay_rf, '[s]')
        stg += ftmp('exc_duration', self.exc_duration, '[s]')
        stg += ftmp('exc_rf', self.exc_rf, '[s]')
        stg += stmp('ch_freqs', str(self.ch_freqs), '[Hz]')
        stg += stmp('cv_freqs', str(self.cv_freqs), '[Hz]')
        stg += dtmp('nr_sectors_per_acq', self.nr_sectors_per_acq, '')
        stg += stmp('sectors_to_measure', str(self.sectors_to_measure), '')
        stg += stmp('meas_bpms_noise', str(self.meas_bpms_noise), '')
        stg += stmp('meas_rf_line', str(self.meas_rf_line), '')
        return stg

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

    def __init__(self, params=None, isonline=True):
        """."""
        params = ACORMParams() if params is None else params
        super().__init__(
            params=params, target=self._do_measure, isonline=isonline)
        self.bpms = dict()

        self.sofb_data = SOFBFactory.create('SI')
        self.configdb = _ConfigDBClient(config_type='si_orbcorr_respm')
        if self.isonline:
            self._create_devices()

    def load_and_apply(self, fname: str):
        """Load and apply `data` and `params` from pickle file.

        Args:
            fname (str): name of the pickle file. Extension is not needed.

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

        anl = self.analysis['rf']
        mat[:sofb.nr_bpms, -1] = anl['mat_colx']
        mat[sofb.nr_bpms:, -1] = anl['mat_coly']
        return mat

    def process_data(
            self, mag_idx_ini=None, mag_min_freq=None, mag_max_freq=None,
            rf_transition_length=10):
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
        self.analysis['magnets'] = self._process_data_magnets(
            self.data['magnets'], mag_idx_ini, mag_min_freq, mag_max_freq)
        rf_data = self.data.get('rf')
        if rf_data is not None:
            self.analysis['rf'] = self._process_data_rf(
                rf_data, rf_transition_length)
        bpms_data = self.data.get('bpms_noise')
        if bpms_data is not None:
            self.analysis['bpms_noise'] = self._process_data_bpms_noise(
                bpms_data)

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
        orbx, orby = self.bpms.get_mturn_orbit()
        bpm0 = self.bpms.devices[0]
        csbpm = self.bpms.csbpm

        data = dict()
        data['orbx'] = orbx
        data['orby'] = orby
        data['bpms_acq_rate'] = csbpm.AcqChan._fields[bpm0.acq_channel]
        data['bpms_nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['bpms_nrsamples_post'] = bpm0.acq_nrsamples_post
        data['bpms_trig_delay_raw'] = self.devices['trigbpms'].delay_raw
        data['bpms_switching_mode'] = csbpm.SwModes._fields[
            bpm0.switching_mode]
        return data

    def get_general_data(self):
        """Get general purpose data.

        Returns:
            dict: general purpose data.

        """
        data = dict()
        data['timestamp'] = _time.time()
        data['rf_frequency'] = self.devices['rfgen'].frequency
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
        self.devices['rfgen'] = RFGen()
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
        self.bpms = FamBPMs(FamBPMs.DEVICES.SI)
        self.devices['fambpms'] = self.bpms
        print(f'ET: = {_time.time()-t00:.2f}s')

    # ---------------- Data Measurement methods ----------------

    def _do_measure(self):
        if self.params.meas_bpms_noise:
            self.data['bpms_noise'] = self._do_measure_bpms_noise()

        if self.params.meas_rf_line:
            self.data['rf'] = self._do_measure_rf_line()

        self.data['magnets'] = self._do_measure_magnets()

    def _do_measure_bpms_noise(self):
        elt = _time.time()

        print('Measuring BPMs Noise:')

        t00 = _time.time()
        print('    Configuring BPMs...', end='')
        self._config_bpms(self.params.nr_points)
        self._config_timing(self.params.delay_corrs)
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        t00 = _time.time()
        print('    Sending Trigger signal...', end='')
        self.bpms.mturn_reset_flags()
        self.devices['evt_study'].cmd_external_trigger()
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        # Wait BPMs PV to update with new data
        t00 = _time.time()
        print('    Waiting BPMs to update...', end='')
        ret = self.bpms.mturn_wait_update_flags(
            timeout=self.params.timeout_bpms)
        if ret:
            print(
                'Problem: timed out waiting BPMs update. '
                f'Error code: {ret:d}')
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        # get data
        data = self.get_general_data()
        data.update(self.get_bpms_data())

        elt -= _time.time()
        elt *= -1
        print(f'    Elapsed Time: {elt:.2f}s')
        return data

    def _do_measure_rf_line(self):
        elt = _time.time()
        print('Measuring RF Line:')

        t00 = _time.time()
        print('    Configuring BPMs...', end='')
        self._config_bpms(self.params.nr_points)
        self._config_timing(self.params.delay_corrs)
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        t00 = _time.time()
        print('    Sending Trigger signal...', end='')
        self.bpms.mturn_reset_flags()
        self.devices['evt_study'].cmd_external_trigger()

        t00 = _time.time()
        print('    Sweep RF...', end='')
        thr = _Thread(
            target=self._sweep_rf,
            args=(self.params.rf_kick, self.params.exc_rf),
            daemon=True)

        _time.sleep(self.params.delay_rf)
        thr.start()
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        # Wait BPMs PV to update with new data
        t00 = _time.time()
        print('    Waiting BPMs to update...', end='')
        ret = self.bpms.mturn_wait_update_flags(
            timeout=self.params.timeout_bpms)
        if ret:
            print(
                'Problem: timed out waiting BPMs update. '
                f'Error code: {ret:d}')
        print(f'Done! ET: {_time.time()-t00:.2f}s')

        # get data
        data = self.get_general_data()
        data.update(self.get_bpms_data())
        data['rf_kick'] = self.params.rf_kick

        elt -= _time.time()
        elt *= -1
        print(f'    Elapsed Time: {elt:.2f}s')
        return data

    def _sweep_rf(self, rf_kick, exc_duration):
        rfgen = self.devices['rfgen']
        freq0 = rfgen.frequency
        rfgen.frequency = freq0 - rf_kick
        _time.sleep(exc_duration/2)
        rfgen.frequency = freq0 + rf_kick
        _time.sleep(exc_duration/2)
        rfgen.frequency = freq0

    def _do_measure_magnets(self):
        elt0 = _time.time()
        data_mags = []

        print('Measuring Magnets:')

        # Shift correctors so first corrector is 01M1
        chs_shifted = self._shift_list(self.sofb_data.ch_names, 1)
        cvs_shifted = self._shift_list(self.sofb_data.cv_names, 1)

        # set operation mode to slowref
        if not self._change_corrs_opmode('slowref'):
            print('Problem: Correctors not in SlowRef mode.')
            return data_mags

        exc_duration = self.params.exc_duration
        ch_kick = self.params.ch_kick
        cv_kick = self.params.cv_kick
        freqh = self.params.ch_freqs
        freqv = self.params.cv_freqs
        nr_points = self.params.nr_points
        nr_secs_acq = self.params.nr_sectors_per_acq
        secs_to_meas = self.params.sectors_to_measure
        nr_secs = len(secs_to_meas)

        loop_size = nr_secs // nr_secs_acq
        loop_size += 1 if nr_secs % nr_secs_acq else 0
        for itr in range(loop_size):
            elt = _time.time()

            secs = secs_to_meas[itr*nr_secs_acq:nr_secs_acq*(itr+1)]
            if len(list(set(secs))) < len(secs):
                print('Problem: Same sectors in same acquisition.')
                break

            print(f'Sectors: '+', '.join(f'{s:02d}:' for s in secs))
            chs_slc, cvs_slc = [], []
            freqhe, freqve = [], []
            for sec in secs:
                chs_slc.extend(chs_shifted[(sec-1)*6:sec*6])
                cvs_slc.extend(cvs_shifted[(sec-1)*8:sec*8])
                freqhe.extend(freqh)
                freqve.extend(freqv)

            t00 = _time.time()
            print('    Configuring BPMs and timing...', end='')
            # No need to configure BPMs when they are already configured
            if not itr or len(secs) != nr_secs_acq:
                self._config_bpms(nr_points * len(secs))
            self._config_timing(self.params.delay_corrs, secs)
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # configure correctors
            t00 = _time.time()
            print('    Configuring Correctors...', end='')
            self._config_correctors(chs_slc, ch_kick, freqhe, exc_duration)
            self._config_correctors(cvs_slc, cv_kick, freqve, exc_duration)
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # set operation mode to cycle
            t00 = _time.time()
            print('    Changing Correctors to Cycle...', end='')
            if not self._change_corrs_opmode('cycle', chs_slc + cvs_slc):
                print('Problem: Correctors not in Cycle mode.')
                break
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # send event through timing system to start acquisitions
            t00 = _time.time()
            print('    Sending Timing signal...', end='')
            self.bpms.mturn_reset_flags()
            self.devices['evt_study'].cmd_external_trigger()
            print(f'Done! ET: {_time.time()-t00:.2f}s')

            # Wait BPMs PV to update with new data
            t00 = _time.time()
            print('    Waiting BPMs to update...', end='')
            ret = self.bpms.mturn_wait_update_flags(
                timeout=self.params.timeout_bpms)
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
            for i, _ in enumerate(secs):
                datas = _dcopy(data)
                datas['orbx'] = orbx[i*nr_points:(i+1)*nr_points]
                datas['orby'] = orby[i*nr_points:(i+1)*nr_points]
                chs_s = chs_slc[i*6:(i+1)*6]
                cvs_s = cvs_slc[i*8:(i+1)*8]
                datas.update(self.get_magnets_data(
                    chs_used=chs_s, cvs_used=cvs_s))
                data_mags.append(datas)

            # set operation mode to slowref
            t00 = _time.time()
            print('    Changing Correctors to SlowRef...', end='')
            if not self._wait_cycle_to_finish(chs_slc + cvs_slc):
                print('Problem: Cycle still not finished.')
                break
            if not self._change_corrs_opmode('slowref', chs_slc + cvs_slc):
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

    def _process_data_rf(self, rf_data, transition_length=10):
        def _fitting_model(tim, *params):
            idx1, idx2, idx3, amp1, amp2 = params
            y = _np.zeros(tim.size)
            y[int(idx1):int(idx2)] = amp1
            y[int(idx2):int(idx3)] = amp2
            return y

        anly = dict()

        fsamp = FamBPMs.get_sampling_frequency(
            rf_data['rf_frequency'], rf_data['bpms_acq_rate'])
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
        exc_dur = self.params.exc_rf
        dly = self.params.delay_rf
        rf_kick = rf_data['rf_kick']
        idx1 = (tim > dly).nonzero()[0][0]
        idx2 = (tim > (exc_dur/2+dly)).nonzero()[0][0]
        idx3 = (tim > (exc_dur+dly)).nonzero()[0][0]
        etax_avg = 0.033e6  # average dispersion function, in [um]
        amp1 = -rf_kick * etax_avg / 1.7e-4 / 499665400
        amp2 = rf_kick * etax_avg / 1.7e-4 / 499665400
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
        anly['mat_colx'] = (orbx_pos - orbx_neg) / rf_kick / 2
        anly['mat_coly'] = (orby_pos - orby_neg) / rf_kick / 2
        return anly

    def _process_data_bpms_noise(self, bpms_data):
        sofb = self.sofb_data
        anly = dict()

        fsamp = FamBPMs.get_sampling_frequency(
            bpms_data['rf_frequency'], bpms_data['bpms_acq_rate'])
        dtim = 1/fsamp
        freqs0 = _np.r_[self.params.ch_freqs, self.params.cv_freqs]
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

        sch = len(self.params.ch_freqs)
        scv = len(self.params.cv_freqs)

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

    def _process_data_magnets(
            self, magnets_data, idx_ini=None, min_freq=None, max_freq=None):
        """."""
        sofb = self.sofb_data
        corr2idx = {name: i for i, name in enumerate(sofb.ch_names)}
        corr2idx.update({name: 120+i for i, name in enumerate(sofb.cv_names)})

        analysis = []
        for data in magnets_data:
            anly = dict()
            fsamp = FamBPMs.get_sampling_frequency(
                data['rf_frequency'], data['bpms_acq_rate'])
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

    def _config_bpms(self, nr_points):
        self.bpms.mturn_config_acquisition(
            acq_rate=self.params.acq_rate,
            nr_points_before=0, nr_points_after=nr_points,
            repeat=False, external=True)

    # ----------------- Timing related methods -----------------------

    def _config_timing(self, cm_delay, sectors=None):
        trigbpm = self.devices['trigbpms']
        trigcorr = self.devices['trigcorrs']
        evt_study = self.devices['evt_study']
        evg = self.devices['evg']

        trigbpm.delay = 0.0
        trigbpm.nr_pulses = 1
        trigbpm.source = 'Study'

        trigcorr.nr_pulses = 1
        trigcorr.source = 'Study'
        ftim = self.devices['rfgen'].frequency / 4  # timing base frequency
        dly = int(cm_delay * ftim)

        # configure event Study to be in External mode
        evt_study.delay = 0
        evt_study.mode = 'External'

        # update event configurations in EVG
        evg.cmd_update_events()

        if sectors is None:
            trigcorr.delay_raw = dly
            print('Done!')
            return

        trigcorr.delay_raw = 0
        # Calculate delta_delay for correctors to be as close as possible to a
        # multiple of the the sampling period to ensure repeatability of
        # experiment along the sectors excited during single acquisition:
        fsamp = FamBPMs.get_sampling_frequency(
            self.devices['rfgen'].frequency, self.params.acq_rate)
        secs_delta_dly = _np.arange(len(sectors), dtype=float)
        secs_delta_dly *= self.params.nr_points / fsamp
        secs_delta_dlyr = _np.round(secs_delta_dly * ftim)

        delta_delay_raw = _np.zeros(trigcorr.delta_delay_raw.size)
        low_level = [_PVName(n) for n in trigcorr.low_level_triggers]
        for sec, ddlyr in zip(sectors, secs_delta_dlyr):
            # Find all low level triggers of this sector and set their delay:
            for j, llt in enumerate(low_level):
                if llt.sub.startswith(f'{sec:02d}'):
                    delta_delay_raw[j] = ddlyr + dly
        trigcorr.delta_delay_raw = delta_delay_raw

    # ----------------- Correctors related methods -----------------------

    def _config_correctors(self, corr_names, kick, freq_vector, exc_duration):
        """."""
        for i, cmn in enumerate(corr_names):
            cmo = self.devices[cmn]
            conv = self.devices[cmn+':StrengthConv'].conv_strength_2_current
            cmo.cycle_type = cmo.CYCLETYPE.Sine
            cmo.cycle_freq = freq_vector[i]
            cmo.cycle_ampl = conv(kick)
            cmo.cycle_offset = cmo.currentref_mon
            cmo.cycle_theta_begin = 0
            cmo.cycle_theta_end = 0
            cmo.cycle_num_cycles = int(exc_duration * freq_vector[i])

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
