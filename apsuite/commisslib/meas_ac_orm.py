"""Main module."""

import operator as _opr
import time as _time
from copy import deepcopy as _dcopy
from functools import reduce as _red
from threading import Thread as _Thread

import matplotlib.pyplot as _mplt
import numpy as _np
import pyaccel as _pa
from mathphys.functions import (
    get_namedtuple as _get_namedtuple,
    load as _load,
    save as _save
)
from scipy.signal import (
    find_peaks as _find_peaks,
    savgol_filter as _savgol_filter
)
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient
from siriuspy.devices import (
    ASLLRF,
    CurrInfoSI,
    Event,
    EVG,
    FamBPMs,
    PowerSupply,
    RFGen,
    StrengthConv,
    SOFB,
    Trigger,
    Tune
)
from siriuspy.search import LLTimeSearch as _LLTime
from siriuspy.sofb.csdev import SOFBFactory

from .. import asparams as _asparams
from ..utils import (
    ParamsBaseClass as _ParamsBaseClass,
    ThreadedMeasBaseClass as _ThreadBaseClass
)
from .meas_bpms_signals import AcqBPMsSignals as _AcqBPMsSignals
from .measure_orbit_stability import OrbitAnalysis as _OrbitAnalysis


class ACORMParams(_ParamsBaseClass):
    """."""

    RFModes = _get_namedtuple('RFModes', ('Step', 'Phase'))

    def __init__(self):
        """."""
        super().__init__()
        self.timeout_bpms = 60  # [s]
        self.timeout_correctors = 20  # [s]
        self.ref_respmat_name = 'ref_respmat'
        self.meas_bpms_noise = True
        self.meas_rf_line = True

        self.corrs_ch2meas = 'all'
        self.corrs_cv2meas = 'all'
        self.corrs_nrruns_per_acq = 4
        self.corrs_excit_time = 4  # [s]
        self.corrs_delay = 5e-3  # [s]
        self.correct_orbit_between_acqs = True
        self.corrs_norm_kicks = False
        self.corrs_ch_kick = 5  # [urad]
        self.corrs_cv_kick = 5  # [urad]
        self.corrs_dorb1ch = 20  # [um]
        self.corrs_dorb1cv = 20  # [um]
        freqs = self.find_primes(14, start=100)
        self.corrs_ch_freqs = freqs[:6]
        self.corrs_cv_freqs = freqs[6 : 6 + 8]

        self.rf_excit_time = 1  # [s]
        self._rf_mode = self.RFModes.Phase
        self.rf_step_kick = 5  # [Hz]
        self.rf_step_delay = 200e-3  # [s]
        self.rf_phase_amp = 2  # [°]
        self._rf_llrf2use = 'B'  # (A or B)

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += ftmp('timeout_bpms', self.timeout_bpms, '[s]')
        stg += ftmp('timeout_correctors', self.timeout_correctors, '[s]')
        stg += stmp('ref_respmat_name', self.ref_respmat_name, '')
        stg += stmp('meas_bpms_noise', str(self.meas_bpms_noise), '')
        stg += stmp('meas_rf_line', str(self.meas_rf_line), '')

        stg += stmp('corrs_ch2meas', str(self.corrs_ch2meas), '')
        stg += stmp('corrs_cv2meas', str(self.corrs_cv2meas), '')
        stg += dtmp('corrs_nrruns_per_acq', self.corrs_nrruns_per_acq, '')
        stg += ftmp('corrs_excit_time', self.corrs_excit_time, '[s]')
        stg += ftmp('corrs_delay', self.corrs_delay, '[s]')
        stg += stmp(
            'correct_orbit_between_acqs',
            str(self.correct_orbit_between_acqs),
            '',
        )
        stg += stmp('corrs_norm_kicks', str(self.corrs_norm_kicks), '')
        stg += ftmp(
            'corrs_ch_kick',
            self.corrs_ch_kick,
            '[urad] (only used if corrs_norm_kicks == False)',
        )
        stg += ftmp(
            'corrs_cv_kick',
            self.corrs_cv_kick,
            '[urad] (only used if corrs_norm_kicks == False)',
        )
        stg += ftmp(
            'corrs_dorb1ch',
            self.corrs_dorb1ch,
            '[um] (only used if corrs_norm_kicks == True)',
        )
        stg += ftmp(
            'corrs_dorb1cv',
            self.corrs_dorb1cv,
            '[um] (only used if corrs_norm_kicks == True)',
        )
        stg += stmp('corrs_ch_freqs', str(self.corrs_ch_freqs), '[Hz]')
        stg += stmp('corrs_cv_freqs', str(self.corrs_cv_freqs), '[Hz]')

        stg += ftmp('rf_excit_time', self.rf_excit_time, '[s]')
        stg += stmp('rf_mode', self.rf_mode_str, str(self.RFModes))
        stg += ftmp('rf_step_kick', self.rf_step_kick, '[Hz]')
        stg += ftmp('rf_step_delay', self.rf_step_delay, '[s]')
        stg += ftmp('rf_phase_amp', self.rf_phase_amp, '[°]')
        stg += ftmp('rf_llrf2use', self.rf_llrf2use, '(A or B)')
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

    @property
    def rf_llrf2use(self):
        """."""
        return self._rf_llrf2use

    @rf_llrf2use.setter
    def rf_llrf2use(self, value):
        if isinstance(value, str) and value.upper() in {'A', 'B'}:
            self._rf_llrf2use = value.upper()
        else:
            raise ValueError('value must be "A" or "B"')

    @staticmethod
    def find_primes(n_primes, start=3):
        """."""
        primes = []
        i = start
        while True:
            for j in range(3, int(_np.sqrt(i)) + 1, 2):
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

    RF_CONDITIONING_FREQ = 1 / 12 / 4e6  # in units of RF frequency
    RF_CONDITIONING_VOLTAGEMIN = 30  # [mV]
    RF_CONDITIONING_DUTY = 55  # [%]

    def __init__(self, isonline=True):
        """."""
        self.params = ACORMParams()
        super().__init__(
            params=self.params, target=self._do_measure, isonline=isonline
        )
        self.bpms = None
        self.verbose = True

        self.sofb_data = SOFBFactory.create('SI')
        self.configdb = _ConfigDBClient(config_type='si_orbcorr_respm')
        self._ref_respmat = dict()
        self._meas_finished_ok = True

        if self.isonline:
            self._create_devices()

    filter_data_frequencies = staticmethod(
        _AcqBPMsSignals.filter_data_frequencies
    )
    calc_svd = staticmethod(_AcqBPMsSignals.calc_svd)

    @staticmethod
    def fitting_matrix(tim, freqs, num_cycles=None, idx_ini=None):
        """Create the matrix used for fitting of fourier components.

        The ordering of the matrix is the following:
           mat[i, 2*j] = cos(2*pi*freqs[j]*tim[i])
           mat[i, 2*j+1] = sin(2*pi*freqs[j]*tim[i])

        Args:
            tim (numpy.ndarray): array with times
            freqs (numpy.ndarray): array with frequencies to fit.
            num_cycles (numpy.ndarray, optional): number of cycles of each
                frequency. If not provided, all data range will be considered.
            idx_ini (int|list|tuple|numpy.ndarray, optional): starting index
                for fitting. If it is an iterable, must have the same size as
                freqs. Defaults to None, which means the first index will be
                the starting point.

        Returns:
            numpy.ndarray: fitting matrix (len(tim), 2*len(freqs))

        """
        if idx_ini is None:
            idx_ini = _np.zeros(freqs.shape, dtype=int)
        elif not isinstance(idx_ini, (list, tuple, _np.ndarray)):
            idx_ini = _np.full(freqs.shape, idx_ini, dtype=int)

        mat = _np.zeros((tim.size, 2 * freqs.size))
        mat2 = mat.copy()
        arg = 2 * _np.pi * freqs[None, :] * tim[:, None]
        cos = _np.cos(arg)
        sin = _np.sin(arg)
        idx_ini = _np.vstack([idx_ini, idx_ini]).T.ravel()

        if num_cycles is not None:
            cond = arg > 2 * _np.pi * num_cycles[None, :]
            cos[cond] = 0
            sin[cond] = 0
        mat[:, ::2] = cos
        mat[:, 1::2] = sin

        for i, idx in enumerate(idx_ini):
            if not idx:
                mat2[:, i] = mat[:, i]
            else:
                mat2[idx:, i] = mat[:-idx, i]
        return mat2

    def get_ref_respmat(self):
        """Get reference response matrix from configdb server."""
        name = self.params.ref_respmat_name
        if name not in self._ref_respmat:
            self._ref_respmat[name] = _np.array(
                self.configdb.get_config_value(name)
            )
        return self._ref_respmat[name].copy()

    @classmethod
    def fit_fourier_components(
        cls, data, freqs, dtim, num_cycles=None, idx_ini=None, pinv=None
    ):
        """Fit Fourier components in signal for the given frequencies.

        Args:
            data (numpy.ndarray, NxM): signal to be fitted consisting of M
                columns of data.
            freqs (numpy.ndarray, K): K frequencies to fit Fourier components.
            dtim (numpy.ndarray, N): time vector for data columns.
            num_cycles (num.ndarray, K, optional): number of cycles of each
                frequency. If not provided, all data range will be considered.
                Not used if pinv is not None.
            idx_ini (int|list|tuple|numpy.ndarray, optional): starting index
                for fitting. If it is an iterable, must have the same size as
                freqs. Defaults to None, which means the first index will be
                the starting point. Not used if pinv is not None.
            pinv (numpy.ndarray, Mx2K, optional): if provided must be the
                pseudo inverve of the fitting matrix. Defaults to None, which
                means the fitting matrix and its pseudo-inverse will be
                calculated.

        Returns:
            cos (numpy.ndarray, KxM): Fourier cosine coefficients.
            sin (numpy.ndarray, KxM): Fourier sine coefficients.
            pinv (numpy.ndarray, Mx2K): pseudo-inverse of fitting matrix.

        """
        if pinv is None:
            tim = _np.arange(data.shape[0]) * dtim
            mat = cls.fitting_matrix(tim, freqs, num_cycles, idx_ini)
            u, s, vt = _np.linalg.svd(mat, full_matrices=False)
            pinv = vt.T / s @ u.T
            coeffs = pinv @ data
        else:
            siz = min(pinv.shape[1], data.shape[0])
            coeffs = pinv[:, :siz] @ data[:siz]
        # coeffs, *_ = _np.linalg.lstsq(mat, data, rcond=None)
        cos = coeffs[::2]
        sin = coeffs[1::2]
        return cos, sin, pinv

    @staticmethod
    def fit_calc_amp_and_phase(cos, sin):
        """."""
        amps = _np.sqrt(cos**2 + sin**2)
        phases = _np.arctan2(cos, sin)
        return amps, phases

    @classmethod
    def fitted_orbit(cls, cos, sin, freqs, tim, num_cycles=None, idx_ini=None):
        """."""
        mat = cls.fitting_matrix(tim, freqs, num_cycles, idx_ini)
        coeffs = _np.zeros((sin.shape[0] * 2, sin.shape[1]), dtype=float)
        coeffs[::2] = cos
        coeffs[1::2] = sin
        return mat @ coeffs

    @staticmethod
    def fit_fourier_components_naff(data, freqs0, dtim):
        """."""
        data = _AcqBPMsSignals.filter_data_frequencies(
            data, freqs0.min() - 10, freqs0.max() + 10, 1 / dtim
        )

        freqs = _np.zeros((len(freqs0), data.shape[1]))
        fourier = _np.zeros((len(freqs0), data.shape[1]), dtype=complex)

        fres, four = _pa.naff.naff_general(
            data.T, nr_ff=len(freqs0) + 4, is_real=True
        )

        fres = _np.abs(fres.T)
        ind = _np.argsort(fres, axis=0)
        fres = _np.take_along_axis(fres, ind, axis=0)
        four = _np.take_along_axis(four.T, ind, axis=0)
        fres /= dtim
        four *= 2

        for freq, fouri, fre, fou in zip(freqs.T, fourier.T, fres.T, four.T):
            idcs = _np.searchsorted(fre, freqs0)
            idsm1 = idcs.copy()
            idcs[idcs >= fre.size] = fre.size - 1
            idsm1[idcs > 0] -= 1
            dif = _np.abs(fre[idcs] - freqs0)
            difm1 = _np.abs(fre[idsm1] - freqs0)
            idcs_min = _np.where(dif < difm1, idcs, idsm1)
            dif = _np.where(dif < difm1, dif, difm1)
            fre = fre[idcs_min]
            fou = fou[idcs_min]
            cond = dif < 1
            freq[cond] = fre[cond]
            fouri[cond] = fou[cond]

        cos = fourier.real
        sin = fourier.imag

        return cos, sin

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
        corr = (arr1 * arr2).sum(axis=-2)
        corr /= _np.linalg.norm(arr1, axis=-2)
        corr /= _np.linalg.norm(arr2, axis=-2)
        return corr

    def save_loco_input_data(
        self,
        respmat_name: str,
        overwrite=False,
        save2servconf=False,
        extra_kwargs=None,
        matrix=None,
    ):
        """Save in `.pickle` format a dictionary with all LOCO input data.

        Args:
            respmat_name (str): Name of the response matrix. A matrix with
                this name will be saved in configdb.
            overwrite (bool, optional): whether to overwrite existing files.
                Defaults to False.
            save2servconf (bool, optional): whether to save response matrix to
                configDB server. Defaults to False.
            extra_kwargs (dict, optional): extra keyword arguments to save to
                loco input file. Defaults to None:

        """
        bpms_anly = self.analysis['bpms_noise']
        data = dict()
        data['method'] = 'AC'
        data['info'] = str(self.params)
        data['timestamp'] = bpms_anly['timestamp']
        data['rf_frequency'] = bpms_anly['rf_frequency']
        data['tunex'] = bpms_anly['tunex']
        data['tuney'] = bpms_anly['tuney']
        data['stored_current'] = bpms_anly['stored_current']
        data['bpm_variation'] = bpms_anly['bpm_variation']
        if matrix is None:
            matrix = self.build_respmat()
        data['respmat'] = matrix
        data.update(extra_kwargs or dict())
        if save2servconf:
            data['orbmat_name'] = respmat_name
            mat = self.save_respmat_to_configdb(respmat_name, mat=mat)
        _save(data, f'{respmat_name:s}', overwrite=overwrite)

    def save_analysis_dictionary(self, filename, overwrite=False):
        """Save internal analysis dict to file.

        Args:
            filename (str): Name of the file to save. It it does not start
                with 'analysis_', this prefix will be prepended to the name.
            overwrite (bool, optional): Whether or not to overwrite existing
                files. Defaults to False.

        """
        if not filename.startswith('analysis_'):
            filename = 'analysis_' + filename
        _save(self.analysis, filename, overwrite=overwrite)

    def load_analysis_dictionary(self, filename, overwrite=False):
        """Load file with analysis dictionary.

        Args:
            filename (str): Name of the file to load. It must start with
                'analysis_', otherwise an ValueError will be raised.
            overwrite (bool, optional): Whether or not to overwrite existing
                files. Defaults to False.

        Raises:
            ValueError: when filename does not start with 'analysis_'

        Returns:
            dict: analysis dictionary.

        """
        if not filename.startswith('analysis_'):
            raise ValueError('File name must start with "analysis_".')
        return _load(self.analysis, filename, overwrite=overwrite)

    def save_respmat_to_configdb(self, name: str, matrix=None):
        """Save response matrix to ConfigDb Server.

        Args:
            name (str): name of the matrix to saved.
            matrix (numpy.ndarray, 320x281, optional): matrix to save.
                If None the object will be built from the analysis data.
                Defaults to None.

        Returns:
            numpy.ndarray, 320x281: saved response matrix. Missing data is
                filled with zeros.

        """
        if matrix is None:
            matrix = self.build_respmat()
        self.configdb.insert_config(name, matrix)
        return matrix

    def build_respmat(self, propty='mat_cols'):
        """Build response matrix from previously analysed data.

        Returns:
            numpy.ndarray, 320x281: response matrix. Missing data is filled
                with zeros.

        """
        sofb = self.sofb_data
        mat = _np.zeros((2 * sofb.nr_bpms, sofb.nr_corrs), dtype=float)
        anls = self.analysis['magnets']
        for anly in anls:
            mat[: sofb.nr_bpms, anly['mat_idcs']] = anly[propty + 'x']
            mat[sofb.nr_bpms :, anly['mat_idcs']] = anly[propty + 'y']
        if 'rf' in self.analysis:
            anl = self.analysis['rf']
            if propty + 'x' in anl:
                mat[: sofb.nr_bpms, -1] = anl[propty + 'x']
                mat[sofb.nr_bpms :, -1] = anl[propty + 'y']
        return mat

    def process_data(
        self,
        mag_idx_ini=None,
        rf_step_trans_len=10,
        rf_phase_central_freq=None,
        rf_phase_window=10,
    ):
        """Process measured data.

        Args:
            mag_idx_ini ([type], optional): initial index of orbit waveform
                where magnets excitation start. Defaults to None.
            rf_step_trans_len (int, optional): Number of indices to ignore
                right before or after RF frequency changes in RF line
                measurements. Defaults to 10. Only used if step mode was used
                as RFMode in measurement.
            rf_phase_central_freq (float, optional): central frequency around
                which data will be filtered in RF line analysis. Defaults to
                None, which means the peak frequency in the range [1700, 2300]
                Hz will be used. Only used if phase mode was used as RFMode in
                measurement.
            rf_phase_window (float, optional): frequency width of the filter
                in RF line analysis. Defaults to 10Hz. Only used if phase mode
                was used as RFMode in measurement.

        """
        self.analysis['magnets'] = self._process_magnets(
            self.data['magnets'], mag_idx_ini
        )

        rf_d = self.data.get('rf')
        if rf_d is not None:
            if 'mode' in rf_d and rf_d['mode'] == self.params.RFModes.Phase:
                anly = self._process_rf_phase(
                    rf_d, rf_phase_window, central_freq=rf_phase_central_freq
                )
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
            'ch_names': [],
            'ch_amplitudes': [],
            'ch_offsets': [],
            'ch_kick_amplitudes': [],
            'ch_kick_offsets': [],
            'ch_frequency': [],
            'ch_num_cycles': [],
            'ch_cycle_type': [],
            'cv_names': [],
            'cv_amplitudes': [],
            'cv_offsets': [],
            'cv_kick_amplitudes': [],
            'cv_kick_offsets': [],
            'cv_frequency': [],
            'cv_num_cycles': [],
            'cv_cycle_type': [],
        }
        for cmn in chs_used:
            data['ch_names'].append(cmn)
            cm = self.devices[cmn]
            conv = self.devices[cmn + ':StrengthConv'].conv_current_2_strength
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
            conv = self.devices[cmn + ':StrengthConv'].conv_current_2_strength
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
        data['sampling_frequency'] = self.bpms.get_sampling_frequency(rf_freq)
        data['nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['nrsamples_post'] = bpm0.acq_nrsamples_post
        data['trig_delay_raw'] = self.devices['trigbpms'].delay_raw
        data['switching_mode'] = bpm0.switching_mode_str
        data['switching_frequency'] = self.bpms.get_switching_frequency(
            rf_freq
        )
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

    def check_measurement_finished_ok(self):
        """Check if measument finished without errors.

        Returns:
            bool: True if measurement finished without errors.

        """
        if self.ismeasuring:
            return False
        return self._meas_finished_ok

    def check_measurement_quality(self, mat_ac=None, mat_dc=None, dthres=1e-3):
        """Check whether last measurement have meaningful results.

        This method checks if the experiment has finished properly and
        compares the measured matrix with the reference respmat.
        If the diagonal terms of both matrices agree uppon the threshold dthres
        it will return True, otherwise it will return False.

        This method will call process_data if needed.

        Args:
            mat_ac (numpy.array, (320, 281), optional): The measured matrix.
                Defaults to None.
            mat_dc (numpy.array, (320, 281), optional): The reference matrix.
                Defaults to None.
            dthres (float, optional): The threshold used for comparison.
                Defaults to 1e-3.

        Returns:
            bool: Whether or not measurement was successful.

        """
        if not self._meas_finished_ok:
            return False

        if not self.analysis:
            self.process_data()

        if mat_ac is None:
            mat_ac = self.build_respmat()
        if mat_dc is None:
            mat_dc = self.get_ref_respmat()
        mat1 = mat_ac.reshape(2, -1, mat_ac.shape[-1])
        mat2 = mat_dc.reshape(2, -1, mat_dc.shape[-1])
        corr = 1 - self.calc_correlation(mat1, mat2)
        corr_ch = corr[:, :120]
        corr_cv = corr[:, 120:280]
        corr_rf = corr[:, -1]
        cond_ok = (corr_ch[0] < dthres).all()
        cond_ok &= (corr_cv[1] < dthres).all()
        cond_ok &= (corr_rf[0] < dthres).all()
        return cond_ok

    def plot_comparison_correlations(self, mat_ac=None, mat_dc=None):
        """Plot comparison of measured response matrix with reference respmat.

        Two graphics will be made, one comparing the column space of both
        matrix and another one comparing their Row space.

        Args:
            mat_ac (numpy.array, (320, 281), optional): The measured matrix.
                Defaults to None.
            mat_dc (numpy.array, (320, 281), optional): The reference matrix.
                Defaults to None.

        Returns:
            matplotlib.Figure: Figure object of the plot.
            tuple: Tuple with both axes of the figure;
            col_dcorr (np.ndarray, (2, 281)): One minus column space
                correlations.

        """
        if mat_ac is None:
            mat_ac = self.build_respmat()
        if mat_dc is None:
            mat_dc = self.get_ref_respmat()

        mat1 = mat_ac.reshape(2, -1, mat_ac.shape[-1])
        mat2 = mat_dc.reshape(2, -1, mat_dc.shape[-1])

        sofb = self.sofb_data
        fig, (ax, ay) = _mplt.subplots(2, 1, figsize=(8, 6))

        corr = 1 - self.calc_correlation(mat1, mat2)
        cch = corr[:, :120]
        ccv = corr[:, 120:280]
        crf = corr[:, -1]
        xch = sofb.ch_pos
        xcv = sofb.cv_pos

        ax.plot(xch, cch[0], '-o', label=r'$M_\mathrm{ch,x}$', color='C0')
        ax.plot(xch, cch[1], '-o', label=r'$M_\mathrm{ch,y}$', color='C4')
        ax.plot(xcv, ccv[0], '-o', label=r'$M_\mathrm{cv,x}$', color='C1')
        ax.plot(xcv, ccv[1], '-o', label=r'$M_\mathrm{cv,y}$', color='tab:red')

        ax.axhline(crf[0], label=r'$M_\mathrm{rf,x}$', ls='-', color='k')
        ax.axhline(crf[1], label=r'$M_\mathrm{rf,y}$', ls='--', color='k')

        ax.legend(framealpha=0.8, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Correctors Position [m]')
        ax.set_yscale('log')
        ax.set_title(
            r'$1-\mathrm{Cor}(\mathrm{Col}(M_\mathrm{AC}), '
            + r'\mathrm{Col}(M_\mathrm{DC}))$'
        )

        mat1 = mat1.swapaxes(-2, -1)
        mat2 = mat2.swapaxes(-2, -1)
        cch = 1 - self.calc_correlation(mat1[:, :120, :], mat2[:, :120, :])
        ccv = 1 - self.calc_correlation(
            mat1[:, 120:280, :], mat2[:, 120:280, :]
        )
        xbp = sofb.bpm_pos

        ay.plot(xbp, cch[0], '-o', label=r'$M_\mathrm{ch,x}$', color='C0')
        ay.plot(xbp, cch[1], '-o', label=r'$M_\mathrm{ch,y}$', color='C4')
        ay.plot(xbp, ccv[0], '-o', label=r'$M_\mathrm{cv,x}$', color='C1')
        ay.plot(xbp, ccv[1], '-o', label=r'$M_\mathrm{cv,y}$', color='tab:red')

        ay.legend(framealpha=0.8, loc='center left', bbox_to_anchor=(1, 0.5))
        ay.set_xlabel('BPMs Position [m]')
        ay.set_yscale('log')
        ay.set_title(
            r'$1-\mathrm{Cor}(\mathrm{Row}(M_\mathrm{AC}), '
            r'\mathrm{Row}(M_\mathrm{DC}))$'
        )

        fig.tight_layout()
        fig.show()
        return fig, (ax, ay), corr

    def plot_comparison_single_corrector(
        self, corr_idx, mat_ac=None, mat_dc=None
    ):
        """Plot single corrector signatures of measured and reference respmat.

        Args:
            corr_idx (int): corrector index [0, 119] for CH, [120, 279] for CV
                and 280 for RF.
            mat_ac (numpy.array, (320, 281), optional): The measured matrix.
                Defaults to None.
            mat_dc (numpy.array, (320, 281), optional): The reference matrix.
                Defaults to None.

        Returns:
            matplotlib.Figure: Figure object of the plot.
            tuple: Tuple with both axes of the figure;

        """
        if mat_ac is None:
            mat_ac = self.build_respmat()
        if mat_dc is None:
            mat_dc = self.get_ref_respmat()

        corr_names = self.sofb_data.ch_names + self.sofb_data.cv_names
        corr_names += ['RF']
        acm = mat_ac[:, corr_idx].copy()
        dcm = mat_dc[:, corr_idx].copy()
        dif = acm - dcm
        pos = self.sofb_data.bpm_pos

        fig, (ax, ay) = _mplt.subplots(2, 1, figsize=(8, 6))
        ax.set_title(f'{corr_names[corr_idx]:s}')
        ax.plot(pos, acm[:160], label='AC')
        ax.plot(pos, dcm[:160], label='DC')
        ax.plot(pos, dif[:160], label='Diff.', lw=0.5)
        ay.plot(pos, acm[160:], label='AC')
        ay.plot(pos, dcm[160:], label='DC')
        ay.plot(pos, dif[160:], label='Diff.', lw=0.5)
        ax.legend(loc='best')
        unit = '[um/urad]' if corr_idx != 280 else '[um/Hz]'
        ax.set_ylabel('Horizontal ' + unit)
        ay.set_ylabel('Vertical ' + unit)
        ay.set_xlabel('BPM position')
        fig.tight_layout()
        fig.show()
        return fig, (ax, ay)

    def plot_scale_conversion_factors(self):
        """Plot single corrector signatures of measured and reference respmat.

        Returns:
            matplotlib.Figure: Figure object of the plot.
            matplotlib.Axes: Axes of the figure;

        """
        mags_data = self.analysis['magnets']
        ch_f = _np.array([anl['mat_colsx_scale'] for anl in mags_data]).ravel()
        cv_f = _np.array([anl['mat_colsy_scale'] for anl in mags_data]).ravel()
        ch_p = self.sofb_data.ch_pos
        cv_p = self.sofb_data.cv_pos

        fig, ax = _mplt.subplots(1, 1, figsize=(6, 4))
        ax.set_title(r'Scale Factors $M_\mathrm{DC} / M_\mathrm{AC}$')
        ax.plot(ch_p, ch_f, label='CH')
        ax.plot(cv_p, cv_f, label='CV')
        ax.legend(loc='best')
        ax.set_ylabel('Relative Factor')
        ax.set_xlabel('Correctors Position [m]')
        fig.tight_layout()
        fig.show()
        return fig, ax

    def plot_phases_vs_amplitudes(self, title='', corrsidx2highlight=None):
        """."""
        fig = _mplt.figure(figsize=(8, 6))
        gs = _mplt.GridSpec(
            2,
            2,
            hspace=0.01,
            wspace=0.01,
            top=0.95,
            left=0.09,
            right=0.99,
            bottom=0.08,
        )
        ahx = fig.add_subplot(gs[0, 0])
        avy = fig.add_subplot(gs[1, 0], sharex=ahx, sharey=ahx)
        ahy = fig.add_subplot(gs[0, 1], sharey=ahx)
        avx = fig.add_subplot(gs[1, 1], sharey=ahx, sharex=ahy)

        phs = self.build_respmat(propty='phase') / _np.pi
        amp = self.build_respmat(propty='amp')

        kws = dict(marker='.', mfc='None', ms=1, ls='None', color='C0')
        spx = slice(None, 160)
        sch = slice(None, 120)
        spy = slice(160, None)
        scv = slice(120, 280)
        sel = spx, sch
        ahx.plot(amp[sel].ravel(), phs[sel].ravel(), **kws)
        sel = spy, sch
        ahy.plot(amp[sel].ravel(), phs[sel].ravel(), **kws)
        sel = spx, scv
        avx.plot(amp[sel].ravel(), phs[sel].ravel(), **kws)
        sel = spy, scv
        avy.plot(amp[sel].ravel(), phs[sel].ravel(), **kws)

        corridx = [] if corrsidx2highlight is None else corrsidx2highlight
        kws.pop('color')
        kws['ms'] = 2
        for i, idx in enumerate(corridx):
            cor = _mplt.cm.jet(i / (len(corridx) - 1))
            if idx < 120:
                ahx.plot(amp[spx, idx], phs[spx, idx], color=cor, **kws)
                ahy.plot(amp[spy, idx], phs[spy, idx], color=cor, **kws)
            else:
                avx.plot(amp[spx, idx], phs[spx, idx], color=cor, **kws)
                avy.plot(amp[spy, idx], phs[spy, idx], color=cor, **kws)

        fig.suptitle(title)

        kws = dict(xy=(0.8, 0.7), xycoords='axes fraction', fontsize='large')
        ahx.annotate(r'$M_\mathrm{CH,x}$', **kws)
        ahy.annotate(r'$M_\mathrm{CH,y}$', **kws)
        avx.annotate(r'$M_\mathrm{CV,x}$', **kws)
        avy.annotate(r'$M_\mathrm{CV,y}$', **kws)

        _mplt.setp(ahx.get_xticklabels(), visible=False)
        _mplt.setp(ahy.get_xticklabels(), visible=False)
        _mplt.setp(ahy.get_yticklabels(), visible=False)
        _mplt.setp(avx.get_yticklabels(), visible=False)
        avy.set_xlabel(r'Amplitudes [$\mu$m]')
        avx.set_xlabel(r'Amplitudes [$\mu$m]')
        ahx.set_ylabel(r'Phases [$\pi$]')
        avy.set_ylabel(r'Phases [$\pi$]')

        fig.show()
        return fig, ((ahx, ahy), (avy, avx))

    def plot_phases_histogram(self, title=''):
        """."""
        fig = _mplt.figure(figsize=(8, 6))
        gs = _mplt.GridSpec(
            2,
            2,
            hspace=0.01,
            wspace=0.01,
            top=0.95,
            left=0.09,
            right=0.99,
            bottom=0.08,
        )
        ahx = fig.add_subplot(gs[0, 0])
        avy = fig.add_subplot(gs[1, 0], sharex=ahx, sharey=ahx)
        ahy = fig.add_subplot(gs[0, 1], sharex=ahx, sharey=ahx)
        avx = fig.add_subplot(gs[1, 1], sharex=ahx, sharey=ahx)

        phs = self.build_respmat(propty='phase') / _np.pi

        kws = dict(bins=60)
        spx = slice(None, 160)
        sch = slice(None, 120)
        spy = slice(160, None)
        scv = slice(120, 280)
        sel = spx, sch
        ahx.hist(phs[sel].ravel(), **kws)
        sel = spy, sch
        ahy.hist(phs[sel].ravel(), **kws)
        sel = spx, scv
        avx.hist(phs[sel].ravel(), **kws)
        sel = spy, scv
        avy.hist(phs[sel].ravel(), **kws)

        fig.suptitle(title)

        kws = dict(xy=(0.45, 2000), xycoords='data', fontsize='large')
        ahx.annotate(r'$M_\mathrm{CH,x}$', **kws)
        ahy.annotate(r'$M_\mathrm{CH,y}$', **kws)
        avx.annotate(r'$M_\mathrm{CV,x}$', **kws)
        avy.annotate(r'$M_\mathrm{CV,y}$', **kws)

        _mplt.setp(ahx.get_xticklabels(), visible=False)
        _mplt.setp(ahy.get_xticklabels(), visible=False)
        _mplt.setp(ahy.get_yticklabels(), visible=False)
        _mplt.setp(avx.get_yticklabels(), visible=False)
        ahx.set_ylabel(r'Counts')
        avy.set_ylabel(r'Counts')
        avx.set_xlabel(r'Phases [$\pi$]')
        avy.set_xlabel(r'Phases [$\pi$]')

        fig.show()
        return fig, ((ahx, ahy), (avy, avx))

    def plot_bpms_fluctuations(self):
        """Plot BPMs flutuations statistics along BPMs and Correctors.

        Returns:
            fig: matplolib.Figure;
            axs: tuple with figure axes.

        """
        sofb = self.sofb_data
        fig, (ax, ay) = _mplt.subplots(2, 1, figsize=(8, 6))

        mat = self.analysis['bpms_noise']['bpm_variation']
        mat = mat.reshape(2, -1, mat.shape[-1])
        mat *= mat
        std = _np.sqrt(mat.mean(axis=1))
        cch = std[:, :120]
        ccv = std[:, 120:280]
        crf = std[:, -1]
        xch = sofb.ch_pos
        xcv = sofb.cv_pos

        ax.plot(xch, cch[0], '-o', label=r'$M_\mathrm{ch,x}$', color='C0')
        ax.plot(xch, cch[1], '-o', label=r'$M_\mathrm{ch,y}$', color='C4')
        ax.plot(xcv, ccv[0], '-o', label=r'$M_\mathrm{cv,x}$', color='C1')
        ax.plot(xcv, ccv[1], '-o', label=r'$M_\mathrm{cv,y}$', color='tab:red')

        ax.axhline(crf[0], label=r'$M_\mathrm{rf,x}$', ls='-', color='k')
        ax.axhline(crf[1], label=r'$M_\mathrm{rf,y}$', ls='--', color='k')

        ax.legend(framealpha=0.8, loc='center left', bbox_to_anchor=(1, 0.5))
        ax.set_xlabel('Correctors Position [m]')
        ax.set_ylabel(r'Fluctuation [$\mu$m]')
        ax.set_yscale('log')
        ax.set_title('BPMs Variation along Correctors')

        cch = _np.sqrt(mat[:, :, :120].mean(axis=-1))
        ccv = _np.sqrt(mat[:, :, 120:260].mean(axis=-1))
        xbp = sofb.bpm_pos

        ay.plot(xbp, cch[0], '-o', label=r'$M_\mathrm{ch,x}$', color='C0')
        ay.plot(xbp, cch[1], '-o', label=r'$M_\mathrm{ch,y}$', color='C4')
        ay.plot(xbp, ccv[0], '-o', label=r'$M_\mathrm{cv,x}$', color='C1')
        ay.plot(xbp, ccv[1], '-o', label=r'$M_\mathrm{cv,y}$', color='tab:red')

        ay.legend(framealpha=0.8, loc='center left', bbox_to_anchor=(1, 0.5))
        ay.set_xlabel('BPMs Position [m]')
        ax.set_ylabel(r'Fluctuation [$\mu$m]')
        ay.set_yscale('log')
        ay.set_title('BPMs Variation along BPMs')

        fig.tight_layout()
        fig.show()
        return fig, (ax, ay)

    def plot_orbit_residue_after_fitting(
        self, bpm_idx=0, excit_idx=0, time_domain=True
    ):
        """Plot orbit residue after fitting.

        Args:
            bpm_idx (int, optional): Index of the BPM to plot. Defaults to 0.
            excit_idx (int, optional): Index of the correctors excitation to
                plot. Defaults to 0.
            time_domain (bool, optional): Whether to plot in time or in
                frequency domain. Defaults to True.

        Returns:
            figure: matplotlib.Figure;
            axes: Tuple with figure axes;
            orbx_fit: vector with fitted horizontal orbit
            orby_fit: vector with fitted vertical orbit
            dorbx: vector with horizontal residue
            dorby: vector with vertical residue

        """
        anly = self.analysis['magnets'][excit_idx]
        dtim = anly['dtim']
        freqs0 = anly['freqs0']
        num_cycles = anly['num_cycles']
        idx_ini = anly['idx_ini']
        cosx = anly['cosx'].T
        cosy = anly['cosy'].T
        sinx = anly['sinx'].T
        siny = anly['siny'].T
        ampx = anly['ampx']
        ampy = anly['ampy']

        orbx = self.data['magnets'][excit_idx]['orbx'].copy()
        orby = self.data['magnets'][excit_idx]['orby'].copy()
        orbx -= orbx.mean(axis=0)
        orby -= orby.mean(axis=0)

        tim = _np.arange(orbx.shape[0]) * dtim
        orbx_fit = self.fitted_orbit(
            cosx, sinx, freqs0, tim, num_cycles, idx_ini
        )
        orby_fit = self.fitted_orbit(
            cosy, siny, freqs0, tim, num_cycles, idx_ini
        )
        dorbx = orbx - orbx_fit
        dorby = orby - orby_fit

        fig, (ax, ay) = _mplt.subplots(2, 1, sharex=True)

        bpmx = orbx[:, bpm_idx]
        dbpmx = dorbx[:, bpm_idx]
        bpmy = orby[:, bpm_idx]
        dbpmy = dorby[:, bpm_idx]
        if time_domain:
            tim = _np.arange(bpmx.size) * dtim * 1e3
            ax.plot(tim, bpmx, '.-', label='Orbit')
            ax.plot(tim, dbpmx, '.-', label='Diff.')
            ay.plot(tim, bpmy, '.-', label='Orbit')
            ay.plot(tim, dbpmy, '.-', label='Diff.')
            ax.set_xlabel('Time [ms]')
            ay.set_xlabel('Time [ms]')
        else:
            siz = bpmx.size
            rfft = _np.fft.rfft
            frs = _np.fft.rfftfreq(siz, d=dtim)
            ax.plot(frs, _np.abs(rfft(bpmx)) * 2 / siz, '.-', label='Orbit')
            ax.plot(frs, _np.abs(rfft(dbpmx)) * 2 / siz, '.-', label='Diff')
            ax.plot(freqs0, ampx[bpm_idx], 'o', label='Fitted')
            ay.plot(frs, _np.abs(rfft(bpmy)) * 2 / siz, '.-', label='Orbit')
            ay.plot(frs, _np.abs(rfft(dbpmy)) * 2 / siz, '.-', label='Diff')
            ay.plot(freqs0, ampy[bpm_idx], 'o', label='Fitted')
            for f in freqs0:
                ax.axvline(f, ls='--', color='k')
                ay.axvline(f, ls='--', color='k')
            ax.set_xlim([freqs0.min() - 2, freqs0.max() + 2])
            ax.set_ylim([4e-4, None])
            ay.set_ylim([5e-4, None])
            ax.set_yscale('log')
            ay.set_yscale('log')
            ax.set_xlabel('Frequency [Hz]')
            ay.set_xlabel('Frequency [Hz]')

        ax.set_ylabel('Horizontal [um]')
        ay.set_ylabel('Vertical [um]')
        ax.legend(
            loc='lower center',
            bbox_to_anchor=(0.5, 1),
            fontsize='small',
            ncol=4,
        )
        fig.tight_layout()
        fig.show()
        return fig, (ax, ay), orbx, orby, dorbx, dorby

    # ------------------ Auxiliary Methods ------------------

    def _log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def _create_devices(self):
        # Create objects to convert kicks to current
        t00 = _time.time()
        self._log('Creating kick converters  -> ', end='')
        self.devices.update({
            n + ':StrengthConv': StrengthConv(n, 'Ref-Mon')
            for n in self.sofb_data.ch_names
        })
        self.devices.update({
            n + ':StrengthConv': StrengthConv(n, 'Ref-Mon')
            for n in self.sofb_data.cv_names
        })
        self._log(f'ET: = {_time.time() - t00:.2f}s')

        # Create objects to interact with correctors
        t00 = _time.time()
        self._log('Creating correctors       -> ', end='')
        props = [
            'Kick-SP',
            'OpMode-Sel',
            'OpMode-Sts',
            'Current-RB',
            'Current-SP',
            'CycleType-Sel',
            'CycleFreq-SP',
            'CycleAmpl-SP',
            'CurrentRef-Mon',
            'CycleOffset-SP',
            'CycleAuxParam-RB',
            'CycleAuxParam-SP',
            'CycleNrCycles-SP',
            'CycleAmpl-RB',
            'CycleOffset-RB',
            'CycleFreq-RB',
            'CycleNrCycles-RB',
            'CycleType-Sts',
            'CycleEnbl-Mon',
            'Current-Mon',
            'CycleAuxParam-SP',
            'CycleAuxParam-RB',
            'ParamPWMFreq-Cte',
        ]
        self.devices.update({
            nme: PowerSupply(nme, props2init=props)
            for nme in self.sofb_data.ch_names
        })
        self.devices.update({
            nme: PowerSupply(nme, props2init=props)
            for nme in self.sofb_data.cv_names
        })
        self._log(f'ET: = {_time.time() - t00:.2f}s')

        # Create object to get stored current
        t00 = _time.time()
        self._log('Creating General Devices  -> ', end='')
        self.devices['currinfo'] = CurrInfoSI()

        # Create RF generator object
        props = ['GeneralFreq-SP', 'GeneralFreq-RB']
        self.devices['rfgen'] = RFGen(props2init=props)

        # Create LLRF Devices
        props = [
            'ALRef-SP',
            'AmpRefMin-SP',
            'PLRef-SP',
            'PhsRefMin-SP',
            'CondDuty-SP',
            'CondDuty-RB',
            'CondEnbl-Sel',
            'CondEnbl-Sts',
            'CondDutyCycle-Mon',
        ]
        self.devices['llrfa'] = ASLLRF(ASLLRF.DEVICES.SIA, props2init=props)
        self.devices['llrfb'] = ASLLRF(ASLLRF.DEVICES.SIB, props2init=props)

        # Create Tune object:
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self._log(f'ET: = {_time.time() - t00:.2f}s')

        # Create BPMs trigger:
        t00 = _time.time()
        self._log('Creating Timing           -> ', end='')
        props = [
            'Src-Sts',
            'NrPulses-RB',
            'DelayRaw-RB',
            'Src-Sel',
            'NrPulses-SP',
            'DelayRaw-SP',
        ]
        self.devices['trigbpms'] = Trigger('SI-Fam:TI-BPM', props2init=props)

        # Create Correctors Trigger:
        props = [
            'Src-Sts',
            'NrPulses-RB',
            'DelayRaw-RB',
            'DeltaDelayRaw-RB',
            'Src-Sel',
            'NrPulses-SP',
            'DelayRaw-SP',
            'LowLvlTriggers-Cte',
            'DeltaDelayRaw-SP',
        ]
        self.devices['trigcorrs'] = Trigger(
            'SI-Glob:TI-Mags-Corrs', props2init=props
        )

        # Create event to start data acquisition sinchronously:
        props = [
            'Mode-Sts',
            'DelayRaw-RB',
            'Mode-Sel',
            'DelayRaw-SP',
            'ExtTrig-Cmd',
        ]
        self.devices['evt_study'] = Event('Study', props2init=props)
        props = ['ContinuousEvt-Sts', 'UpdateEvt-Cmd']
        self.devices['evg'] = EVG(props2init=props)
        self._log(f'ET: = {_time.time() - t00:.2f}s')

        # Create BPMs
        t00 = _time.time()
        self._log('Creating BPMs             -> ', end='')
        self.bpms = FamBPMs(mturn_signals2acq='XY', props2init='acq')
        self.devices['fambpms'] = self.bpms
        self._log(f'ET: = {_time.time() - t00:.2f}s')

        # Create SOFB
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)

    # ---------------- Measurement Methods ----------------

    def _do_measure(self):
        self.analysis = dict()
        tim_state = self._get_timing_state()
        if self.params.meas_bpms_noise:
            self.data['bpms_noise'] = self._do_measure_bpms_noise()

        if self._stopevt.is_set() or not self._meas_finished_ok:
            self._set_timing_state(tim_state)
            self._log('Stopped!')
            return

        if self.params.meas_rf_line:
            self.data['rf'] = self._do_measure_rf_line()

        if self._stopevt.is_set() or not self._meas_finished_ok:
            self._set_timing_state(tim_state)
            self._log('Stopped!')
            return

        self.data['magnets'] = self._do_measure_magnets()
        self._set_timing_state(tim_state)
        self._log('All measurements finished!!')

    def _do_measure_bpms_noise(self):
        self._meas_finished_ok = True
        elt = _time.time()
        par = self.params

        self._log('Measuring BPMs Noise:')

        t00 = _time.time()
        self._log('    Configuring BPMs...', end='')
        rf_freq = self.devices['rfgen'].frequency
        nr_points = par.corrs_excit_time + par.corrs_delay * 2
        nr_points *= self.bpms.get_sampling_frequency(rf_freq, acq_rate='FAcq')
        nr_points = int(_np.ceil(nr_points))
        ret = self._config_bpms(nr_points, rate='FAcq')
        if ret < 0:
            idx = -int(ret) - 1
            self._log(f'BPM {idx:d} did not finish last acquisition.')
        elif ret > 0:
            idx = int(ret) - 1
            self._log(f'BPM {idx:d} is not ready for acquisition.')
        if ret:
            self._meas_finished_ok = False
        self._config_timing()
        self._log(f'Done! ET: {_time.time() - t00:.2f}s')

        t00 = _time.time()
        self._log('    Sending Trigger signal...', end='')
        self.bpms.reset_mturn_initial_state()
        self.devices['evt_study'].cmd_external_trigger()
        self._log(f'Done! ET: {_time.time() - t00:.2f}s')

        # Wait BPMs PV to update with new data
        t00 = _time.time()
        self._log('    Waiting BPMs to update...', end='')
        ret = self.bpms.wait_update_mturn(timeout=par.timeout_bpms)
        if ret != 0:
            if ret > 0:
                tag = self.bpms.bpm_names[int(ret) - 1]
                pos = self.bpms.mturn_signals2acq[int((ret % 1) * 10) - 1]
                self._log(
                    'Problem: This BPM did not update: '
                    + tag
                    + ', signal '
                    + pos
                )
            elif ret == -1:
                self._log('Problem: Initial timestamps were not defined.')
            elif ret == -2:
                self._log('Problem: signals size changed.')
            self._meas_finished_ok = False
        self._log(f'Done! ET: {_time.time() - t00:.2f}s')

        # get data
        _time.sleep(0.5)
        data = self.get_general_data()
        data.update(self.get_bpms_data())
        data['ch_freqs'] = par.corrs_ch_freqs
        data['cv_freqs'] = par.corrs_cv_freqs

        elt -= _time.time()
        elt *= -1
        self._log(f'    Elapsed Time: {elt:.2f}s')
        return data

    def _do_measure_rf_line(self):
        self._meas_finished_ok = True
        elt = _time.time()
        par = self.params
        self._log('Measuring RF Line:')

        rate = 'FOFB' if par.rf_mode == par.RFModes.Phase else 'FAcq'

        t00 = _time.time()
        self._log('    Configuring BPMs...', end='')
        rf_freq = self.devices['rfgen'].frequency
        nr_points = par.rf_excit_time + par.rf_step_delay * 2
        nr_points *= self.bpms.get_sampling_frequency(rf_freq, acq_rate=rate)
        nr_points = int(_np.ceil(nr_points))
        ret = self._config_bpms(nr_points, rate=rate)
        if ret < 0:
            self._log(f'BPM {-ret - 1:d} did not finish last acquisition.')
        elif ret > 0:
            self._log(f'BPM {ret - 1:d} is not ready for acquisition.')
        if ret:
            self._meas_finished_ok = False
        self._config_timing()
        self._log(f'Done! ET: {_time.time() - t00:.2f}s')

        if par.rf_mode == par.RFModes.Phase:
            t00 = _time.time()
            self._log('    Turning RF Conditioning On...', end='')
            self._turn_on_llrf_conditioning()
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

        t00 = _time.time()
        self._log('    Sending Trigger signal...', end='')
        self.bpms.reset_mturn_initial_state()
        self.devices['evt_study'].cmd_external_trigger()
        self._log(f'    Done! ET: {_time.time() - t00:.2f}s')

        if par.rf_mode == par.RFModes.Step:
            t00 = _time.time()
            self._log('    Sweep RF...', end='')
            thr = _Thread(target=self._sweep_rf, daemon=True)
            _time.sleep(par.rf_step_delay)
            thr.start()
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

        # Wait BPMs PV to update with new data
        t00 = _time.time()
        self._log('    Waiting BPMs to update...', end='')
        ret = self.bpms.wait_update_mturn(timeout=par.timeout_bpms)
        if ret != 0:
            if ret > 0:
                tag = self.bpms.bpm_names[int(ret) - 1]
                pos = self.bpms.mturn_signals2acq[int((ret % 1) * 10) - 1]
                self._log(
                    'Problem: This BPM did not update: '
                    + tag
                    + ', signal '
                    + pos
                )
            elif ret == -1:
                self._log('Problem: Initial timestamps were not defined.')
            elif ret == -2:
                self._log('Problem: signals size changed.')
            self._meas_finished_ok = False
        self._log(f'Done! ET: {_time.time() - t00:.2f}s')

        data = dict()
        llrf = self.devices['llrfa']
        data['llrfa_cond_voltage_min'] = llrf.voltage_refmin_sp
        data['llrfa_cond_phase_min'] = llrf.phase_refmin_sp
        data['llrfa_cond_voltage'] = llrf.voltage_sp
        data['llrfa_cond_phase'] = llrf.phase_sp
        data['llrfa_cond_duty_cycle'] = llrf.conditioning_duty_cycle
        data['llrfa_cond_state'] = llrf.conditioning_state
        llrf = self.devices['llrfb']
        data['llrfb_cond_voltage_min'] = llrf.voltage_refmin_sp
        data['llrfb_cond_phase_min'] = llrf.phase_refmin_sp
        data['llrfb_cond_voltage'] = llrf.voltage_sp
        data['llrfb_cond_phase'] = llrf.phase_sp
        data['llrfb_cond_duty_cycle'] = llrf.conditioning_duty_cycle
        data['llrfb_cond_state'] = llrf.conditioning_state

        if par.rf_mode == par.RFModes.Phase:
            t00 = _time.time()
            self._log('    Turning RF Conditioning Off...', end='')
            self._turn_off_llrf_conditioning()
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

        # get data
        _time.sleep(0.5)
        data.update(self.get_general_data())
        data.update(self.get_bpms_data())
        data['mode'] = par.rf_mode
        data['step_kick'] = par.rf_step_kick
        data['excit_time'] = par.rf_excit_time
        data['step_delay'] = par.rf_step_delay
        data['mom_compac'] = _asparams.SI_MOM_COMPACT
        data['llrf_cond_freq'] = (
            data['rf_frequency'] * self.RF_CONDITIONING_FREQ
        )

        elt -= _time.time()
        elt *= -1
        self._log(f'    Elapsed Time: {elt:.2f}s')
        return data

    def _sweep_rf(self):
        kick = self.params.rf_step_kick
        excit_time = self.params.rf_excit_time
        rfgen = self.devices['rfgen']
        freq0 = rfgen.frequency
        rfgen.frequency = freq0 - kick
        _time.sleep(excit_time / 2)
        rfgen.frequency = freq0 + kick
        _time.sleep(excit_time / 2)
        rfgen.frequency = freq0

    def _turn_on_llrf_conditioning(self):
        stg_llrf = 'llrf' + self.params.rf_llrf2use.lower()
        llrf = self.devices[stg_llrf]
        llrf.voltage_refmin = llrf.voltage_sp
        llrf.phase_refmin = llrf.phase_sp - self.params.rf_phase_amp
        llrf.set_duty_cycle(self.RF_CONDITIONING_DUTY)
        llrf.cmd_turn_on_conditioning()

    def _turn_off_llrf_conditioning(self):
        stg_llrf = 'llrf' + self.params.rf_llrf2use.lower()
        llrf = self.devices[stg_llrf]
        llrf.cmd_turn_off_conditioning()
        llrf.voltage_refmin = self.RF_CONDITIONING_VOLTAGEMIN
        llrf.phase_refmin = llrf.phase_sp

    def _do_measure_magnets(self):
        self._meas_finished_ok = True
        elt0 = _time.time()
        data_mags = []

        self._log('Measuring Magnets:')

        # Shift correctors so first corrector is 01M1
        ch_names = self.sofb_data.ch_names
        cv_names = self.sofb_data.cv_names
        chs_shifted = self._shift_list(ch_names, 1)
        cvs_shifted = self._shift_list(cv_names, 1)

        tout_cor = self.params.timeout_correctors
        excit_time = self.params.corrs_excit_time
        freqh = self.params.corrs_ch_freqs
        freqv = self.params.corrs_cv_freqs

        rf_freq = self.devices['rfgen'].frequency
        nr_points = excit_time + self.params.corrs_delay * 2
        nr_points *= self.bpms.get_sampling_frequency(rf_freq, acq_rate='FAcq')
        nr_points = int(_np.ceil(nr_points))

        ch2meas = self.params.corrs_ch2meas
        cv2meas = self.params.corrs_cv2meas
        if isinstance(ch2meas, str) and ch2meas.startswith('all'):
            ch2meas = chs_shifted
        if isinstance(cv2meas, str) and cv2meas.startswith('all'):
            cv2meas = cvs_shifted

        # set operation mode to slowref
        if not self._change_corrs_opmode(
            'slowref', ch2meas + cv2meas, timeout=tout_cor
        ):
            self._log('Problem: Correctors not in SlowRef mode.')
            self._meas_finished_ok = False
            return data_mags

        ch_kicks = _np.full(len(ch2meas), self.params.corrs_ch_kick)
        cv_kicks = _np.full(len(cv2meas), self.params.corrs_cv_kick)
        if self.params.corrs_norm_kicks:
            orm = self.get_ref_respmat()

            ch_idx = _np.array([ch_names.index(n) for n in ch2meas])
            cv_idx = _np.array([cv_names.index(n) for n in cv2meas])
            cv_idx += len(chs_shifted)
            ch_kicks = self.params.corrs_dorb1ch / orm[:160, ch_idx].std(
                axis=0
            )
            cv_kicks = self.params.corrs_dorb1cv / orm[160:, cv_idx].std(
                axis=0
            )

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
            self._log(f'  Acquisition {itr + 1:02d}/{nacqs:02d}')

            chs_slc, cvs_slc = [], []
            ch_kick, cv_kick = [], []
            freqhe, freqve = [], []
            off = itr * nr1acq
            for run in range(nr1acq):
                slch = slice((off + run) * nfh, (off + run + 1) * nfh)
                slcv = slice((off + run) * nfv, (off + run + 1) * nfv)
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
            self._log('    Configuring BPMs and timing...', end='')
            ret = self._config_bpms(nr_points * len(chs_slc), rate='FAcq')
            if ret < 0:
                self._log(f'BPM {-ret - 1:d} did not finish last acquisition.')
            elif ret > 0:
                self._log(f'BPM {ret - 1:d} is not ready for acquisition.')
            if ret:
                self._meas_finished_ok = False
                break
            self._config_timing(
                self.params.corrs_delay, chs_slc, cvs_slc, nr_points=nr_points
            )
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

            # configure correctors
            t00 = _time.time()
            chs_f = _red(_opr.add, chs_slc)
            cvs_f = _red(_opr.add, cvs_slc)
            self._log('    Configuring Correctors...', end='')
            self._config_correctors(chs_f, ch_kick, freqhe, excit_time)
            self._config_correctors(cvs_f, cv_kick, freqve, excit_time)
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

            # set operation mode to cycle
            t00 = _time.time()
            self._log('    Changing Correctors to Cycle...', end='')
            if not self._change_corrs_opmode('cycle', chs_f + cvs_f, tout_cor):
                self._log('Problem: Correctors not in Cycle mode.')
                self._meas_finished_ok = False
                break
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

            # send event through timing system to start acquisitions
            t00 = _time.time()
            self._log('    Sending Timing signal...', end='')
            self.bpms.reset_mturn_initial_state()
            self.devices['evt_study'].cmd_external_trigger()
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

            # Wait BPMs PV to update with new data
            t00 = _time.time()
            self._log('    Waiting BPMs to update...', end='')
            ret = self.bpms.wait_update_mturn(timeout=self.params.timeout_bpms)
            if ret != 0:
                if ret > 0:
                    tag = self.bpms.bpm_names[int(ret) - 1]
                    pos = self.bpms.mturn_signals2acq[int((ret % 1) * 10) - 1]
                    self._log(
                        'Problem: This BPM did not update: '
                        + tag
                        + ', signal '
                        + pos
                    )
                elif ret == -1:
                    self._log('Problem: Initial timestamps were not defined.')
                elif ret == -2:
                    self._log('Problem: signals size changed.')
                self._meas_finished_ok = False
                break
            _time.sleep(0.5)
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

            # get data for each sector separately:
            data = self.get_general_data()
            data.update(self.get_bpms_data())
            orbx = data.pop('orbx')
            orby = data.pop('orby')
            for i, (chs, cvs) in enumerate(zip(chs_slc, cvs_slc)):
                datas = _dcopy(data)
                datas['orbx'] = orbx[i * nr_points : (i + 1) * nr_points]
                datas['orby'] = orby[i * nr_points : (i + 1) * nr_points]
                datas.update(self.get_magnets_data(chs_used=chs, cvs_used=cvs))
                data_mags.append(datas)

            # set operation mode to slowref
            t00 = _time.time()
            self._log('    Changing Correctors to SlowRef...', end='')
            if not self._wait_cycle_to_finish(chs_f + cvs_f, timeout=tout_cor):
                self._log('Problem: Cycle still not finished.')
                break
            if not self._change_corrs_opmode(
                'slowref', chs_f + cvs_f, tout_cor
            ):
                self._log('Problem: Correctors not in SlowRef mode.')
                break
            if self.params.correct_orbit_between_acqs:
                self.devices['sofb'].correct_orbit_manually(
                    nr_iters=5, residue=1
                )
            self._log(f'Done! ET: {_time.time() - t00:.2f}s')

            elt -= _time.time()
            elt *= -1
            self._log(f'  Elapsed Time: {elt:.2f}s')
            if self._stopevt.is_set():
                self._log('Stopping...')
                break

        # set operation mode to slowref
        if not self._change_corrs_opmode(
            'slowref', ch2meas + cv2meas, timeout=tout_cor
        ):
            self._log('Problem: Correctors still not in SlowRef mode.')
            return data_mags

        elt0 -= _time.time()
        elt0 *= -1
        self._log(f'Finished!!  ET: {elt0 / 60:.2f}min')
        return data_mags

    # ---------------- Data Processing methods ----------------

    def _process_rf_step(self, data, transition_length=10):
        t0_ = _time.time()
        self._log('Processing RF Step...', end='')
        fsamp = data['sampling_frequency']
        fswitch = data['switching_frequency']
        sw_mode = data['switching_mode']
        dtim = 1 / fsamp

        anly = dict()
        anly['sampling_frequency'] = fsamp
        anly['dtim'] = dtim
        anly['fswitch'] = fswitch
        anly['sw_mode'] = sw_mode
        anly['timestamp'] = data['timestamp']
        anly['stored_current'] = data['stored_current']
        anly['tunex'] = data['tunex']
        anly['tuney'] = data['tuney']
        anly['rf_frequency'] = data['rf_frequency']
        anly['acq_rate'] = data['acq_rate']
        anly['nrsamples'] = data['nrsamples_pre'] + data['nrsamples_post']

        orbx = data['orbx'].copy()
        orby = data['orby'].copy()
        orbx -= orbx.mean(axis=0)
        orby -= orby.mean(axis=0)

        tim = _np.arange(orbx.shape[0]) * dtim
        anly['time'] = tim
        kick = data['step_kick']

        etax_avg = 0.033e6  # average dispersion function, in [um]
        rf_freq = data['rf_frequency']
        mom_compac = data.get('mom_compac', _asparams.SI_MOM_COMPACT)
        amp = kick * etax_avg / mom_compac / rf_freq
        avg_orbx_smoothed = _savgol_filter(
            orbx.mean(axis=1), window_length=5, polyorder=2
        )  # TODO: use only dispersive BPMs for averaging
        peaks_neg = _find_peaks(avg_orbx_smoothed, amp / 2)[0]
        peaks_pos = _find_peaks(-avg_orbx_smoothed, amp / 2)[0]

        idx1 = peaks_neg[0]
        idx2 = peaks_neg[-1]
        idx3 = peaks_pos[0]
        idx4 = peaks_pos[-1]

        anly['idx1'] = idx1
        anly['idx2'] = idx2
        anly['idx3'] = idx3
        anly['idx4'] = idx4
        anly['amp'] = amp
        anly['transition_length'] = transition_length

        sec1_ini = idx1 + transition_length
        sec1_fin = idx2 - transition_length
        sec2_ini = idx3 + transition_length
        sec2_fin = idx4 - transition_length
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
        anly['mat_colsx'] = (orbx_pos - orbx_neg) / kick / 2
        anly['mat_colsy'] = (orby_pos - orby_neg) / kick / 2

        self._log(f'Done! ET: {_time.time() - t0_:2f}s')
        return anly

    def _process_rf_phase(self, data, window=10, central_freq=None):
        anly = dict()

        t0_ = _time.time()
        self._log('Processing RF Phase...', end='')

        fsamp = data['sampling_frequency']
        fswitch = data['switching_frequency']
        sw_mode = data['switching_mode']
        rf_freq = data['rf_frequency']
        dtim = 1 / fsamp

        anly = dict()
        anly['sampling_frequency'] = fsamp
        anly['dtim'] = dtim
        anly['fswitch'] = fswitch
        anly['sw_mode'] = sw_mode
        anly['timestamp'] = data['timestamp']
        anly['stored_current'] = data['stored_current']
        anly['tunex'] = data['tunex']
        anly['tuney'] = data['tuney']
        anly['rf_frequency'] = data['rf_frequency']
        anly['acq_rate'] = data['acq_rate']
        anly['nrsamples'] = data['nrsamples_pre'] + data['nrsamples_post']

        orbx = data['orbx'].copy()
        orby = data['orby'].copy()
        orbx -= orbx.mean(axis=0)
        orby -= orby.mean(axis=0)
        if fsamp / fswitch > 1 and sw_mode == 'switching':
            orbx = _AcqBPMsSignals.filter_switching_cycles(
                orbx, fsamp, freq_switching=fswitch
            )
            orby = _AcqBPMsSignals.filter_switching_cycles(
                orby, fsamp, freq_switching=fswitch
            )

        f_cond = rf_freq * self.RF_CONDITIONING_FREQ

        # Find peak with maximum amplitude to filter data
        if central_freq is None:
            # Possible range to find peak
            harm_min = int(1700 / f_cond)
            harm_max = int(2300 / f_cond)
            freqs = _np.arange(harm_min, harm_max) * f_cond
            freqx = _np.fft.rfftfreq(orbx.shape[0], d=dtim)

            # find indices of data frequencies close to excited frequencies
            idcs = _np.searchsorted(freqx, freqs)
            idsm1 = idcs - 1
            cond = _np.abs(freqx[idcs] - freqs) < _np.abs(freqx[idsm1] - freqs)
            idcs_min = _np.where(cond, idcs, idsm1)

            dftx = _np.abs(_np.fft.rfft(orbx, axis=0))[idcs_min]
            dfty = _np.abs(_np.fft.rfft(orby, axis=0))[idcs_min]
            amaxx = dftx.argmax(axis=0)
            amaxy = dfty.argmax(axis=0)
            vals, cnts = _np.unique(_np.r_[amaxx, amaxy], return_counts=True)
            cnts_argmax = cnts.argmax()
            central_freq = freqx[idcs_min][vals[cnts_argmax]]
            anly['central_freq_count'] = cnts[cnts_argmax]

        anly['central_freq_guess'] = central_freq
        central_freq = int(round(central_freq / f_cond)) * f_cond
        fmin = central_freq - window / 2
        fmax = central_freq + window / 2
        anly['central_freq'] = central_freq
        anly['window'] = window
        anly['freq_min'] = fmin
        anly['freq_max'] = fmax

        orbx = _AcqBPMsSignals.filter_data_frequencies(
            orbx, fmin=fmin, fmax=fmax, fsampling=fsamp
        )
        orby = _AcqBPMsSignals.filter_data_frequencies(
            orby, fmin=fmin, fmax=fmax, fsampling=fsamp
        )

        ref_respmat = self.get_ref_respmat()
        etaxy = ref_respmat[:, -1]
        nrbpms = etaxy.size // 2
        etax, etay = etaxy[:nrbpms], etaxy[nrbpms:]
        eta_meas = _OrbitAnalysis.calculate_eta_meas(orbx, orby, etax, etay)
        anly['mat_colsx'] = eta_meas[:nrbpms]
        anly['mat_colsy'] = eta_meas[nrbpms:]

        self._log(f'Done! ET: {_time.time() - t0_:2f}s')
        return anly

    def _process_bpms_noise(self, data):
        sofb = self.sofb_data

        t0_ = _time.time()
        self._log('Processing BPMs Noise...', end='')

        ch_freqs = data['ch_freqs']
        cv_freqs = data['cv_freqs']

        fsamp = data['sampling_frequency']
        dtim = 1 / fsamp
        freqs0 = _np.r_[ch_freqs, cv_freqs]

        anly = dict()
        anly['sampling_frequency'] = fsamp
        anly['dtim'] = dtim
        anly['freqs0'] = freqs0
        anly['timestamp'] = data['timestamp']
        anly['stored_current'] = data['stored_current']
        anly['tunex'] = data['tunex']
        anly['tuney'] = data['tuney']
        anly['rf_frequency'] = data['rf_frequency']
        anly['acq_rate'] = data['acq_rate']
        anly['nrsamples'] = data['nrsamples_pre'] + data['nrsamples_post']

        orbx = data['orbx'].copy()
        orby = data['orby'].copy()
        orbx -= orbx.mean(axis=0)
        orby -= orby.mean(axis=0)

        tim = _np.arange(orbx.shape[0]) * dtim
        anly['time'] = tim

        cosx, sinx, pinv = self.fit_fourier_components(orbx, freqs0, dtim)
        cosy, siny, _ = self.fit_fourier_components(
            orby, freqs0, dtim, pinv=pinv
        )
        ampx, _ = self.fit_calc_amp_and_phase(cosx, sinx)
        ampy, _ = self.fit_calc_amp_and_phase(cosy, siny)
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

        match = _np.zeros((2 * sofb.nr_bpms, sch))
        matcv = _np.zeros((2 * sofb.nr_bpms, scv))
        match[: sofb.nr_bpms] = ampx_ch
        match[sofb.nr_bpms :] = ampy_ch
        matcv[: sofb.nr_bpms] = ampx_cv
        matcv[sofb.nr_bpms :] = ampy_cv

        match = _np.roll(_np.tile(match, (sofb.nr_ch // sch) + 1), -1)
        matcv = _np.roll(_np.tile(matcv, (sofb.nr_cv // scv) + 1), -1)
        mat = _np.zeros((2 * sofb.nr_bpms, sofb.nr_corrs))
        mat[:, : sofb.nr_ch] = match[:, : sofb.nr_ch]
        mat[:, sofb.nr_ch : sofb.nr_chcv] = matcv[:, : sofb.nr_cv]

        ampx_rf = orbx.std(axis=0)
        ampy_rf = orby.std(axis=0)
        anly['noisex_rf'] = ampx_rf
        anly['noisey_rf'] = ampy_rf
        mat[: sofb.nr_bpms, -1] = ampx_rf
        mat[sofb.nr_bpms :, -1] = ampy_rf

        anly['bpm_variation'] = mat

        self._log(f'Done! ET: {_time.time() - t0_:2f}s')
        return anly

    def _process_magnets(self, magnets_data, idx_ini=None):
        """."""
        sofb = self.sofb_data
        corr2idx = {name: i for i, name in enumerate(sofb.ch_names)}
        corr2idx.update({
            name: 120 + i for i, name in enumerate(sofb.cv_names)
        })

        self._log('Processing Magnets Data:')
        analysis = []
        ref_respmat = self.get_ref_respmat()
        args = corr2idx, ref_respmat
        pinv1 = pinv2 = pinv3 = None
        for i, data in enumerate(magnets_data):
            self._log(
                f'  Acquisition {i + 1:02d}/{len(magnets_data):02d} ', end=''
            )
            t0_ = _time.time()
            if idx_ini is not None:
                anl, pinv3, _ = self._process_single_excit(
                    data, idx_ini, *args, naff=False, pinv=pinv3
                )
                analysis.append(anl)
                self._log(f'ET: {_time.time() - t0_:1f}s')
                continue

            # If idx_ini is not given, find adequate initial index for fitting.
            # Since the excitation is a sine wave, we want the phases to be
            # all close to 0 or pi. So this algorithm finds the index that
            # brings all phases as close as possible to these values.
            fsamp = data['sampling_frequency']
            delay = 4 / data['rf_frequency']
            dly_raw = data['corrs_trig_delta_delay_raw']
            delay *= dly_raw[dly_raw > 0].min()
            idx1 = int(delay * fsamp)
            idx2 = idx1 + 5
            _, pinv1, phs1 = self._process_single_excit(
                data, idx1, *args, pinv=pinv1
            )
            _, pinv2, phs2 = self._process_single_excit(
                data, idx2, *args, pinv=pinv2
            )
            coef = _np.polynomial.polynomial.polyfit(
                [idx1, idx2], _np.vstack([phs1, phs2]), deg=1
            )
            idx_ini = _np.round(-coef[0] / coef[1]).astype(int)

            # Evaluate data with optimal idx_ini:
            anl, *_ = self._process_single_excit(
                data, idx_ini, *args, naff=False, pinv=None
            )
            analysis.append(anl)
            self._log(f'ET: {_time.time() - t0_:1f}s')

        self._log('Done processing Magnets Data!')
        return analysis

    def _process_single_excit(
        self, data, idx_ini, corr2idx, ref_respmat, naff=False, pinv=None
    ):
        fsamp = data['sampling_frequency']
        dtim = 1 / fsamp
        ch_freqs = _np.array(data['ch_frequency'])
        cv_freqs = _np.array(data['cv_frequency'])
        freqs0 = _np.r_[ch_freqs, cv_freqs]
        nr_bpms = self.sofb_data.nr_bpms
        nch = ch_freqs.size
        ncv = cv_freqs.size

        anly = dict()
        anly['sampling_frequency'] = fsamp
        anly['dtim'] = dtim
        anly['freqs0'] = freqs0
        anly['timestamp'] = data['timestamp']
        anly['stored_current'] = data['stored_current']
        anly['tunex'] = data['tunex']
        anly['tuney'] = data['tuney']
        anly['rf_frequency'] = data['rf_frequency']
        anly['acq_rate'] = data['acq_rate']
        anly['corr_names'] = data['ch_names'] + data['cv_names']
        anly['nrsamples'] = data['nrsamples_pre'] + data['nrsamples_post']

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

        anly['idx_ini'] = idx_ini

        if naff:
            cosx, sinx = self.fit_fourier_components_naff(orbx, freqs0, dtim)
            cosy, siny = self.fit_fourier_components_naff(orby, freqs0, dtim)
        else:
            cosx, sinx, pinv = self.fit_fourier_components(
                orbx, freqs0, dtim, num_cycles, idx_ini, pinv
            )
            cosy, siny, pinv = self.fit_fourier_components(
                orby, freqs0, dtim, num_cycles, idx_ini, pinv
            )
        ampx, phasex = self.fit_calc_amp_and_phase(cosx, sinx)
        ampy, phasey = self.fit_calc_amp_and_phase(cosy, siny)

        # Compensate BPMs mis-synchronization:
        phsxch = phasex[:nch, :] / _np.pi
        phsycv = phasey[nch:, :] / _np.pi
        # Wrap phases close to -1 and 1 around 0:
        phsxch = (phsxch + 0.5) % 1 - 0.5
        phsycv = (phsycv + 0.5) % 1 - 0.5
        # Find average phase over all frequencies for each BPM
        meanphs = phsxch.mean(axis=0)
        meanphs += phsycv.mean(axis=0)
        meanphs /= 2
        # Subtract average phase and wrap result to [-pi, pi]
        phasex += _np.pi * (1 - meanphs)
        phasey += _np.pi * (1 - meanphs)
        phasex %= 2 * _np.pi
        phasey %= 2 * _np.pi
        phasex -= _np.pi
        phasey -= _np.pi
        anly['phase_mean'] = meanphs

        # Determine signal of elements:
        signx = _np.ones(ampx.shape)
        signx[_np.abs(phasex) > (_np.pi / 2)] = -1
        signy = _np.ones(ampy.shape)
        signy[_np.abs(phasey) > (_np.pi / 2)] = -1
        mat_colsx = (signx * ampx / kicks[:, None]).T
        mat_colsy = (signy * ampy / kicks[:, None]).T

        # Re-scale columns so that diagonal terms STD match reference matrix:
        xscalefactor = ref_respmat[:nr_bpms, ch_idcs].std(axis=0)
        yscalefactor = ref_respmat[nr_bpms:, cv_idcs].std(axis=0)
        xscalefactor /= mat_colsx[:, :nch].std(axis=0)
        yscalefactor /= mat_colsy[:, nch:].std(axis=0)
        mat_colsx[:, :nch] *= xscalefactor
        mat_colsx[:, nch:] *= yscalefactor
        mat_colsy[:, :nch] *= xscalefactor
        mat_colsy[:, nch:] *= yscalefactor

        anly['ampx'] = ampx.T
        anly['ampy'] = ampy.T
        anly['phasex'] = phasex.T
        anly['phasey'] = phasey.T
        anly['cosx'] = cosx.T
        anly['cosy'] = cosy.T
        anly['sinx'] = sinx.T
        anly['siny'] = siny.T
        anly['signx'] = signx.T
        anly['signy'] = signy.T
        anly['mat_colsx'] = mat_colsx
        anly['mat_colsy'] = mat_colsy
        anly['mat_colsx_scale'] = xscalefactor
        anly['mat_colsy_scale'] = yscalefactor

        # Return average phase along BPMs for each frequency so that caller
        # can use this info to optmize idx_ini.
        phs = _np.hstack([phsxch.T, phsycv.T])
        phs = phs.reshape(-1, nch + ncv, 2).mean(axis=-1).mean(axis=0)
        return anly, pinv, phs

    # ----------------- BPMs related methods -----------------------

    def _config_bpms(self, nr_points, rate='FAcq'):
        return self.bpms.config_mturn_acquisition(
            acq_rate=rate,
            nr_points_before=0,
            nr_points_after=nr_points,
            repeat=False,
            external=True,
        )

    # ----------------- Timing related methods -----------------------

    def _get_timing_state(self):
        trigbpm = self.devices['trigbpms']
        trigcorr = self.devices['trigcorrs']
        evt_study = self.devices['evt_study']
        return {
            'trigbpm_source': trigbpm.source,
            'trigbpm_nr_pulses': trigbpm.nr_pulses,
            'trigbpm_delay_raw': trigbpm.delay_raw,
            'trigcorr_source': trigcorr.source,
            'trigcorr_nr_pulses': trigcorr.nr_pulses,
            'trigcorr_delay_raw': trigcorr.delay_raw,
            'trigcorr_delta_delay_raw': trigcorr.delta_delay_raw,
            'evt_study_mode': evt_study.mode,
            'evt_study_delay_raw': evt_study.delay_raw,
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
        if 'trigbpm_delay_raw' in state:
            trigbpm.delay_raw = state['trigbpm_delay_raw']
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
        if 'evt_study_delay_raw' in state:
            evt_study.delay_raw = state['evt_study_delay_raw']
        _time.sleep(0.1)
        evg.cmd_update_events()

    def _config_timing(self, cm_dly=0, chs=None, cvs=None, nr_points=None):
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
        state['trigbpm_delay_raw'] = 0.0

        state['evt_study_mode'] = 'External'
        state['evt_study_delay_raw'] = 0

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
        runs_delta_dly = _np.arange(nr_runs, dtype=float)
        runs_delta_dly *= nr_points / fsamp
        runs_delta_dlyr = _np.round(runs_delta_dly * ftim)

        # get low level trigger names to be configured in each run of the
        # acquisition:
        ll_trigs = []
        for ch, cv in zip(chs, cvs):
            llt = set()
            for c in ch + cv:
                trig = _LLTime.get_trigger_name(c + ':BCKPLN')
                llt.add(trig)
            ll_trigs.append(llt)

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
            conv = self.devices[cmn + ':StrengthConv'].conv_strength_2_current
            cmo.cycle_type = cmo.CYCLETYPE.Sine
            cmo.cycle_freq = freqs[i]
            cmo.cycle_ampl = conv(kicks[i])
            cmo.cycle_offset = cmo.currentref_mon
            cmo.cycle_theta_begin = 0
            cmo.cycle_theta_end = 0
            cmo.cycle_num_cycles = int(excit_time * freqs[i])
            # NOTE: There is a bug in the firmware of the power supplies
            # (apparently comparison >= should be replaced by > in line 353 of
            # the file siggen.c of the repository C28) that makes the endpoint
            # of the cycle not be equal to the starting point. So we need to
            # add a very small phase at the ending of the senoid to compensate
            # for this bug. The code bellow adds a phase compatible with a
            # small fraction (0.1) of the phase advance between two points of
            # the signal at the end of the cycling.
            fsamp = cmo['ParamPWMFreq-Cte']
            params = cmo.cycle_aux_param
            params[1] = freqs[i] / fsamp * 360
            params[1] *= 0.1
            cmo.cycle_aux_param = params

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

        for cmn in corr_names:
            dt_ = _time.time()
            cmo = self.devices[cmn]
            if not cmo._wait('OpMode-Sts', mode_sts, timeout=timeout):
                self._log('ERR:' + cmn + ' did not change to ' + mode)
                return False
            dt_ -= _time.time()
            timeout = max(timeout + dt_, 0)
            cmo.current = cmo.current
        return True

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
