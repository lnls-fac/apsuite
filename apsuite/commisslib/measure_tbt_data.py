"""."""
import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
import matplotlib.pyplot as _mplt
import datetime as _datetime
from siriuspy.devices import PowerSupplyPU

from .meas_bpms_signals import AcqBPMsSignals as _AcqBPMsSignals, \
    AcqBPMsSignalsParams as _AcqBPMsSignalsParams

class TbTDataAnalysis(_AcqBPMsSignals):
    """."""

    def __init__(self, filename='', isonline=False):
        """Analysis of betatron motion and linear optics using Turn-by-turn
        data"""
        super().__init__(isonline=isonline, ispost_mortem=False)

    @property
    def fname(self):
        """."""
        return self._fname

    @fname.setter
    def fname(self, val):
        self._fname = val

    @property
    def trajx(self):
        """."""
        return self._trajx

    @trajx.setter
    def trajx(self, val):
        self._trajx = val

    @property
    def trajy(self):
        """."""
        return self._trajy

    @trajy.setter
    def trajx(self, val):
        self._trajy = val

    def linear_optics_analysis(self):
        raise NotImplementedError

    def harmonic_analysis(self):
        """."""
        raise NotImplementedError

    def principal_components_analysis(self):
        raise NotImplementedError

    def independent_component_analysis(self):
        raise NotImplementedError

    def equilibrium_params_analysis(self):
        raise NotImplementedError

    # plotting methods
    def plot_traj_spectrum():
        raise NotImplementedError


    def _get_tune_guess(self, matrix):
        """."""
        matrix_dft, tune = self.calc_spectrum(matrix, fs=1, axis=0)
        tunes = tune[:, None] * _np.ones(matrix.shape[-1])[None, :]
        peak_idcs = _np.abs(matrix_dft).argmax(axis=0)
        tune_peaks = [tunes[idc, col] for col, idc in enumerate(peak_idcs)]
        return _np.mean(tune_peaks)  # tune guess

    def _get_amplitudes_and_phases_guess(self, matrix, tune):
        """Calculate initial guesses for harmonic TbT model as
           Eqs. (5.2)-(5.4) of Ref. [2]"""
        N = matrix.shape[0]
        ilist = _np.arange(N)
        cos = _np.cos(2 * _np.pi * tune * ilist)
        sin = _np.sin(2 * _np.pi * tune * ilist)
        C, S = _np.dot(cos, matrix), _np.dot(sin, matrix)
        C *= 2/N
        S *= 2/N
        amplitudes = _np.sqrt(C**2 + S**2)
        phases = _np.unwrap(_np.arctan2(C, S))
        return amplitudes, phases

    def harmonic_tbt_model(self, ilist, amplitude, tune, phase):
        """Harmonic motion model for positions seen at a given BPM"""
        return amplitude * _np.cos(2 * _np.pi * tune * ilist + phase)

    def harmonic_tbt_model_vectorized(self, ilist, amplitude, tune, phase):
        """Harmonic motion model for positions seen at a given BPM"""
        return amplitude[None, :] * _np.cos(2 * _np.pi * tune * ilist[:, None] + phase[None, :])

    def fit_harmonic_model(self, matrix, amp_guesses,
                           tune_guess, phase_guesses):
        """Fits harmonic TbT model to data.

        Args:
            matrix (N,M)-array: data matrix containing N turns and BPMs
            amp_guesses M-array: amplitude guess for each BPM time-series
            tune_guess (float): betatron tune guess
            phase_guesses M-array: phase guess for each BPM

        Returns:
            params (3, M)-array: fitted ampliude, tune and phase for each BPM.
                                each column corresponds to a BPM, each row
                                corresponds to amplitudes, tune and phase
        """
        ilist = _np.arange(matrix.shape[0])
        params = _np.zeros((3, matrix.shape[-1]))
        for bpm_idx, bpm_data in enumerate(matrix.T):
            p0=[amp_guesses[bpm_idx],tune_guess,phase_guesses[bpm_idx]]
            popt, *_ = _curve_fit(f=self.harmonic_tbt_model,
                                  xdata=ilist,ydata=bpm_data,
                                  p0=p0)
            params[:, bpm_idx] = popt
        return params

    def calculate_betafunc_and_action(self, amplitudes, nominal_beta):
        """Calculates beta function and betatron action as in Eq. (9)
            of Ref. [1]"""
        action = _np.sum(amplitudes**4)
        action /= _np.sum(amplitudes**2 * nominal_beta)
        beta = amplitudes**2/action
        return beta, action

class TbTDataParams(_AcqBPMsSignalsParams):
    """."""

    def __init__(self, hkick=None, vkick=None):
        super().__init__()
        self.nrpoints_before = 100
        self.nrpoints_after = 2000
        self.acq_rate = 'TbT'
        self.signals2acq = 'XYS'
        self._hkick = hkick
        self._vkick = vkick


    def __str__(self):
        stg = super().__str__()
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        stmp = '{0:26s} = {1:9}  {2:s}\n'.format
        if self.hkick is None:
            stg += stmp('hkick', 'same', '(current value will not be changed)')
        else:
            stg += ftmp('hkick', self.hkick, '[urad]')
        if self.hkick is None:
            stg += stmp('vkick', 'same', '(current value will not be changed)')
        else:
            stg += ftmp('vkick', self.vkick, '[urad]')
        return stg

    @property
    def hkick(self):
        """."""
        return self._hkick

    @hkick.setter
    def hkick(self, val):
        self._hkick = val

    @property
    def vkick(self):
        """."""
        return self._vkick

    @vkick.setter
    def vkick(self, val):
        self._vkick = val

class MeasureTbTData(_AcqBPMsSignals):
    """."""
    def __init__(self, filename='', isonline=False):
        """."""
        self.params = TbTDataParams()
        self.isonline = isonline
        self._fname = filename

    def create_devices(self):
        super().create_devices()
        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['pinghv'] = PowerSupplyPU(PowerSupplyPU.DEVICES.SI_PING_V)

    @property
    def fname(self):
        """."""
        return self._fname

    @fname.setter
    def fname(self, val):
        self._fname = val

    def get_magnets_state(self):
        """."""
        pingh, pingv = self.devices['pingh'], self.devices['pingv']
        hkick, vkick = pingh.strength, pingv.strength
        return hkick, vkick

    def recover_magnets_state(self, hkick, vkick):
        """."""
        self.set_magnets_state(hkick, vkick)

    def set_magnets_state(self, hkick, vkick):
        """."""
        pingh, pingv = self.devices['pingh'], self.devices['pingv']
        if hkick is not None:
            pingh.strength = hkick
        if vkick is not None:
            pingv.strength = vkick

    def prepare_magnets(self):
        """."""
        hkick, vkick = self.params.hkick, self.params.vkick
        hkick = hkick/1e6 if hkick is not None else None
        vkick = vkick/1e6 if vkick is not None else None
        self.set_magnets_state(hkick, vkick)

    def do_measurement(self):
        """."""
        currinfo = self.devices['currinfo']
        init_timing_state = self.get_timing_state()
        init_magnets_state = self.get_magnets_state()
        current_before = currinfo.current()
        self.prepare_timing()
        self.prepare_magnets()
        self.data['measurement_error'] = False  # error flag
        try:
            self.acquire_data()  # BPMs signals + relevant info are acquired
                                 # such as timestamps tunes, stored current
                                 # rf frequency, acq rate, nr samples, etc.
        except Exception as e:
            print(f'An error occurred during acquisition: {e}')
            self.data['measurement_error'] = True
        self.recover_timing_state(init_timing_state)
        self.recover_magnets_state(init_magnets_state)
        self.data['current_before'] = current_before
        self.data['current_after'] = self.data.pop('stored_current')
        self.data['trajx'] = self.data.pop('orbx')
        self.data['trajy'] = self.data.pop('orby')
