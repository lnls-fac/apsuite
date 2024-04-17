"""."""

import datetime as _datetime
import time as _time

import matplotlib.pyplot as _mplt
import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
from siriuspy.devices import PowerSupplyPU, Trigger

from .meas_bpms_signals import AcqBPMsSignals as _AcqBPMsSignals, \
    AcqBPMsSignalsParams as _AcqBPMsSignalsParams


class TbTDataParams(_AcqBPMsSignalsParams):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.signals2acq = "XYS"
        self.acq_rate = "TbT"
        self.nrpoints_before = 100
        self.nrpoints_after = 2000

        self.timing_event = "Linac"
        self.event_mode = "Injection"

        self._pingers2kick = None  # 'H', 'V' or 'HV'
        self.hkick = None  # [urad]
        self.vkick = None  # [urad]
        self.trigpingh_delay = None
        self.trigpingv_delay = None
        pingers2kick = str(self._pingers2kick).lower()
        self.trigpingh_nrpulses = 1 if "h" in pingers2kick else 0
        self.trigpingv_nrpulses = 1 if "v" in pingers2kick else 0

    def __str__(self):
        """."""
        stg = super().__str__()
        ftmp = "{0:26s} = {1:9.6f}  {2:s}\n".format
        dtmp = "{0:26s} = {1:9d}  {2:s}\n".format
        stmp = "{0:26s} = {1:9}  {2:s}\n".format
        stg += stmp("pingers2kick", self.pingers2kick, "")
        if self.hkick is None:
            stg += stmp("hkick", "same", "(current value will not be changed)")
        else:
            stg += ftmp("hkick", self.hkick, "[urad]")
        dly = self.trigpingh_delay
        if dly is None:
            stg += stmp(
                "trigpingh_delay",
                "same",
                "(current value will not be changed)",
            )
        else:
            stg += ftmp("trigpingh_delay", dly, "[us]")
        stg += dtmp("trigpingh_nrpulses", self.trigpingh_nrpulses, "")
        if self.vkick is None:
            stg += stmp("vkick", "same", "(current value will not be changed)")
        else:
            stg += ftmp("vkick", self.vkick, "[urad]")
        dly = self.trigpingv_delay
        if dly is None:
            stg += stmp(
                "trigpingv_delay",
                "same",
                "(current value will not be changed)",
            )
        else:
            stg += ftmp("trigpingv_delay", dly, "[us]")
        stg += dtmp("trigpingv_nrpulses", self.trigpingv_nrpulses, "")
        return stg

    @property
    def pingers2kick(self):
        """."""
        return self._pingers2kick

    @pingers2kick.setter
    def pingers2kick(self, value):
        if (value is not None) and (value.lower() not in "hv"):
            raise ValueError('Invalid keyword. Set "None", "H", "V" or "HV"')
        else:
            self._pingers2kick = value


class MeasureTbTData(_AcqBPMsSignals):
    """."""

    PINGERH_TRIGGER = "SI-01SA:TI-InjDpKckr"
    PINGERV_TRIGGER = "SI-19C4:TI-PingV"

    def __init__(self, isonline=False):
        """."""
        super().__init__(isonline=isonline, ispost_mortem=False)
        self.params = TbTDataParams()

    def create_devices(self):
        """."""
        super().create_devices()
        for pinger in self.pingers2kick:
            if pinger.lower == "h":
                self.devices["pingh"] = PowerSupplyPU(
                    PowerSupplyPU.DEVICES.SI_INJ_DPKCKR
                )
            self.devices["trigpingh"] = Trigger(self.PINGERH_TRIGGER)
            if pinger.lower() == "v":
                self.devices["pinghv"] = PowerSupplyPU(
                    PowerSupplyPU.DEVICES.SI_PING_V
                )
                self.devices["trigpingv"] = Trigger(self.PINGERV_TRIGGER)

    def get_timing_state(self):
        """."""
        state = super().get_timing_state()
        trigpingh = self.devices["trigpingh"]
        state["trigpingh_source"] = trigpingh.source
        state["trigpingh_nrpulses"] = trigpingh.nr_pulses
        state["trigpingh_delay"] = trigpingh.delay
        trigpingv = self.devices["trigpingv"]
        state["trigpingv_source"] = trigpingv.source
        state["trigpingv_nrpulses"] = trigpingv.nr_pulses
        state["trigpingv_delay"] = trigpingv.delay
        return state

    def prepare_timing(self, state=None):
        """."""
        super().prepare_timing(state)  # BPM trigger timing
        # magnets trigger timing below
        prms = self.params
        trigpingh = self.devices["trigpingh"]
        trigpingh.source = state.get("trigpingh_source", prms.timing_event)
        trigpingh.nr_pulses = state.get(
            "trigpingh_nrpulses", prms.trigpingh_nrpulses
        )
        dly = state.get("trigpingh_delay", prms.trigpingh_delay)
        if dly is not None:
            trigpingh.delay = dly

        trigpingv = self.devices["trigpingv"]
        trigpingv.source = state.get("trigpingv_source", prms.timing_event)
        trigpingh.nr_pulses = state.get(
            "trigpingv_nrpulses", prms.trigpingv_nrpulses
        )
        dly = state.get("trigpingv_delay", prms.trigpingv_delay)
        if dly is not None:
            trigpingv.delay = dly

    def get_magnets_strength(self):
        """."""
        return self.devices["pingh"].strength, self.devices["pingv"].strength

    def set_magnets_strength(self, hkick=None, vkick=None):
        """."""
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]
        if hkick is None:
            hkick = self.params.hkick / 1e3  # [urad] -> [mrad]
        pingh.strength = hkick
        if vkick is None:
            vkick = self.params.vkick / 1e3  # [urad] -> [mrad]
        pingv.strength = vkick
        _time.sleep(0.5)

    def do_measurement(self):
        """."""
        currinfo = self.devices["currinfo"]
        init_timing_state = self.get_timing_state()
        init_magnets_strength = self.get_magnets_strength()
        current_before = currinfo.current()
        self.prepare_timing()
        self.set_magnets_strength()  # gets strengths from params
        self.data["measurement_error"] = False  # error flag
        try:
            self.acquire_data()
        # BPMs signals + relevant info are acquired
        # such as timestamps tunes, stored current
        # rf frequency, acq rate, nr samples, etc.
        except Exception as e:
            print(f"An error occurred during acquisition: {e}")
            self.data["measurement_error"] = True
        self.recover_timing_state(init_timing_state)
        self.set_magnets_strength(init_magnets_strength)  # restore strengths
        self.data["current_before"] = current_before
        self.data["current_after"] = self.data.pop("stored_current")

    def get_default_fname(self):
        """."""
        hkick, vkick = self.params.hkick, self.params.vkick
        tm = self.data['timestamp']
        fmt = '%Y-%m-%d-%H-%M-%S'
        tmstp = _datetime.datetime.fromtimestamp(tm).strftime(fmt)
        stg = f'hkick_{int(round(hkick)):3d}_vkick_{int(round(vkick)):3d}'
        stg += f'_urad_{tmstp}'
        return stg


class TbTDataAnalysis(_AcqBPMsSignals):
    """."""

    def __init__(self, filename="", isonline=False):
        """Analysis of linear optics using Turn-by-turn data."""
        self.params = TbTDataParams()
        self.isonline = isonline
        self._ispost_mortem = False

        self._fname = filename
        self._trajx, self._trajy = None, None
        self._trajsum = None

        # load if fname, load method

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
    def trajy(self, val):
        self._trajy = val

    def linear_optics_analysis(self):
        """."""
        raise NotImplementedError

    def harmonic_analysis(self):
        """."""
        raise NotImplementedError

    def principal_components_analysis(self):
        """."""
        raise NotImplementedError

    def independent_component_analysis(self):
        """."""
        raise NotImplementedError

    def equilibrium_params_analysis(self):
        """."""
        raise NotImplementedError

    # plotting methods
    def plot_traj_spectrum():
        """."""
        raise NotImplementedError

    def _get_tune_guess(self, matrix):
        """."""
        matrix_dft, tune = self.calc_spectrum(matrix, fs=1, axis=0)
        tunes = tune[:, None] * _np.ones(matrix.shape[-1])[None, :]
        peak_idcs = _np.abs(matrix_dft).argmax(axis=0)
        tune_peaks = [tunes[idc, col] for col, idc in enumerate(peak_idcs)]
        return _np.mean(tune_peaks)  # tune guess

    def _get_amplitudes_and_phases_guess(self, matrix, tune):
        """Calculate initial amplitude & phase guesses for harmonic TbT model.

        Implements Eqs. (5.2)-(5.4) from Ref. [2]
        """
        N = matrix.shape[0]
        ilist = _np.arange(N)
        cos = _np.cos(2 * _np.pi * tune * ilist)
        sin = _np.sin(2 * _np.pi * tune * ilist)
        C, S = _np.dot(cos, matrix), _np.dot(sin, matrix)
        C *= 2 / N
        S *= 2 / N
        amplitudes = _np.sqrt(C**2 + S**2)
        phases = _np.unwrap(_np.arctan2(C, S))
        return amplitudes, phases

    def harmonic_tbt_model(self, ilist, amplitude, tune, phase):
        """Harmonic motion model for positions seen at a given BPM."""
        return amplitude * _np.cos(2 * _np.pi * tune * ilist + phase)

    def harmonic_tbt_model_vectorized(self, ilist, amplitude, tune, phase):
        """Harmonic motion model for positions seen at a given BPM."""
        model = amplitude[None, :]
        model *= _np.cos(2 * _np.pi * tune * ilist[:, None] + phase[None, :])
        return model

    def fit_harmonic_model(
        self, matrix, amp_guesses, tune_guess, phase_guesses
    ):
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
            p0 = [amp_guesses[bpm_idx], tune_guess, phase_guesses[bpm_idx]]
            popt, *_ = _curve_fit(
                f=self.harmonic_tbt_model, xdata=ilist, ydata=bpm_data, p0=p0
            )
            params[:, bpm_idx] = popt
        return params

    def calculate_betafunc_and_action(self, amplitudes, nominal_beta):
        """Calculates beta function and betatron action.

        As in Eq. (9) of Ref. [1]
        """
        action = _np.sum(amplitudes**4)
        action /= _np.sum(amplitudes**2 * nominal_beta)
        beta = amplitudes**2 / action
        return beta, action
