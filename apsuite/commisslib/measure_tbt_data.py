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

        self._pingers2kick = "none"  # 'H', 'V' or 'HV'
        self.hkick = None  # [urad]
        self.vkick = None  # [urad]
        self.trigpingh_delay = None
        self.trigpingv_delay = None
        self.magnets_timeout = 5.

    def __str__(self):
        """."""
        stg = "BPMs & timing params\n"
        stg += "\n"
        stg += super().__str__()
        ftmp = "{0:26s} = {1:9.6f}  {2:s}\n".format
        dtmp = "{0:26s} = {1:9d}  {2:s}\n".format
        stmp = "{0:26s} = {1:9}  {2:s}\n".format
        stg += "\n"
        stg += "Pingers params\n"
        stg += "\n"
        stg += stmp("pingers2kick", self.pingers2kick, "")
        stg += ftmp("magnets_timeout", self.magnets_timeout, "[s]")
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
        return stg

    @property
    def pingers2kick(self):
        """."""
        return self._pingers2kick

    @pingers2kick.setter
    def pingers2kick(self, val):
        self._pingers2kick = str(val).lower()


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
        self.devices["pingh"] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR
        )
        self.devices["trigpingh"] = Trigger(self.PINGERH_TRIGGER)

        self.devices["pinghv"] = PowerSupplyPU(PowerSupplyPU.DEVICES.SI_PING_V)
        self.devices["trigpingv"] = Trigger(self.PINGERV_TRIGGER)
        return

    def get_timing_state(self):
        """."""
        state = super().get_timing_state()
        trigpingh = self.devices["trigpingh"]
        state["trigpingh_state"] = trigpingh.state  # correct?
        state["trigpingh_source"] = trigpingh.source
        state["trigpingh_delay"] = trigpingh.delay
        trigpingv = self.devices["trigpingv"]
        state["trigpingv_state"] = trigpingv.state  # correct?
        state["trigpingv_source"] = trigpingv.source
        state["trigpingv_delay"] = trigpingv.delay
        return state

    def prepare_timing(self, state=None):
        """."""
        super().prepare_timing(state)  # BPM trigger timing
        # magnets trigger timing below
        prms = self.params
        trigpingh = self.devices["trigpingh"]
        pingers2kick = prms.pingers2kick
        if ("h" in pingers2kick) or (state.get("trigpingh_state", 0) == 1):
            trigpingh.cmd_enable(timeout=prms.timeout)
            trigpingh.source = state.get("trigpingh_source", prms.timing_event)
            dly = state.get("trigpingh_delay", prms.trigpingh_delay)
            if dly is not None:
                trigpingh.delay = dly
        else:
            trigpingh.cmd_disable(timeout=prms.timeout)

        trigpingv = self.devices["trigpingv"]
        if ("v" in pingers2kick) or (state.get("trigpingv_state", 0) == 1):
            trigpingv.cmd_enable(timeout=prms.timeout)
            trigpingv.source = state.get("trigpingv_source", prms.timing_event)
            dly = state.get("trigpingv_delay", prms.trigpingv_delay)
            if dly is not None:
                trigpingv.delay = dly
        else:
            trigpingv.cmd_disable(timeout=prms.timeout)

    def get_magnets_strength(self):
        """."""
        return self.devices["pingh"].strength, self.devices["pingv"].strength

    def get_magnets_state(self):
        """."""
        return self.devices["pingh"].pwrstate, self.devices["pingv"].pwrstate

    def restore_magnets_state(self, state):
        """."""
        pinghpulse, pingvpulse = state
        pingh, pingv = self.devices['pingh'], self.devices['pingv']
        if pinghpulse:
            pingh_ok = pingh.cmd_turn_on(timeout=self.params.magnets_timeout)
        else:
            pingh_ok = pingh.cmd_turn_off(timeout=self.params.magnets_timeout)
        if pingvpulse:
            pingh_ok = pingv.cmd_turn_on(timeout=self.params.magnets_timeout)
        else:
            pingv_ok = pingv.cmd_turn_off(timeout=self.params.magnets_timeout)
        return pingh_ok and pingv_ok

    def set_magnets_strength(
        self, hkick=None, vkick=None, magnets_timeout=None
    ):
        """Set pingers strengths, check if was set & indicate which failed."""
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]
        if hkick is None:
            hkick = self.params.hkick / 1e3  # [urad] -> [mrad]
        pingh.set_strength(hkick, tol=0.1 * hkick, timeout=0, wait_mon=False)
        if vkick is None:
            vkick = self.params.vkick / 1e3  # [urad] -> [mrad]
        pingv.set_strength(vkick, tol=0.1 * vkick, timeout=0, wait_mon=False)

        # wait magnets ramp and check if correctly set
        if magnets_timeout is None:
            magnets_timeout = self.params.magnets_timeout
        t0 = _time.time()
        pingh_ok = pingh.set_strength(
            hkick, tol=0.05 * hkick, timeout=magnets_timeout, wait_mon=False
        )
        elapsed_time = _time.time() - t0
        magnets_timeout -= elapsed_time
        pingv_ok = pingv.set_strength(
            vkick, tol=0.05 * vkick, timeout=magnets_timeout, wait_mon=False
        )

        if (not pingh_ok) or (not pingv_ok):
            bad_pingers = "pingh " if not pingh_ok else ""
            bad_pingers += "pingv" if not pingv_ok else ""
            print(f"Some magnets were not set.\n\tBad pingers: {bad_pingers}")
            return False

        return True

    def prepare_magnets(self):
        """."""
        print('Preparing magnets...')
        mags_timeout = self.params.magnets_timeout
        pingers2kick = self.params.pingers2kick
        pingh = self.devices['pingh']
        pingv = self.devices['pingv']

        if 'h' in pingers2kick:
            pingh_ok = pingh.cmd_turn_on_pulse(self, timeout=mags_timeout)
            print('Failed to turn-on pingh.') if not pingh_ok else None
        else:
            pingh_ok = pingv.cmd_turn_off_pulse(self, timeout=mags_timeout)
            print('Failed to turn-off pingh.') if not pingh_ok else None
        print(f'pingh succesfully turned-{pingh.pwrstate}')

        if 'v' in pingers2kick:
            pingv_ok = pingv.cmd_turn_on_pulse(self, timeout=mags_timeout)
            print('Failed to turn-on pingv.') if not pingv_ok else None
        else:
            pingv_ok = pingv.cmd_turn_off_pulse(self, timeout=mags_timeout)
            print('Failed to turn-off pingh.') if not pingv_ok else None
        print(f'pingv succesfully turned-{pingv.pwrstate}')
        print('Setting magnets strengths...')
        if pingh_ok and pingv_ok:
            return self.set_magnets_strength()

    def do_measurement(self):
        """."""
        currinfo = self.devices["currinfo"]
        init_timing_state = self.get_timing_state()
        init_magnets_state = self.get_magnets_state()
        init_magnets_strength = self.get_magnets_strength()
        current_before = currinfo.current()
        self.prepare_timing()
        # TODO: prepare_magnets actions when strnegth is None
        mags_ok = self.prepare_magnets()  # gets strengths from params
        if mags_ok:
            print("Magnets strengths were succesfully set.")
            try:
                self.acquire_data()
                # BPMs signals + relevant info are acquired
                # such as timestamps tunes, stored current
                # rf frequency, acq rate, nr samples, etc.
                self.data["current_before"] = current_before
                self.data["current_after"] = self.data.pop("stored_current")
                self.data["init_magnets_strengths"] = init_magnets_strength
                strenghts = self.get_magnets_strength()
                self.data["manets_strengths"] = strenghts * 1e3  # [urad]
                print("Acquisition was succesful.")
            except Exception as e:
                print(f"An error occurred during acquisition: {e}")
        else:
            print("Did not measure. Restoring magnets & timing initial state.")
        self.recover_timing_state(init_timing_state)
        mags_ok = self.restore_magnets_state(init_magnets_state)
        mags_ok = self.set_magnets_strength(init_magnets_strength)  # restore
        if not mags_ok:
            msg = "Magnets strengths were not restored to initial values."
            msg += "Restore manually."
            print(msg)
            print("Initial strengths:")
            print(f"\t pingh:{init_magnets_strength[0]:.4f} [mrad]")
            print(f"\t pingv:{init_magnets_strength[1]:.4f} [mrad]")
        else:
            print("Magnets strengths succesfully restored to initial values.")
        print('Measurement finished.')

    def get_default_fname(self):
        """."""
        prms = self.params
        stg = "kickedbeam_data"
        stg += f"_{prms.acq_rate}_rate"

        pingers2kick = prms.pingers2kick
        if pingers2kick == "none":
            stg += "hkick_inactive_vkick_inactive"
        else:
            hkick, vkick = int(round(prms.hkick)), int(round(prms.vkick))
            stg += f"hkick_{hkick:3d}_urad" if "h" in pingers2kick else \
                "hkick_inactive"
            stg += f"vkick_{vkick:3d}_urad" if "h" in pingers2kick else \
                "vkick_inactive"

        tm = self.data["timestamp"]
        fmt = "%Y-%m-%d-%H-%M-%S"
        tmstp = _datetime.datetime.fromtimestamp(tm).strftime(fmt)
        stg += f"{tmstp}"
        return stg


class TbTDataAnalysis(MeasureTbTData):
    """."""

    def __init__(self, filename="", isonline=False):
        """Analysis of linear optics using Turn-by-turn data."""
        super().__init__(isonline=isonline)
        self.fname = filename

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

    def _get_fourier_components(self, matrix, tune):
        """Performs linear fit for Fourier components amplitudes."""
        ilist = _np.arange(matrix.shape[0])
        cos = _np.cos(2 * _np.pi * tune * ilist)
        sin = _np.sin(2 * _np.pi * tune * ilist)

        coeff_mat = _np.concatenate((cos[:, None], sin[:, None]), axis=1)
        fourier_components = _np.linalg.pinv(coeff_mat) @ matrix

        return fourier_components

    def _get_amplitude_phase(self, fourier_components):
        amplitudes = _np.sqrt(_np.sum(fourier_components**2, axis=0))
        phases = _np.arctan2(
            fourier_components[0, :], fourier_components[-1, :]
        )
        return amplitudes, phases

    def harmonic_tbt_model(self, ilist, *args, return_ravel=True):
        """Harmonic motion model for positions seen at a given BPM."""
        nbpms = (len(args)-1) // 2
        tune = args[0]
        amps = _np.array(args[1:nbpms+1])
        phases = _np.array(args[-nbpms:])
        x = ilist.reshape((-1, nbpms), order="F")
        wr = 2 * _np.pi * tune
        mod = amps[None, :] * _np.sin(wr * x + phases[None, :])
        if return_ravel:
            return mod.ravel()
        return mod

    def fit_harmonic_model(
        self, matrix, tune_guess, amplitudes_guess, phases_guess
    ):
        """Fits harmonic TbT model to data."""
        bpmdata = matrix.ravel()
        nturns, nbpms = matrix.shape[0], matrix.shape[1]
        xdata = _np.tile(_np.arange(nturns), nbpms).ravel()
        p0 = _np.concatenate(
            ([tune_guess], amplitudes_guess, phases_guess)
        ).tolist()

        popt, pcov = _curve_fit(
            f=self.harmonic_tbt_model, xdata=xdata, ydata=bpmdata, p0=p0
        )
        return popt, _np.diagonal(pcov)

    def calculate_betafunc_and_action(self, amplitudes, nominal_beta):
        """Calculates beta function and betatron action.

        As in Eq. (9) of Ref. [1]
        """
        action = _np.sum(amplitudes**4)
        action /= _np.sum(amplitudes**2 * nominal_beta)
        beta = amplitudes**2 / action
        return beta, action
