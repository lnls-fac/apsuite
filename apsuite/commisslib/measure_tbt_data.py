"""."""

import datetime as _datetime
import time as _time

import matplotlib.pyplot as _mplt
import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
from siriuspy.devices import PowerSupplyPU, Trigger

import pyaccel as _pa
from pymodels import si as _si

from ..optics_analysis import ChromCorr as _ChromCorr, TuneCorr as _TuneCorr
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
        self.hkick = None  # [mrad]
        self.vkick = None  # [mrad]
        self.trigpingh_delay = None
        self.trigpingv_delay = None
        self.magnets_timeout = 5.0

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
            stg += ftmp("hkick", self.hkick, "[mrad]")
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
            stg += ftmp("vkick", self.vkick, "[mrad]")
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

        self.devices["pingv"] = PowerSupplyPU(PowerSupplyPU.DEVICES.SI_PING_V)
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
        print("Setting BPMs timing")
        super().prepare_timing(state)  # BPM trigger timing
        state = dict() if state is None else state
        print("Setting magnets timing")
        # magnets trigger timing below
        prms = self.params
        trigpingh = self.devices["trigpingh"]
        pingers2kick = prms.pingers2kick
        if ("h" in pingers2kick) or (state.get("trigpingh_state", 0) == 1):
            trigh_ok = trigpingh.cmd_enable(timeout=prms.timeout)
            trigpingh.source = state.get("trigpingh_source", prms.timing_event)
            dly = state.get("trigpingh_delay", prms.trigpingh_delay)
            if dly is not None:
                trigpingh.delay = dly
        else:
            trigh_ok = trigpingh.cmd_disable(timeout=prms.timeout)
        print(f"PingerH trigger set: {trigh_ok}")

        trigpingv = self.devices["trigpingv"]
        if ("v" in pingers2kick) or (state.get("trigpingv_state", 0) == 1):
            trigv_ok = trigpingv.cmd_enable(timeout=prms.timeout)
            trigpingv.source = state.get("trigpingv_source", prms.timing_event)
            dly = state.get("trigpingv_delay", prms.trigpingv_delay)
            if dly is not None:
                trigpingv.delay = dly
        else:
            trigv_ok = trigpingv.cmd_disable(timeout=prms.timeout)
        print(f"PingerV trigger set: {trigv_ok}")

        return trigh_ok and trigv_ok

    def get_magnets_strength(self):
        """."""
        return self.devices["pingh"].strength, self.devices["pingv"].strength

    def get_magnets_state(self):
        """."""
        state = dict()
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]
        state["pingh_pwr"] = pingh.pwrstate
        state["pingv_pwr"] = pingv.pwrstate
        state["pingh_pulse"] = pingh.pulse
        state["pingv_pulse"] = pingv.pulse
        return state

    def set_magnets_state(self, state):
        """Restore magnets pwr and pulse states."""
        timeout = self.params.magnets_timeout
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]

        if state["pingh_pwr"]:
            pingh_ok = pingh.cmd_turn_on(timeout)
        else:
            pingh_ok = pingh.cmd_turn_off(timeout)

        if state["pingv_pwr"]:
            pingv_ok = pingv.cmd_turn_on(timeout)
        else:
            pingv_ok = pingv.cmd_turn_off(timeout)

        if state["pingh_pulse"]:
            pingh_ok = pingh.cmd_turn_on_pulse(timeout)
        else:
            pingh_ok = pingh.cmd_turn_off_pulse(timeout)

        if state["pingv_pulse"]:
            pingv_ok = pingv.cmd_turn_on_pulse(timeout)
        else:
            pingv_ok = pingv.cmd_turn_off_pulse(timeout)

        return pingh_ok and pingv_ok

    def set_magnets_strength(
        self, hkick=None, vkick=None, magnets_timeout=None
    ):
        """Set pingers strengths, check if was set & indicate which failed."""
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]
        if hkick is not None:
            pingh.set_strength(hkick, timeout=0, wait_mon=False)
        else:
            print("pingh not changed (None-type strength)")
            pingh_ok = True

        if vkick is not None:
            pingv.set_strength(vkick, timeout=0, wait_mon=False)
        else:
            print("pingv not changed (None-type strength)")
            pingv_ok = True

        # wait magnets ramp and check if correctly set
        if magnets_timeout is None:
            magnets_timeout = self.params.magnets_timeout
        t0 = _time.time()
        if hkick is not None:
            pingh_ok = pingh.set_strength(
                hkick,
                tol=0.05 * abs(hkick),
                timeout=magnets_timeout,
                wait_mon=False,
            )
        elapsed_time = _time.time() - t0
        magnets_timeout -= elapsed_time
        if vkick is not None:
            pingv_ok = pingv.set_strength(
                vkick,
                tol=0.05 * abs(vkick),
                timeout=magnets_timeout,
                wait_mon=False,
            )

        if (not pingh_ok) or (not pingv_ok):
            bad_pingers = "pingh " if not pingh_ok else ""
            bad_pingers += "pingv" if not pingv_ok else ""
            print(f"Some magnets were not set.\n\tBad pingers: {bad_pingers}")
            return False
        return True

    def prepare_magnets(self):
        """."""
        print("Preparing magnets...")
        pingers2kick = self.params.pingers2kick
        state = dict()
        if "h" in pingers2kick:
            state["pingh_pwr"] = 1
            state["pingh_pulse"] = 1
        else:
            state["pingh_pwr"] = 0
            state["pingh_pulse"] = 0

        if "v" in pingers2kick:
            state["pingv_pwr"] = 1
            state["pingv_pulse"] = 1
        else:
            state["pingv_pwr"] = 0
            state["pingv_pulse"] = 0

        state_ok = self.set_magnets_state(state)

        if state_ok:
            hkick = self.params.hkick
            vkick = self.params.vkick
            print("Setting magnets strengths...")
            return self.set_magnets_strength(hkick, vkick)
        else:
            print("Magnets state not set.")
            return False

    def do_measurement(self):
        """."""
        currinfo = self.devices["currinfo"]
        init_timing_state = self.get_timing_state()
        init_magnets_state = self.get_magnets_state()
        init_magnets_strength = self.get_magnets_strength()
        current_before = currinfo.current
        timing_ok = self.prepare_timing()
        if timing_ok:
            print("Timing configs. were succesfully set")
        mags_ok = self.prepare_magnets()  # gets strengths from params
        if mags_ok:
            print("Magnets ready.")
        if mags_ok and timing_ok:
            try:
                self.acquire_data()
                # BPMs signals + relevant info are acquired
                # such as timestamps tunes, stored current
                # rf frequency, acq rate, nr samples, etc.
                self.data["current_before"] = current_before
                self.data["current_after"] = self.data.pop("stored_current")
                self.data["init_magnets_strengths"] = init_magnets_strength
                print("Acquisition was succesful.")
            except Exception as e:
                print(f"An error occurred during acquisition: {e}")
        else:
            print("Did not measure. Restoring magnets & timing initial state.")
        timing_ok = self.recover_timing_state(init_timing_state)
        if not timing_ok:
            print("Timing was not restored to initial state.")
        mags_ok = self.set_magnets_state(init_magnets_state)
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
        print("Measurement finished.")

    def get_data(self):
        """."""
        data = super().get_data()
        data["magnets_state"] = self.get_magnets_state()
        data["magnets_strengths"] = self.get_magnets_strength()  # [mrad]
        return data

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
            stg += (
                f"hkick_{hkick:3d}_mrad"
                if "h" in pingers2kick
                else "hkick_inactive"
            )
            stg += (
                f"vkick_{vkick:3d}_mrad"
                if "h" in pingers2kick
                else "vkick_inactive"
            )

        tm = self.data["timestamp"]
        fmt = "%Y-%m-%d-%H-%M-%S"
        tmstp = _datetime.datetime.fromtimestamp(tm).strftime(fmt)
        stg += f"{tmstp}"
        return stg


class TbTDataAnalysis(MeasureTbTData):
    """."""
    SYNCH_TUNE = 0.004713

    def __init__(self, filename="", isonline=False):
        """Analysis of linear optics using Turn-by-turn data."""
        super().__init__(isonline=isonline)
        self._fname = filename
        self.timestamp = None
        self.trajx, self.trajy = None, None  # zero-mean trajectories in [mm]
        self.trajsum = None
        self.tunex, self.tuney = None, None
        self.acq_rate = None
        self.sampling_freq = None
        self.switching_freq = None
        self.rf_freq = None
        self.trajx_turns_slice = None
        self.trajy_turns_slice = None
        self.model_optics = None
        self.fitted_optics = None
        self.pca_optics = None
        if self._fname:
            self.load_and_apply(self._fname)

    def __str__(self):
        """."""
        stg = ""
        data = self.data
        if data:
            stg += "\n"
            stg += "Measurement data:\n"

            ftmp = "{0:26s} = {1:9.6f}  {2:s}\n".format
            stmp = "{0:26s} = {1:9}  {2:s}\n".format
            gtmp = "{0:<15s} = {1:}  {2:}\n".format

            stg += gtmp("timestamp", self.timestamp, "")
            stg += "\n"
            stg += "Storage Ring State\n"

            stg += ftmp("current_before", data["current_before"], "mA")
            stg += ftmp("current_after", data["current_after"], "mA")
            stg += ftmp("tunex", data["tunex"], "")
            stg += ftmp("tuney", data["tuney"], "")
            stg += stmp("tunex_enable", bool(data["tunex_enable"]), "")
            stg += stmp("tuney_enable", bool(data["tuney_enable"]), "")

            stg += "\n"
            stg += "EVT state\n"

            stg += stmp("evt_mode", data["timing_state"]["evt_mode"], "")
            stg += stmp("evt_delay", data["timing_state"]["evt_delay"], "")

            stg += "\n"
            stg += "BPMs state\n"

            stg += stmp("acq_rate", data["acq_rate"], "")
            stg += stmp("nrsamples_pre", data["nrsamples_pre"], "")
            stg += stmp("nrsamples_post", data["nrsamples_post"], "")
            stg += stmp("switching_mode", data["switching_mode"], "")
            stg += stmp("switching_frequency", data["switching_frequency"], "")
            stg += stmp(
                "trigbpm_source", data["timing_state"]["trigbpm_source"], ""
            )
            stg += stmp(
                "trigbpm_nrpulses",
                data["timing_state"]["trigbpm_nrpulses"],
                "",
            )
            stg += stmp(
                "trigbpm_delay", data["timing_state"]["trigbpm_delay"], ""
            )

            stg += "\n"
            stg += "Pingers state\n"

            stg += stmp(
                "trigpingh_state", data["timing_state"]["trigpingh_state"], ""
            )
            stg += stmp(
                "trigpingh_source",
                data["timing_state"]["trigpingh_source"],
                "",
            )
            stg += stmp(
                "trigpingh_delay", data["timing_state"]["trigpingh_delay"], ""
            )
            # stg += stmp("pingh_pwr", data["magnets_state"]["pingh_pwr"], "")
            # stg += stmp(
                # "pingh_pulse", data["magnets_state"]["pingh_pulse"], ""
            # )
            stg += ftmp("hkick", data["magnets_strengths"][0], "mrad")

            stg += stmp(
                "trigpingv_state", data["timing_state"]["trigpingv_state"], ""
            )
            stg += stmp(
                "trigpingv_source",
                data["timing_state"]["trigpingv_source"],
                "",
            )
            stg += stmp(
                "trigpingv_delay", data["timing_state"]["trigpingv_delay"], ""
            )
            # stg += stmp("pingv_pwr", data["magnets_state"]["pingv_pwr"], "")
            # stg += stmp(
                # "pingv_pulse", data["magnets_state"]["pingv_pulse"], ""
            # )
            stg += ftmp("vkick", data["magnets_strengths"][1], "mrad")
        return stg

    @property
    def fname(self):
        """."""
        return self._fname

    @fname.setter
    def fname(self, val):
        """."""
        self._fname = val
        self.load_and_apply(val)

    def load_and_apply(self, fname):
        """Load data and copy often used data to class attributes."""
        keys = super().load_and_apply(fname)
        if keys:
            print("The following keys were not used:")
            print("     ", str(keys))
        data = self.data
        timestamp = _datetime.datetime.fromtimestamp(data["timestamp"])
        self.timestamp = timestamp
        trajx, trajy = data["orbx"].copy() * 1e-3, data["orby"].copy() * 1e-3
        trajsum = data["sumdata"].copy()
        # zero mean in samples dimension
        trajx -= trajx.mean(axis=0)[None, :]
        trajy -= trajy.mean(axis=0)[None, :]
        self.trajx, self.trajy, self.trajsum = trajx, trajy, trajsum
        self.tunex, self.tuney = data['tunex'], data['tuney']
        self.acq_rate = data["acq_rate"]
        self.rf_freq = data["rf_frequency"]
        self.sampling_freq = self.data["sampling_frequency"]
        self.switching_freq = self.data["switching_frequency"]
        return

    def linear_optics_analysis(self):
        """."""
        self.harmonic_analysis()
        self.principal_components_analysis()
        raise NotImplementedError

    def harmonic_analysis(self, guess_tunes=False):
        """."""
        if guess_tunes:
            tunex, tuney = self._guess_tune_from_dft()
        else:
            tunex, tuney = self.tunex, self.tuney

        fitted_optics = dict()
        for pinger in self.params.pingers2kick:
            if pinger == "h":
                from_turn2turn = self.trajx_turns_slice
                turns_slice = slice(
                    from_turn2turn[0], from_turn2turn[1] + 1, 1
                )
                traj = self.trajx[turns_slice, :].copy()
                tune = tunex
                label = "x"
            else:
                from_turn2turn = self.trajy_turns_slice
                turns_slice = slice(
                    from_turn2turn[0], from_turn2turn[1] + 1, 1
                )
                traj = self.trajy[turns_slice, :].copy()
                tune = tuney
                label = "y"

            # TODO: adapt turns selection to grant integer nr of cycles
            nbpms = traj.shape[-1]

            fourier = self._get_fourier_components(traj, tune)
            amps, phases = self._get_amplitude_phase(fourier)
            params_guess = _np.concatenate(([tune], amps, phases)).tolist()
            n = self._get_independent_variables(from_turn2turn, nbpms)

            initial_fit = self.harmonic_tbt_model(
                n, *params_guess, return_ravel=False
            )

            params_fit, params_error = self.fit_harmonic_model(
                from_turn2turn, traj, *params_guess
            )
            params_fit = params_fit.tolist()
            final_fit = self.harmonic_tbt_model(
                n, *params_fit, return_ravel=False
            )

            tune = params_fit[0]
            amps = _np.array(params_fit[1 : nbpms + 1])
            phases_fit = _np.array(params_fit[-nbpms:])

            self._get_nominal_optics(tunes=(tunex, tuney))
            beta_model = self.model_optics["beta"+label]
            phases_model = self.model_optics["phase"+label]
            beta_fit, action = self.calc_beta_and_action(amps, beta_model)

            # collect fitted data
            fitted_optics["tune"+label] = tune
            fitted_optics["tune_err"+label] = params_error[0]
            fitted_optics["beta"+label] = beta_fit
            # TODO: propagate amplitude errors to beta errors
            fitted_optics["beta"+label+"_err"] = params_error[1 : nbpms + 1]
            fitted_optics["phase"+label] = phases_fit
            fitted_optics["phase"+label+"_err"] = params_error[-nbpms:]
            fitted_optics["action"+label] = action
            # TODO: propagate amplitude errors to action error
            fitted_optics["traj"+label+"_init_fit"] = initial_fit
            fitted_optics["traj"+label+"_final_fit"] = final_fit

            self.fitted_optics = fitted_optics

            self.plot_betabeat_and_phase_error(
                beta_model, beta_fit, phases_model, phases_fit,
                title=f"beta{label} & phase{label} - from harmonic model fit"
            )

            # TODO: compare fit with trajectory
        self.fitted_optics = fitted_optics

    def principal_components_analysis(self):
        """."""
        pca_optics = dict()
        for pinger in self.params.pingers2kick:
            tunes = self.tunex, self.tuney
            if self.model_optics is None:
                self._get_nominal_optics(tunes)
            beta_model = self.model_optics["beta"+pinger]
            phase_model = self.model_optics["phase"+pinger]
            beta_pca, phase_pca = self.calc_beta_and_phase_with_pca(self.trajx)
            self.plot_betabeat_and_phase_error(
                beta_model, beta_pca, phase_model, phase_pca,
                title=f'beta{pinger} & phase{pinger} - from PCA'
            )
            pca_optics["beta"+pinger] = beta_pca
            pca_optics["phase"+pinger] = phase_pca
        self.pca_optics = pca_optics

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

    def plot_trajs(self, bpm_index=0, timescale=0):
        """Plot trajectories and sum-signal at a given BPM and time-scale.

        Timescale 0 : Harmonic motion
        Timescale 1 : Chromaticity decoherence modulation
        Timescale 2 : Transverse decoherence modulations
        """
        nr_pre = self.data['nrsamples_pre']
        nr_post = self.data['nrsamples_post']
        n = 5

        if not timescale:
            nmax_x, nmax_y = int(1 / self.tunex), int(1 / self.tuney)
            slicex = (nr_pre - n, nr_pre + nmax_x + n + 1)
            slicey = (nr_pre - n, nr_pre + nmax_y + n + 1)
            slicesum = (nr_pre - n, nr_pre + max(nmax_x, nmax_y) + n + 1)

        nmax = int(1 / self.SYNCH_TUNE)
        if timescale == 1:
            slicex = (nr_pre - n, nr_pre + nmax + n + 1)
            slicey = slicex
            slicesum = slicex

        elif timescale == 2:
            slicex = (nr_pre + nmax, nr_pre + nr_post + 1)
            slicey = slicex
            slicesum = slicex

        trajx = self.trajx[:, bpm_index]
        trajy = self.trajy[:, bpm_index]
        trajsum = self.trajsum[:, bpm_index]

        fig, ax = _mplt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(
            f"{self.acq_rate.upper()} acquisition at BPM {bpm_index:3d}"
        )
        ax[0].set_title("horizontal trajectory")
        ax[0].plot(trajx, "-", mfc="none", color="blue")
        ax[0].set_xlim(slicex)
        ax[0].set_ylabel("position [mm]")
        ax[1].set_title("vertical trajectory")
        ax[1].plot(trajy, "-", mfc="none", color="red")
        ax[1].set_xlim(slicey)
        ax[1].sharey(ax[0])
        ax[2].set_title("BPM sum signal")
        ax[2].plot(trajsum, "-", mfc="none", color="k")
        ax[2].set_xlim(slicesum)

        ax[2].set_ylabel("sum [a.u.]")

        fig.supxlabel("turn index")
        fig.tight_layout()
        _mplt.show()
        return fig, ax

    def plot_trajs_vs_fit(self):
        """."""
        raise NotImplementedError

    def plot_betabeat_and_phase_error(
        self, beta_model, beta_meas, phase_model, phase_meas, title=None
    ):
        """."""
        fig, ax = _mplt.subplots(1, 3, figsize=(15, 5))
        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("beta and phase")
        ax[0].plot(beta_model, "o-", label="Model", mfc="none")
        ax[0].plot(beta_meas, "o--", label="Meas", mfc="none")
        ax[0].set_ylabel("beta function")
        ax[0].legend()

        beta_beat = (beta_model - beta_meas) / beta_model
        ax[1].plot(
            beta_beat * 100,
            "o-",
            label=f"rms = {beta_beat.std()*100:.2f} %",
            mfc="none",
        )
        ax[1].set_ylabel("beta beating [%]")
        ax[1].legend()
        delta_mu = phase_model - phase_meas
        ax[2].plot(phase_model, "o-", label="Model", mfc="none")
        ax[2].plot(phase_meas, "o--", label="Meas", mfc="none")
        ax[2].plot(
            delta_mu,
            "o-",
            label=f"max abs err: {_np.abs(delta_mu.max()):.2f})",
            mfc="none",
        )
        ax[2].set_ylabel("phase advance [rad]")
        ax[2].legend()

        fig.supxlabel("BPM")
        fig.tight_layout()
        _mplt.show()

    def _guess_tune_from_dft(self):
        """."""
        tune_guesses = list()
        for plane in "xy":
            traj = self.trajx if plane == "x" else self.trajy
            traj_dft, tune = self.calc_spectrum(traj, fs=1, axis=0)
            tunes = tune[:, None] * _np.ones(traj.shape[-1])[None, :]
            peak_idcs = _np.abs(traj_dft).argmax(axis=0)
            tune_peaks = [
                tunes[idc, col] for col, idc in enumerate(peak_idcs)
            ]  # peaking tune at each BPM's DFT.
            tune_guesses.append(_np.mean(tune_peaks))
        return tune_guesses

    def _get_fourier_components(self, matrix, tune):
        """Performs linear fit for Fourier components amplitudes."""
        n = _np.arange(matrix.shape[0])
        cos = _np.cos(2 * _np.pi * tune * n)
        sin = _np.sin(2 * _np.pi * tune * n)

        coeff_mat = _np.concatenate((cos[:, None], sin[:, None]), axis=1)
        fourier_components = _np.linalg.pinv(coeff_mat) @ matrix

        return fourier_components

    def _get_amplitude_phase(self, fourier_components):
        amplitudes = _np.sqrt(_np.sum(fourier_components**2, axis=0))
        phases = _np.arctan2(
            fourier_components[0, :], fourier_components[-1, :]
        )
        return amplitudes, _np.unwrap(phases)

    def harmonic_tbt_model(self, n, *args, return_ravel=True):
        """Harmonic motion model for positions seen at a given BPM."""
        nbpms = (len(args) - 1) // 2
        tune = args[0]
        amps = _np.array(args[1 : nbpms + 1])
        phases = _np.array(args[-nbpms:])
        x = n.reshape((-1, nbpms), order="F")
        wr = 2 * _np.pi * tune
        mod = amps[None, :] * _np.sin(wr * x + phases[None, :])
        if return_ravel:
            return mod.ravel()
        return mod

    def _get_independent_variables(self, from_turn2turn, nbpms):
        arange = _np.arange(from_turn2turn[0], from_turn2turn[1] + 1, 1)
        return _np.tile(arange, nbpms)

    def fit_harmonic_model(self, from_turn2turn, traj, *params_guess):
        """Fits harmonic TbT model to data."""
        bpmdata = traj.ravel()
        nbpms = traj.shape[1]
        xdata = self._get_independent_variables(from_turn2turn, nbpms)
        # TODO: add exception in case fit fails
        popt, pcov = _curve_fit(
            f=self.harmonic_tbt_model, xdata=xdata, ydata=bpmdata,
            p0=params_guess
        )
        return popt, _np.diagonal(pcov)

    def calc_beta_and_action(self, amplitudes, nominal_beta):
        """Calculates beta function and betatron action.

        As in Eq. (9) of Ref. [1]
        """
        action = _np.sum(amplitudes**4)
        action /= _np.sum(amplitudes**2 * nominal_beta)
        beta = amplitudes**2 / action
        return beta, action

    def calc_beta_and_phase_with_pca(self, matrix, beta_model):
        """."""
        _, svals, vtmat = self.calc_svd(matrix, full_matrices=False)
        beta_meas = (
            svals[0] ** 2 * vtmat[0, :] ** 2 + svals[1] ** 2 * vtmat[1, :] ** 2
        )
        beta_meas /= _np.std(beta_meas) / _np.std(beta_model)
        phase_meas = _np.arctan2(
            svals[1] * vtmat[1, :], svals[0] * vtmat[0, :]
        )
        phase_meas = _np.abs(_np.unwrap(phase_meas))  # why the abs?
        return beta_meas, phase_meas

    def _get_nominal_optics(self, tunes=None, chroms=None):
        """."""
        if self.model_optics is None:
            model_optics = dict()
            model = _si.create_accelerator()
            model = _si.fitted_models.vertical_dispersion_and_coupling(model)
            model.radiation_on = False
            model.cavity_on = False
            model.vchamber_on = False

            if tunes is not None:
                tunecorr = _TuneCorr(model, acc="SI")
                tunex = tunes[0] + 49
                tuney = tunes[0] + 14
                tunecorr.correct_parameters(goal_parameters=(tunex, tuney))

            chroms = (2.5, 2.5)
            chromcorr = _ChromCorr(model, acc="SI")
            chromcorr.correct_parameters(goal_parameters=chroms)

            famdata = _si.get_family_data(model)
            twiss, *_ = _pa.optics.calc_twiss(
                accelerator=model, indices="open"
            )
            bpms_idcs = _pa.lattice.flatten(famdata["BPM"]["index"])
            model_optics["betax"] = twiss.betax[bpms_idcs]
            model_optics["phasex"] = twiss.mux[bpms_idcs]
            model_optics["betay"] = twiss.betay[bpms_idcs]
            model_optics["phasey"] = twiss.muy[bpms_idcs]
            self.model_optics = model_optics
