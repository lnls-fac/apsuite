"""."""
import datetime as _datetime
import time as _time

import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _gridspec
import numpy as _np
from sklearn.decomposition import FastICA as _FastICA
from scipy.optimize import curve_fit as _curve_fit

from mathphys.sobi import SOBI as _SOBI
from siriuspy.devices import PowerSupplyPU, Trigger
from siriuspy.sofb.csdev import SOFBFactory

import pyaccel as _pa
from pymodels import si as _si

from ..optics_analysis import ChromCorr as _ChromCorr, TuneCorr as _TuneCorr
from .meas_bpms_signals import AcqBPMsSignals as _AcqBPMsSignals, \
    AcqBPMsSignalsParams as _AcqBPMsSignalsParams


class TbTDataParams(_AcqBPMsSignalsParams):
    """."""
    BPMS_NAMES = SOFBFactory.create("SI").bpm_names

    def __init__(self):
        """."""
        super().__init__()
        self.signals2acq = "XYS"
        self.acq_rate = "TbT"
        self.nrpoints_before = 100
        self.nrpoints_after = 2000

        self._pingers2kick = "none"  # 'H', 'V' or 'HV'
        self.hkick = None  # [mrad]
        self.vkick = None  # [mrad]
        self.pingh_calibration = 1.5416651659146232
        self.pingv_calibration = 1.02267573
        self.trigpingh_delay = None
        self.trigpingv_delay = None
        self.magnets_timeout = 120.
        self.restore_init_state = True

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
        stg += ftmp("pingh_calibration", self.pingh_calibration, "")
        stg += ftmp("pingv_calibration", self.pingv_calibration, "")

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
        stg += stmp("restore_init_state", self.restore_init_state, "")
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
        state["trigpingh_state"] = trigpingh.state
        state["trigpingh_source"] = trigpingh.source_str
        state["trigpingh_delay"] = trigpingh.delay

        trigpingv = self.devices["trigpingv"]
        state["trigpingv_state"] = trigpingv.state
        state["trigpingv_source"] = trigpingv.source_str
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
        pingh_cal = self.params.pingh_calibration
        pingv_cal = self.params.pingv_calibration
        pingh_str = self.devices["pingh"].strength / pingh_cal
        pingv_str = self.devices["pingv"].strength / pingv_cal
        return pingh_str, pingv_str

    def get_magnets_state(self):
        """."""
        state = dict()
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]
        pingh_str, pingv_str = self.get_magnets_strength()
        state["pingh_pwr"] = pingh.pwrstate
        state["pingv_pwr"] = pingv.pwrstate
        state["pingh_pulse"] = pingh.pulse
        state["pingv_pulse"] = pingv.pulse
        state["pingh_strength"] = pingh_str
        state["pingv_strength"] = pingv_str
        state["pingh_voltage"] = pingh.voltage
        state["pingv_voltage"] = pingv.voltage

        return state

    def set_magnets_state(self, state, wait_mon=True):
        """Set magnets strengths, pwr and pulse states."""
        timeout = self.params.magnets_timeout
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]

        # turn-off pulse before changing strengths
        pingh_ok = pingh.cmd_turn_off_pulse(timeout)
        pingv_ok = pingv.cmd_turn_off_pulse(timeout)

        # Power and strengths
        if state.get("pingh_pwr", False):
            pingh_ok = pingh.cmd_turn_on(timeout=self.params.magnets_timeout)
            if pingh_ok:
                pingh_ok = self.set_magnets_strength(
                    hkick=state["pingh_strength"], wait_mon=wait_mon
                )
            else:
                print("Failed at turning-on pingh.")
        else:
            pingh_ok = self.set_magnets_strength(
                hkick=state["pingh_strength"], wait_mon=wait_mon
            )
            if pingh_ok:
                pingh_ok = pingh.cmd_turn_off(
                    timeout=self.params.magnets_timeout
                )

        if state.get("pingv_pwr", False):
            pingv_ok = pingv.cmd_turn_on(timeout=self.params.magnets_timeout)
            if pingv_ok:
                pingv_ok = self.set_magnets_strength(
                    vkick=state["pingv_strength"], wait_mon=wait_mon
                )
            else:
                print("Failed at turning-on pingv.")
        else:
            pingv_ok = self.set_magnets_strength(
                vkick=state["pingv_strength"], wait_mon=wait_mon
            )
            if pingv_ok:
                pingv_ok = pingv.cmd_turn_off(
                    timeout=self.params.magnets_timeout
                )

        if pingh_ok and pingv_ok:
            print("Magnets power-state and strengths set.")
        else:
            print("Failed at setting magnets power-state and strengths")

        print("Changing pulse-state")

        # Pulse state
        if state["pingh_pulse"]:
            pingh_ok = pingh_ok and pingh.cmd_turn_on_pulse(timeout)
        else:
            pingh_ok = pingh_ok and pingh.cmd_turn_off_pulse(timeout)
        msg = "pingh pulse set" if pingh_ok else "pingh pulse not set"
        print("\t"+msg)

        if state["pingv_pulse"]:
            pingv_ok = pingv_ok and pingv.cmd_turn_on_pulse(timeout)
        else:
            pingv_ok = pingv_ok and pingv.cmd_turn_off_pulse(timeout)
        msg = "pingv pulse set" if pingv_ok else "pingv pulse not set"
        print("\t"+msg)

        return pingh_ok and pingv_ok

    def set_magnets_strength(
        self, hkick=None, vkick=None, magnets_timeout=None, wait_mon=True
    ):
        """Set pingers strengths, check if was set & indicate which failed."""
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]
        if hkick is not None:
            hkick *= self.params.pingh_calibration
            print("changing pingh")
            pingh.set_strength(hkick, timeout=0, wait_mon=wait_mon)
        else:
            pingh_ok = True

        if vkick is not None:
            vkick *= self.params.pingv_calibration
            print("changing pingv")
            pingv.set_strength(vkick, timeout=0, wait_mon=wait_mon)
        else:
            pingv_ok = True

        # wait magnets ramp and check if correctly set
        if magnets_timeout is None:
            magnets_timeout = self.params.magnets_timeout
        t0 = _time.time()

        if hkick is not None:
            print("waiting pingh")
            pingh_ok = pingh.set_strength(
                hkick,
                tol=0.05 * abs(hkick),
                timeout=magnets_timeout,
                wait_mon=wait_mon,
            )

        elapsed_time = _time.time() - t0
        magnets_timeout -= elapsed_time

        if vkick is not None:
            print("waiting pingv")
            pingv_ok = pingv.set_strength(
                vkick,
                tol=0.05 * abs(vkick),
                timeout=magnets_timeout,
                wait_mon=wait_mon,
            )

        if (not pingh_ok) or (not pingv_ok):
            bad_pingers = "pingh " if not pingh_ok else ""
            bad_pingers += "pingv" if not pingv_ok else ""
            msg = "Some magnets strengths were not set.\n"
            msg += f"\t Bad pingers: {bad_pingers}"
            return False

        return True

    def prepare_magnets(self):
        """."""
        print("Preparing magnets...")
        pingers2kick = self.params.pingers2kick
        state = dict()
        if "h" in pingers2kick:
            state["pingh_pwr"] = 1  # always make sure its on
            state["pingh_pulse"] = 1
        else:
            # state["pingh_pwr"] = 0  # but will not be turning-off.
            state["pingh_pulse"] = 0  # only changing pulse-sts

        if "v" in pingers2kick:
            state["pingv_pwr"] = 1
            state["pingv_pulse"] = 1
        else:
            # state["pingv_pwr"] = 0
            state["pingv_pulse"] = 0

        state["pingh_strength"] = self.params.hkick
        state["pingv_strength"] = self.params.vkick

        return self.set_magnets_state(state)

    def do_measurement(self):
        """."""
        currinfo = self.devices["currinfo"]
        init_timing_state = self.get_timing_state()
        init_magnets_state = self.get_magnets_state()
        current_before = currinfo.current
        timing_ok = self.prepare_timing()
        if timing_ok:
            print("Timing configs. were successfully set")
            mags_ok = self.prepare_magnets()  # gets strengths from params
            if mags_ok:
                print("Magnets ready.")
                try:
                    self.acquire_data()
                    # BPMs signals + relevant info are acquired
                    # such as timestamps tunes, stored current
                    # rf frequency, acq rate, nr samples, etc.
                    self.data["current_before"] = current_before
                    self.data["current_after"] = self.data.pop(
                        "stored_current"
                    )
                    self.data["init_magnets_state"] = init_magnets_state
                    print("Acquisition was successful.")
                except Exception as e:
                    print(f"An error occurred during acquisition: {e}")
        else:
            print("Did not measure")
        if self.params.restore_init_state:
            print("Restoring magnets & timing initial state.")
            timing_ok = self.prepare_timing(init_timing_state)
            if not timing_ok:
                print("Timing was not restored to initial state.")
            mags_ok = self.set_magnets_state(
                init_magnets_state, wait_mon=False
            )
            if not mags_ok:
                msg = "Magnets state or strengths were not restored."
                msg += "Restore manually."
                print(msg)
                print(init_magnets_state)
            else:
                print("Magnets state & strengths successfully restored.")
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
        stg += f"_{prms.acq_rate}_rate_"

        pingers2kick = prms.pingers2kick
        stg += (
            f"hkick_{prms.hkick:.4f}_mrad_".replace(".", "p").replace("-", "m")
            if "h" in pingers2kick
            else "hkick_inactive_"
        )
        stg += (
            f"vkick_{prms.vkick:.4f}_mrad_".replace(".", "p").replace("-", "m")
            if "v" in pingers2kick
            else "vkick_inactive_"
        )
        tm = self.data["timestamp"]
        fmt = "%Y-%m-%d-%H-%M-%S"
        tmstp = _datetime.datetime.fromtimestamp(tm).strftime(fmt)
        stg += f"{tmstp}"
        return stg


class TbTDataAnalysis(MeasureTbTData):
    """."""
    SYNCH_TUNE = 0.004713  # check this

    def __init__(self, filename="", isonline=False):
        """Analysis of linear optics using Turn-by-turn data."""
        super().__init__(isonline=isonline)

        self._fname = filename
        self.timestamp = None
        self.trajx, self.trajy = None, None  # zero-mean trajectories in [mm]
        self.trajsum = None

        self._tunex, self._tuney = None, None
        self.acq_rate = None
        self.nrsamples_pre = None
        self.nrsamples_post = None
        self.sampling_freq = None
        self.switching_freq = None
        self.rf_freq = None

        self.trajx_turns_slice = None
        self.trajy_turns_slice = None
        self.bpms2use = None

        self.model_optics = None
        self.fitting_data = None
        self.pca_data = None
        self.ica_data = None

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
            stg += "\n"

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
            stg += "\n"

            stg += stmp("acq_rate", data["acq_rate"], "")
            stg += stmp("nrsamples_pre", data["nrsamples_pre"], "")
            stg += stmp("nrsamples_post", data["nrsamples_post"], "")
            stg += stmp("switching_mode", data["switching_mode"], "")
            stg += stmp("switching_frequency", data["switching_frequency"], "")
            stg += stmp(
                "trigbpm_source",
                data["timing_state"]["trigbpm_source"],
                ""
            )
            stg += stmp(
                "trigbpm_nrpulses",
                data["timing_state"]["trigbpm_nrpulses"],
                ""
            )
            stg += stmp(
                "trigbpm_delay",
                data["timing_state"]["trigbpm_delay"],
                ""
            )

            stg += "\n"
            stg += "Pingers state\n"
            stg += "\n"

            stg += stmp(
                "trigpingh_state",
                data["timing_state"]["trigpingh_state"],
                ""
            )
            stg += stmp(
                "trigpingh_source",
                data["timing_state"]["trigpingh_source"],
                ""
            )
            stg += stmp(
                "trigpingh_delay",
                data["timing_state"]["trigpingh_delay"],
                ""
            )
            stg += stmp("pingh_pwr", data["magnets_state"]["pingh_pwr"], "")

            stg += stmp(
                "pingh_pulse",
                data["magnets_state"]["pingh_pulse"],
                ""
            )
            stg += ftmp("hkick", data["magnets_strengths"][0], "mrad")
            stg += "\n"

            stg += stmp(
                "trigpingv_state",
                data["timing_state"]["trigpingv_state"],
                ""
            )
            stg += stmp(
                "trigpingv_source",
                data["timing_state"]["trigpingv_source"],
                ""
            )
            stg += stmp(
                "trigpingv_delay",
                data["timing_state"]["trigpingv_delay"],
                ""
            )
            stg += stmp("pingv_pwr", data["magnets_state"]["pingv_pwr"], "")

            stg += stmp(
                "pingv_pulse",
                data["magnets_state"]["pingv_pulse"],
                ""
            )
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

    @property
    def tunex(self):
        """."""
        return self._tunex

    @tunex.setter
    def tunex(self, val):
        """."""
        if val is not None:
            self._tunex = val
            nrpre = self.nrsamples_pre
            self.trajx_turns_slice = (nrpre, nrpre + int(1 / val))

    @property
    def tuney(self):
        """."""
        return self._tuney

    @tuney.setter
    def tuney(self, val):
        """."""
        if val is not None:
            self._tuney = val
            nrpre = self.nrsamples_pre
            self.trajy_turns_slice = (nrpre, nrpre + int(1 / val))

    def load_and_apply(self, fname):
        """Load data and copy often used data to class attributes."""
        keys = super().load_and_apply(fname)
        if keys:
            print("The following keys were not used:")
            print("     ", str(keys))

        # make often used data attributes
        data = self.data
        epoch_tmstp = data.get("timestamp", None)
        timestamp = _datetime.datetime.fromtimestamp(
            epoch_tmstp
        ) if epoch_tmstp is not None else None
        self.timestamp = timestamp
        trajx = data.get("orbx", None).copy() * 1e-3
        trajy = data.get("orby", None).copy() * 1e-3
        trajsum = data.get("sumdata", None).copy()

        # zero mean in samples dimension
        trajx -= trajx.mean(axis=0)[None, :]
        trajy -= trajy.mean(axis=0)[None, :]
        self.trajx, self.trajy, self.trajsum = trajx, trajy, trajsum

        self.nrsamples_pre = data.get("nrsamples_pre", None)
        self.nrsamples_post = data.get("nrsamples_post", None)
        self.tunex = data.get('tunex', None)
        self.tuney = data.get('tuney', None)
        self.acq_rate = data.get("acq_rate", None)
        self.rf_freq = data.get("rf_frequency", None)
        self.sampling_freq = data.get("sampling_frequency", None)
        self.switching_freq = data.get("switching_frequency", None)
        return

    def linear_optics_analysis(self):
        """Linear optics (beta-beating & phase adv. errors) analysis.

        Determines beta-functions and phase-advances on BPMs via sinusoidal
        fitting of TbT data in the harmonic motion timescale, as well as via
        spatio-temporal modal analysis using Principal Components Analysis
        (PCA) and Independent Components Analysis (ICA).
        """
        self.harmonic_analysis()
        self.principal_components_analysis()
        self.independent_components_analysis()

    def harmonic_analysis(
        self, guess_tunes=True, plot=True, compare_meas2model=True
    ):
        r"""Linear optics analysis using sinusoidal model for TbT data.

        TbT motion at the i-th turn and j-th BPM  in the timescale of less
        than $\nu^{-1}$ turns reads

            $$ x_{ij} = A_j \sin (2 \pi \nu i + \phi_j) $$.

        The betatron tune $\nu$ and the amplitudes $A_j$ and phases $\phi_j$
        are fitted at each BPM.

        Phase-advance between adjacent BPMs is calulated from the fitted BPM
        phase $\phi_j$. The betatron action $J$ is determined from the fitted
        amplitudes $A_j$ and the nominal beta-function as in  eq. (9) of ref.
        [1]. Fitting of beta-functions from the calculated action and
        fitted amplitudes.


        Args:
            guess_tunes (bool, optional): whether to use the initial guess for
            the tunes from the data DFT or use the measured tunes. Defaults to
            True.

            plot (bool, optional): whether to plot analysis results (beta &
            phase advance). Defaults to True.

            compare_meas2model (bool, optional): whether to plot measured and
            nominal beta-functions and BPMs phase-advance, as well as
            beta-beting and phase-advance errors or plot only beta-beating and
            phase-advance-errors. Defaults to True

        References:
        [1] X.R. Resende, M.B. Alves, L. Liu, and F.H. de Sá, “Equilibrium and
            Nonlinear Beam Dynamics Parameters From Sirius Turn-by-Turn BPM
            Data”, in Proc. IPAC'21, Campinas, SP, Brazil, May 2021, pp.
            1935-1938. doi:10.18429/JACoW-IPAC2021-TUPAB219

        [2] Huang, X. Beam-based correction and optimization for accelerators.
            Section 5.1. CRC Press, 2020.
        """
        if guess_tunes:
            tunex, tuney = self._guess_tune_from_dft()
        else:
            tunex, tuney = self.tunex, self.tuney

        fitting_data = dict()
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

            # get initial guess of amplitudes and phases with linear fit
            fourier = self._get_fourier_components(traj, tune)
            amps, phases = self._get_amplitude_phase(fourier)
            params_guess = _np.concatenate(([tune], amps, phases)).tolist()

            # evaluate harmonic motion model with initial guess for params
            n = self._get_independent_variables(from_turn2turn, nbpms)
            initial_fit = self.harmonic_tbt_model(
                n, *params_guess, return_ravel=False
            )

            # perform nonlinear fit to refine search for params
            params_fit, params_error = self.fit_harmonic_model(
                from_turn2turn, traj, *params_guess
            )
            params_fit = params_fit.tolist()

            # evaluate harmonic motion model with nonlinear fit params
            final_fit = self.harmonic_tbt_model(
                n, *params_fit, return_ravel=False
            )

            # collect fitted params (tune, amplitudes and phases at BPMs)
            tune = params_fit[0]
            amps = _np.array(params_fit[1 : nbpms + 1])
            phases_fit = _np.array(params_fit[-nbpms:])

            if self.model_optics is None:
                self._get_nominal_optics(tunes=(tunex + 49., tuney + 14.))
            beta_model = self.model_optics["beta"+label]
            phases_model = self.model_optics["phase"+label]

            # fit beta-function & action from fitted amplitudes & nominal beta
            beta_fit, action = self.calc_beta_and_action(amps, beta_model)

            # evaluate fitting residues
            residue = traj - final_fit

            # store fitting data
            fitting_data["tune"+label] = tune
            fitting_data["tune_err"+label] = params_error[0]
            fitting_data["amplitudes"+label] = amps
            fitting_data["beta"+label] = beta_fit
            # TODO: propagate amplitude errors to beta errors
            fitting_data["beta"+label+"_err"] = params_error[1 : nbpms + 1]
            fitting_data["phase"+label] = phases_fit
            fitting_data["phase"+label+"_err"] = params_error[-nbpms:]
            fitting_data["action"+label] = action
            # TODO: propagate amplitude errors to action error
            fitting_data["traj"+label+"_init_fit"] = initial_fit
            fitting_data["traj"+label+"_final_fit"] = final_fit
            fitting_data["fitting"+label+"_residue"] = residue

            # Plot results (beta_beating and phase adv. error)
            if plot:
                title = f"Sinusoidal fit analysis - beta{label} & phase{label}"
                self.plot_betabeat_and_phase_error(
                    beta_model, beta_fit, phases_model, phases_fit,
                    title=title,
                    compare_meas2model=compare_meas2model,
                    bpms2use=self.bpms2use
                )
        self.fitting_data = fitting_data

    def principal_components_analysis(
                self,
                stackxy=True,
                planes="xy",
                plot=True,
                compare_meas2model=True
            ):
        r"""Peform linear optics analysis Principal Components Analysis (PCA).

        Calculates beta-functions and betatron phase-advance at the BPMs using
        PCA.

        PCA aims to identify principal axes along which the covariance
        matrix of the data is diagonal. For betatron-dominated motion, there
        are two pricipal components (cosine and sine modes) for each plane
        (horizontal and vertical) that can be related to the betatron
        functions and phase advance at the corresponding planes, as described
        in refs [1,2].

        For a beam history matrix X with with turn-by-turn samples along the
        rows (nturns x nbpms), it can be shown (ref. [1]) that the principal
        components diagonalizing the covariance matrix are the columns of the
        spatial patterns matrix, V, where V is such that H = U S V.T. The
        leading modes of the V matrix are the beatron modes [2].

        In the language of Blind Source Separation, if the data matrix X can
        be expressed as a linear mixture of uncorrelated source signals
        arranged as the columns of matrix S, i.e. X = S A^T, then, using PCA,
        the mixing matrix A can be identified with
        A = V @ S.T / sqrt{nturns - 1}.
        Acting on X with A's pseudo-inverse (the unmixing matrix) gives the
        whitened, uncorrelated source signals S = U \sqrt{nturns - 1}, with
        S.T @ S / (nturns - 1 ) = identity. This choice for the normalization
        of mixing matrix and whitened sources follows scikit-learn's [2]
        convention  and is compatible with the convention adopted in the
        independent components analysis (ICA).

        Args:
            stackxy (bool, optional): stack hrizontal (x) and vertical (y)
                trajectories and carry out analysis with nsamples x 320 history
                matrix. Defaults to True. In this case, the leading 2 pairs of
                SVD modes are associated with 4 dominant singular values,
                which are identified with the horizontal and vertical betatron
                modes.

            planes (str, optional): "x", "y" or "xy". Analyze the horizontal,
                the vertical or both planes one at a time. Defaults to "xy",
                in which case the x and y planes will be analyzed seprately.
                Not used if stackxy is True.

            plot (bool, optional): whether to plot the analysis results
                (beta-function and phase-advances). Defaults to True.

            compare_meas2model (bool, optional): whether to plot a comparison
                between the measured and nominal beta-functions and BPMs
                phase-advance, as well as the beta-beating and phase-advance
                errors. Defaults to True.

        References:
        [1] Wang, Chun-xi and Sajaev, Vadim and Yao, Chih-Yuan. Phase advance
            and ${\beta}$ function measurements using model-independent
            analysis. Phys. Rev. ST Accel. Beams. Vol 6, issue 10. DOI 10.1103/
            PhysRevSTAB.6.104001

        [2] Huang, X. Beam-Based Correction and Optimization for Accelerators,
            Ch 5.2. CRC Press. 2020.

        [3] Scikit-learn examples. "Blind Source Separation using FastICA".
            https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py
        """
        tunes = self.tunex + 49., self.tuney + 14.
        if self.model_optics is None:
            self._get_nominal_optics(tunes)

        # get model optics
        betax_model = self.model_optics["betax"]
        betay_model = self.model_optics["betay"]
        phasex_model = self.model_optics["phasex"]
        phasey_model = self.model_optics["phasey"]

        pca_data = dict()
        if stackxy:
            traj = _np.concatenate((self.trajx, self.trajy), axis=1)
            # perform PCA via SVD of history matrix
            umat, svals, vtmat = self.calc_svd(traj, full_matrices=False)

            # collect source signals and mixing matrix
            signals = umat * _np.sqrt(traj.shape[0] - 1)  # data signals
            mixing_matrix = vtmat.T @ _np.diag(svals)
            mixing_matrix /= _np.sqrt(traj.shape[0] - 1)

            # calulate tunes of the source signals
            spec1, tunes1 = self.calc_spectrum(signals[:, 0], axis=0)
            spec2, tunes2 = self.calc_spectrum(signals[:, 2], axis=0)
            # modes 0 & 1 and modes 2 & 3 have have equal spectra
            # therefore, it suffices to analyze mode 0 and mode 2 only

            tune1 = tunes1[_np.argmax(_np.abs(spec1))]
            tune2 = tunes2[_np.argmax(_np.abs(spec2))]

            # identify which signal is which (x or y)
            xidcs, yidcs = self.identify_modes(
                tune1, tune2, self.tunex, self.tuney
            )

            # extract betatron sine & cosine modes fom mixing matrix
            sin_modex = mixing_matrix[:160, xidcs[-1]]
            cos_modex = mixing_matrix[:160, xidcs[0]]
            sin_modey = mixing_matrix[160:, yidcs[-1]]
            cos_modey = mixing_matrix[160:, yidcs[0]]

            # calculate beta function & phase from betatron modes
            betax, phasex = self.get_beta_and_phase_from_betatron_modes(
                sin_modex, cos_modex, betax_model
            )
            betay, phasey = self.get_beta_and_phase_from_betatron_modes(
                sin_modey, cos_modey, betay_model
            )

            # concatenate results
            beta = _np.concatenate((betax, betay))
            phase = _np.concatenate((phasex, phasey))

            # calculate signal variance
            signalx = _np.sqrt(_np.sum(svals[xidcs[0]:xidcs[1] + 1]**2))
            signaly = _np.sqrt(_np.sum(svals[yidcs[0]:yidcs[1] + 1]**2))
            # and signal noise
            noise = _np.sqrt(_np.sum(svals[3:]**2))
            snrx, snry = signalx / noise, signaly / noise

            # calculate error bars as in Appendix A of
            # Wang, Chun-xi and Sajaev, Vadim and Yao, Chih-Yuan. Phase advance
            # and ${\beta}$ function measurements using model-independent
            # analysis. Phys. Rev. ST Accel. Beams. Vol 6, issue 10.
            # DOI 10.1103/PhysRevSTAB.6.104001

            nrsamples = self.nrsamples_pre + self.nrsamples_post
            phasex_error = 1 / snrx / _np.sqrt(nrsamples)
            phasex_error *= _np.sqrt(betax_model.mean() / 2 / betax_model)
            betax_error = 2 * betax_model * phasex_error

            phasey_error = 1 / snry / _np.sqrt(nrsamples)
            phasey_error *= _np.sqrt(betay_model.mean() / 2 / betay_model)
            betay_error = 2 * betay_model * phasey_error

            # TODO: plot stacked data
            # plot_results
            if plot:
                self.plot_betabeat_and_phase_error(
                    _np.concatenate((betax_model, betay_model)), beta,
                    _np.concatenate((phasex_model, phasey_model)), phase,
                    title="PCA Analysis: beta & phase (stacked x/y)",
                    compare_meas2model=compare_meas2model,
                    bpms2use=self.bpms2use
                )

            # save analysis data
            pca_data["singular_values"] = svals
            pca_data["source_signals"] = signals
            pca_data["mixing_matrix"] = mixing_matrix
            pca_data["xidcs"] = xidcs
            pca_data["yidcs"] = yidcs
            pca_data["beta"] = beta
            pca_data["phase"] = phase
            pca_data["snrx"] = snrx
            pca_data["snry"] = snry
            pca_data["beta_err"] = _np.concatenate((betax_error, betay_error))
            pca_data["phase_err"] = _np.concatenate(
                (phasex_error, phasey_error)
            )
            self.pca_data = pca_data
            return

        for plane in planes:
            if plane == "x":
                traj = self.trajx
            else:
                traj = self.trajy

            # get model optics
            beta_model = self.model_optics["beta"+plane]
            phase_model = self.model_optics["phase"+plane]

            # perform PCA via SVD of history matrix
            umat, svals, vtmat = self.calc_svd(traj, full_matrices=False)

            # collect source signals and mixing matrix
            signals = umat * _np.sqrt(traj.shape[0] - 1)  # data signals
            mixing_matrix = vtmat.T @ _np.diag(svals)
            mixing_matrix /= _np.sqrt(traj.shape[0] - 1)

            # determine betatron sine and cosine modes
            sin_mode = mixing_matrix[:, 1]  # check this
            cos_mode = mixing_matrix[:, 0]

            # calculate beta function & phase from betatron modes
            beta, phase = self.get_beta_and_phase_from_betatron_modes(
                sin_mode, cos_mode, beta_model
            )

            signal = _np.sqrt(_np.sum(svals[:2]**2))  # signal variance
            noise = _np.sqrt(_np.sum(svals[2:]**2))  # white noise
            snr = signal / noise
            nrsamples = self.nrsamples_pre + self.nrsamples_post
            phase_error = 1 / snr / _np.sqrt(nrsamples)
            phase_error *= _np.sqrt(beta_model.mean() / 2 / beta_model)
            beta_error = 2 * beta_model * phase_error

            # plot_results
            if plot:
                self.plot_betabeat_and_phase_error(
                    beta_model, beta,
                    phase_model, phase,
                    title=f"PCA Analysis: beta{plane} & phase{plane}",
                    compare_meas2model=compare_meas2model,
                    bpms2use=self.bpms2use
                )

            # save analysis data
            pca_data["singular_values_"+plane] = svals
            pca_data["source_signals_"+plane] = signals
            pca_data["mixing_matrix_"+plane] = mixing_matrix
            pca_data["beta"+plane] = beta
            pca_data["phase"+plane] = phase
            pca_data["snr"+plane] = snr
            pca_data["beta"+plane+"_err"] = beta_error
            pca_data["phase"+plane+"_err"] = phase_error
        self.pca_data = pca_data

    def identify_modes(self, tune1, tune2, tunex, tuney):
        """Identify the x and y betatron modes in mixing matrix.

        When calculating the mixing matrix via PCA or ICA, the horizontal and
        betatron modes need the be determined. PCA sorts the modes with
        increasing variance (singular values), while ICA sorts the modes
        arbitrairly. By calculating the tune of the source signals
        corresponding to a given betatron mode and comparing it to the
        reference horizontal and vertical tunes, the hortizontal and vertical
        beatron modes can be identified.

        Args:
            tune1 (float): tune of the 1st & 2nd mode
            tune2 (float): tune of the 3rd and 4th mode
            tunex (float): _description_
            tuney (float): _description_

        Returns:
            _type_: _description_
        """
        diff1x = _np.abs(tune1 - tunex)
        diff1y = _np.abs(tune1 - tuney)
        diff2x = _np.abs(tune2 - tunex)
        diff2y = _np.abs(tune2 - tuney)

        # Assign based on minimum differences
        if diff1x < diff2x and diff2y < diff1y:
            xidcs = 0, 1
            yidcs = 2, 3
        else:
            xidcs = 2, 3
            yidcs = 0, 1

        return xidcs, yidcs

    def independent_components_analysis(
            self, n_components=8, method="FastICA", plot=True,
            compare_meas2model=True
    ):
        r"""Peforms Independent Components Analysis (ICA).

        Calculates beta-functions and betatron phase-advance at the BPMs using
        ICA.

        ICA aims to identify the linear transformation (unmixing matrix)
        revealing statistically independent source signals. Just as in PCA,
        the beatron motion sine and cosine modes can be used to calculate
        beta-functions and BPMs phase advances.

        While PCA aims to identify the linear transformation revealing
        uncorrelated source signals, ICA seeks the transformation
        revealing statistically independent signals, a requirement much
        stronger than uncorrelatedness.

        ICA often performs better at blind source separation for linear
        mixtures of sinals with non-gaussian distributions, which is relevant
        for when several source signals have similar variance. This method
        thus is generally more robust at betatron motion identification when
        there are contaminating signals, bad acquisitions or similar variance
        between horizontal and vertical modes (partticularly relevant when
        betatron coupling is significant).

        ICA can be implemented with second-order blind source identification
        (SOBI) [1], based on simultaneous diagonalization of the time-shifted
        data covariance matrices or with information-theoretic
        approaches seeking the maximization of the statistical indpendence of
        the estimated source signals [2]. We use the latter, as implemented in
        scikit-learn's FastICA [3,4].

        The variance convention is the same as in PCA analysis: whiten source
        signals, with the mixing matrix containing the modes energy/variance
        [4].

        Args:
            n_components (int, optional): number of independent components to
            decompose the data

            plot (bool, optiional): whether to plot the analysis results.
            Defaults to True

            compare_meas2model (bool, optional): whether to plot measured and
            nominal beta-functions and BPMs phase-advance, as well as
            beta-beting and phase-advance errors or plot only beta-beating and
            phase-advance-errors. Defaults to True

        References:

        [1] Huang, X. Beam-Based Correction and Optimization for Accelerators,
            Section 5.2.3. CRC Press. 2020.

        [2] A. Hyvärinen, E. Oja. Independent component analysis: algorithms
            and applications. Neural Networks. Volume 13, Issues 4-5, 2000,
            Pages 411-430, https://doi.org/10.1016/S0893-6080(00)00026-5.

        [3] scikit-learn.decomposition.FastICA documentation.
            https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.FastICA.html

        [4] Scikit-learn examples. "Blind Source Separation using FastICA"
            https://scikit-learn.org/stable/auto_examples/decomposition/plot_ica_blind_source_separation.html#sphx-glr-auto-examples-decomposition-plot-ica-blind-source-separation-py
        """
        ica_data = dict()
        for pinger in self.params.pingers2kick:
            if pinger == "h":
                traj = self.trajx
                label = "x"
            else:
                traj = self.trajy
                label = "y"
            tunes = self.tunex + 49., self.tuney + 14.

            if self.model_optics is None:
                self._get_nominal_optics(tunes)

            # get model optics
            beta_model = self.model_optics["beta"+label]
            phase_model = self.model_optics["phase"+label]

            # perform Independent Component Analysis (ICA)
            if method == "FastICA":
                ica = _FastICA(
                    n_components=n_components,
                    whiten="unit-variance",
                    algorithm="parallel",
                    tol=1e-12
                )
            if method == "SOBI":
                ica = _SOBI(
                    n_components=n_components,
                    n_lags=5,
                    whiten="unit-variance",
                    isreal=True,
                    verbose=False,
                )
            # collect source signals & mixing matrix
            signals = ica.fit_transform(traj)
            mixing_matrix = ica.mixing_

            # determine betatron modes from mixing matrix
            # largest variance should be contained in the betatron modes
            idcs = _np.argsort(_np.std(mixing_matrix, axis=0))[-2:]
            sin_mode = mixing_matrix[:, idcs[0]]
            cos_mode = mixing_matrix[:, idcs[-1]]

            # determine which betatron mode is the sine & which is cosine
            # sine mode starts off close to zero
            if _np.abs(sin_mode[0]) > _np.abs(cos_mode[0]):
                cos_mode, sin_mode = sin_mode, cos_mode
                idcs[0], idcs[1] = idcs[1], idcs[0]

            # calculate beta function & phase from betatron modes
            beta, phase = self.get_beta_and_phase_from_betatron_modes(
                sin_mode, cos_mode, beta_model
            )

            # plot results
            if plot:
                self.plot_betabeat_and_phase_error(
                    beta_model, beta,
                    phase_model, phase,
                    title=f"ICA Analysis: beta{label} & phase{label}",
                    compare_meas2model=compare_meas2model,
                    bpms2use=self.bpms2use
                )

            # save results
            ica_data["source_signals_"+label] = signals
            ica_data["mixing_matrix_"+label] = mixing_matrix
            ica_data["beta"+label] = beta
            ica_data["phase"+label] = phase
        self.ica_data = ica_data

    def equilibrium_params_analysis(self):
        """."""
        raise NotImplementedError

    # plotting methods
    def plot_trajs_spectrum(
        self, bpm_index=0, trajx=None, trajy=None, title=""
    ):
        """."""
        trajx = self.trajx if trajx is None else trajx
        trajy = self.trajy if trajy is None else trajy

        trajx_spec, tunesx = self.calc_spectrum(trajx, fs=1)
        trajy_spec, tunesy = self.calc_spectrum(trajy, fs=1)

        fig, axs = _mplt.subplots(2, 1, figsize=(12, 8))

        axs[0].plot(
            tunesx, _np.abs(trajx_spec)[:, bpm_index])
        axs[1].plot(
            tunesy, _np.abs(trajy_spec)[:, bpm_index])

        if not title:
            title = "kicked beam trajectories spectrum \n"
            title += f"kicks = ({self.params.hkick},{self.params.vkick}) mrad"

        axs[0].set_title(title)

        axs[0].set_ylabel(r'$x$ [mm]')
        axs[1].set_ylabel(r'$y$ [mm]')
        axs[1].set_xlabel('tune')
        fig.tight_layout()

        return fig, axs

    def plot_trajs(self, bpm_index=0, timescale=0, compare_fit=False):
        """Plot trajectories and sum-signal at a given BPM and timescale.

        Args:
            bpm_index (int, optional): Which BPM's reading to show.
                Defaults to 0 (first BPM).
            timescale (int, optional): Turn-by-turn timescale, where:
                timescale = 0 : ~ 20 turns; harmonic motion
                timescale = 1 : ~ 200 turns; chromaticity decoherence
                    modulation
                timescale = 2 : ~ 2000 turns; transverse decoherence
                    modulations
            Defaults to 0.
            compare_fit (bool, optional): whether to plot acquisitions and the
            fitting and the fit residue. Defaults to False.

        Returns:
            fig, ax: matplotlib figure and axes
        """
        if self.fitting_data is None and compare_fit:
            msg = "No fitting was performed yet."
            msg += "Plotting measured data only."
            print(msg)
            compare_fit = False

        nr_pre = self.data['nrsamples_pre']
        nr_post = self.data['nrsamples_post']

        if not timescale:
            nmax_x, nmax_y = int(1 / self.tunex), int(1 / self.tuney)
            slicex = (nr_pre, nr_pre + nmax_x + 1)
            slicey = (nr_pre, nr_pre + nmax_y + 1)
            slicesum = (nr_pre, nr_pre + max(nmax_x, nmax_y) + 1)

        nmax = int(1 / self.SYNCH_TUNE)
        if timescale == 1:
            slicex = (nr_pre, nr_pre + nmax + 1)
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
        name = self.params.BPMS_NAMES[bpm_index]
        fig.suptitle(
            f"{self.acq_rate.upper()} acq. at BPM {bpm_index:03d} ({name})"
        )

        ax[0].set_title("horizontal trajectory")
        ax[0].plot(trajx, "-", mfc="none", color="blue", label="acq.")

        if compare_fit and "h" in self.params.pingers2kick:
            fit = self.fitting_data["trajx_final_fit"]
            res = self.fitting_data["fittingx_residue"]
            init, end = self.trajx_turns_slice
            ax[0].plot(
                _np.arange(init, end + 1, 1),
                fit[:, bpm_index],
                "x-", mfc="none", color="blue", label="fit"
            )
            ax[0].plot(
                _np.arange(init, end + 1, 1),
                res[:, bpm_index],
                "x-", mfc="none", color="green", label="residue"
            )
            ax[0].legend()
        ax[0].set_xlim(slicex)
        ax[0].set_ylabel("position [mm]")

        ax[1].set_title("vertical trajectory")
        ax[1].plot(trajy, "-", mfc="none", color="red", label="acq.")

        if compare_fit and "v" in self.params.pingers2kick:
            fit = self.fitting_data["trajy_final_fit"]
            res = self.fitting_data["fittingy_residue"]
            init, end = self.trajy_turns_slice
            ax[1].plot(
                _np.arange(init, end + 1, 1),
                fit[:, bpm_index],
                "x-", mfc="none", color="red", label="fit"
            )
            ax[1].plot(
                _np.arange(init, end + 1, 1),
                res[:, bpm_index],
                "x-", mfc="none", color="green", label="residue"
            )
            ax[1].legend()
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

    def plot_modal_analysis(self):
        """."""
        # TODO: reuse code from `principal_components_analysis`
        trajs = _np.concatenate((self.trajx, self.trajy), axis=1)

        u, s, vt = self.calc_svd(trajs, full_matrices=False)

        fig = _mplt.figure(figsize=(14, 12))

        gs = _gridspec.GridSpec(4, 2)

        svals = fig.add_subplot(gs[0, 0])
        var = fig.add_subplot(gs[1, 0])

        source1 = fig.add_subplot(gs[0, 1])
        source2 = fig.add_subplot(gs[1, 1], sharex=source1)

        spatial1 = fig.add_subplot(gs[2, 0])
        spatial2 = fig.add_subplot(gs[3, 0], sharex=spatial1)

        spec1 = fig.add_subplot(gs[2, 1])
        spec2 = fig.add_subplot(gs[3, 1], sharex=spec1, sharey=spec1)

        svals.plot(s, 'o', color='k', mfc='none')
        svals.set_title('singular values spectrum')
        svals.set_yscale("log")

        var.plot(_np.cumsum(s) / _np.sum(s), 'o', color="k", mfc='none')
        var.set_title("explained variance")
        var.set_xlabel("rank")

        source1.plot(u[:, 0], label='mode 0')
        source1.plot(u[:, 1], label='mode 1')
        source1.set_title("temporal modes - axis 1")
        source1.set_xlabel("turns index")
        source1.legend()

        source2.plot(u[:, 2], label='mode 2')
        source2.plot(u[:, 3], label='mode 3')
        source2.set_title("temporal modes - axis 2")
        source2.set_xlabel("turns index")
        source2.legend()

        spatial1.plot(vt.T[:, 0], label='mode 0')
        spatial1.plot(vt.T[:, 1], label='mode 1')
        spatial1.set_title("spatial modes - axis 1")
        spatial1.set_xlabel("BPMs index (H/V)")
        spatial1.legend()

        spatial2.plot(vt.T[:, 2], label='mode 2')
        spatial2.plot(vt.T[:, 3], label='mode 3')
        spatial2.set_title("spatial modes - axis 2")
        spatial2.set_xlabel("BPMs index (H/V)")
        spatial2.legend()

        freq, fourier = _np.fft.rfftfreq(n=u.shape[0]), _np.fft.rfft(u, axis=0)

        spec1.plot(
            freq, _np.abs(fourier)[:, 0], 'o-',
            color="C0", mfc="none", label='mode 0'
        )
        spec1.plot(
            freq, _np.abs(fourier)[:, 1], 'x-', color="C0", label='mode 1'
        )
        spec1.set_title("temporal modes spectrum - axis 1")
        spec1.set_xlabel("fractional tune")
        spec1.legend()

        spec2.plot(
            freq, _np.abs(fourier)[:, 2], 'o-',
            color="C1", mfc="none", label='mode 2'
        )
        spec2.plot(
            freq, _np.abs(fourier)[:, 3], 'x-', color="C1", label='mode 3'
        )
        spec2.set_title("temporal modes spectrum - axis 2")
        spec2.set_xlabel("fractional tune")
        spec2.legend()

        _mplt.tight_layout()

        _mplt.show()

    def plot_fit_comparison(self, bpm_index=0, timescale=0):
        """Plot comparison of fit vs. acqusitions.

        Args:
            bpm_index (int, optional): BPM at which to compare. Defaults to 0.
            timescale (int, optional): TbT timescale to plot.
                0 = harmonic TbT motion,
                1 = chromatic decoherence modulated motion,
                2 = amplitude decoherence modulated motion. Defaults to 0.

        Returns:
            fig, ax: figure env and axes containing the plots
        """
        return self.plot_trajs(
            bpm_index=bpm_index, timescale=timescale, compare_fit=True
            )

    def plot_rms_residue(self, axis=0):
        """Plot RMS fitting residue along BPMs or along Turns.

        Args:
            axis (int, optional): If 0, plot along BPMs. If 1, along the
            turns. Defaults to 0.
        """
        for pinger in self.params.pingers2kick:
            label = "x" if "h" in pinger else "y"
            _mplt.figure()
            along = "BPMs" if not axis else "Turns"
            _mplt.title(f"RMS fitting residue along {along}")
            _mplt.plot(
                self.fitting_data["fitting"+label+"_residue"].std(axis=axis),
                "o-", mfc='none'
            )
            _mplt.xlabel(f"{along} index")
            _mplt.ylabel("residue [mm]")

    def plot_betabeat_and_phase_error(
        self, beta_model, beta_meas, phase_model, phase_meas, title=None,
        compare_meas2model=False, bpms2use=None
    ):
        """."""
        # TODO: plot error bars if they are available
        beta_model, beta_meas = beta_model.copy(), beta_meas.copy()
        phase_model, phase_meas = phase_model.copy(), phase_meas.copy()

        if bpms2use is not None:
            beta_model[~bpms2use] = _np.nan
            beta_meas[~bpms2use] = _np.nan
            phase_model[~bpms2use] = _np.nan
            phase_meas[~bpms2use] = _np.nan

        if compare_meas2model:
            fig, axs = _mplt.subplots(2, 2, figsize=(15, 10))
        else:
            fig, axs = _mplt.subplots(1, 2, figsize=(15, 5))

        if title is not None:
            fig.suptitle(title)
        else:
            fig.suptitle("beta and phase")

        # Beta plots
        if compare_meas2model:
            ax_beta = axs[0, 0]
            ax_beta.plot(
                beta_model, "o-", label="Model", mfc="none"
            )
            ax_beta.plot(
                beta_meas, "o--", label="Meas", mfc="none"
            )
            ax_beta.set_ylabel("beta function")
            ax_beta.legend()

        # Beta beating plot
        ax_beat = axs[0, 1] if compare_meas2model else axs[0]
        beta_beat = (beta_model - beta_meas) / beta_model
        beta_beat *= 100  # [%]
        ax_beat.plot(
            beta_beat,
            "o-",
            label=f"rms = {beta_beat[~_np.isnan(beta_beat)].std():.2f} %",
            mfc="none",
        )
        ax_beat.set_ylabel("beta beating [%]")
        ax_beat.legend()

        # Phase advance plot
        if compare_meas2model:
            model_phase_advance = _np.diff(phase_model)
            meas_phase_advance = _np.abs(_np.diff(phase_meas))

            ax_phase = axs[1, 0]
            ax_phase.plot(
                model_phase_advance, "o-", label="Model", mfc="none"
            )
            ax_phase.plot(
                meas_phase_advance, "o--", label="Meas", mfc="none"
            )
            ax_phase.set_ylabel("BPMs phase advance [rad]")
            ax_phase.legend()

        # Phase advance error plot
        ax_phase_err = axs[1, 1] if compare_meas2model else axs[1]
        rph_err = model_phase_advance - meas_phase_advance
        rph_err /= model_phase_advance
        rph_err *= 100  # [%]
        ax_phase_err.plot(
            rph_err,
            "o-",
            label=f"rms.err={rph_err[~_np.isnan(rph_err)].std():.2f}",
            mfc="none"
        )
        ax_phase_err.set_ylabel("BPMs fractional phase advance error [rad]")
        ax_phase_err.legend()

        fig.supxlabel("BPM index")
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

        As in Eq. (9) of Ref:

        X.R. Resende, M.B. Alves, L. Liu, and F.H. de Sá, “Equilibrium and
        Nonlinear Beam Dynamics Parameters From Sirius Turn-by-Turn BPM Data”,
        in Proc. IPAC'21, Campinas, SP, Brazil, May 2021, pp. 1935-1938.
        doi:10.18429/JACoW-IPAC2021-TUPAB219
        """
        action = _np.sum(amplitudes**4)
        action /= _np.sum(amplitudes**2 * nominal_beta)
        beta = amplitudes**2 / action
        return beta, action

    def calc_beta_and_phase_with_pca(self, matrix, beta_model=None):
        """."""
        _, svals, vtmat = self.calc_svd(matrix, full_matrices=False)

        if beta_model is None:
            phase_meas = _np.arctan2(
                svals[1] * vtmat[1, :], svals[0] * vtmat[0, :]
            )
            phase_meas = (_np.unwrap(phase_meas))
            return phase_meas

        beta_meas = (
            svals[0] ** 2 * vtmat[0, :] ** 2 + svals[1] ** 2 * vtmat[1, :] ** 2
        )
        beta_meas /= _np.std(beta_meas) / _np.std(beta_model)

        return beta_meas, phase_meas

    def get_beta_and_phase_from_betatron_modes(
        self, sin_mode, cos_mode, beta_model
    ):
        """Calulate beta & phase at BPMs from sine and cosine modes.

        Reference: X. Huang. Beam-based correction and optimization for
        accelerators. Eq. 5.44, pg. 139.

        Args:
            sin_mode (160-array): sine mode of mixing matrix
            cos_mode (160-array): cosine mode of mixing matrix
            beta_model (160-array): nominal beta function, used for
            determining the measured beta function scaling factor

        Returns:
            beta: 160-array of measured beta function at BPMs
            phase: 160-array of betatron phase at BPMs

        """
        beta = (sin_mode**2 + cos_mode**2)
        beta /= _np.std(beta) / _np.std(beta_model)
        phase = _np.arctan2(sin_mode, cos_mode)
        phase = _np.unwrap(phase, discont=_np.pi)
        return beta, phase

    def _get_nominal_optics(self, tunes=None, chroms=None):
        r"""Gets nominal lattice optics functions and phase-advances.

        Args:
            tunes (tuple, optional): (nux, nuy) - betatron tunes fractional
            part. Defaults to None, in which case (0.16, 0.22) is used.
            chroms (tuple, optional): (\chix, \chiy) - machine chromaticity.
            Defaults to None, in which case (2.5, 2.5) is used.
        """
        if self.model_optics is None:
            model_optics = dict()
            model = _si.create_accelerator()
            model = _si.fitted_models.vertical_dispersion_and_coupling(model)
            model.radiation_on = False
            model.cavity_on = False
            model.vchamber_on = False

            if tunes is not None:
                tunecorr = _TuneCorr(model, acc="SI")
                tunecorr.correct_parameters(goal_parameters=tunes)

            chroms = (3.8, 3.1) if chroms is None else chroms
            chromcorr = _ChromCorr(model, acc="SI")
            chromcorr.correct_parameters(goal_parameters=chroms)

            famdata = _si.get_family_data(model)
            # TODO: add also EdTeng optics calculations
            twiss, *_ = _pa.optics.calc_twiss(
                accelerator=model, indices="open"
            )
            bpms_idcs = _pa.lattice.flatten(famdata["BPM"]["index"])
            model_optics["betax"] = twiss.betax[bpms_idcs]
            model_optics["phasex"] = twiss.mux[bpms_idcs]
            model_optics["betay"] = twiss.betay[bpms_idcs]
            model_optics["phasey"] = twiss.muy[bpms_idcs]
            self.model_optics = model_optics
