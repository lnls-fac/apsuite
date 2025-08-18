"""."""

import datetime as _datetime
import time as _time
import warnings

import matplotlib.gridspec as _gridspec
import matplotlib.pyplot as _mplt
import numpy as _np
import pyaccel as _pa
from pymodels import si as _si
from scipy.optimize import curve_fit as _curve_fit
from siriuspy.devices import PowerSupplyPU as _PowerSupplyPU, \
     Trigger as _Trigger
from siriuspy.sofb.csdev import SOFBFactory
from sklearn.decomposition import FastICA as _FastICA

from ..optics_analysis import ChromCorr as _ChromCorr, TuneCorr as _TuneCorr
from .meas_bpms_signals import AcqBPMsSignals as _AcqBPMsSignals, \
    AcqBPMsSignalsParams as _AcqBPMsSignalsParams
from ..utils import ThreadedMeasBaseClass as _ThreadBaseClass


class TbTDataParams(_AcqBPMsSignalsParams):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.signals2acq = "XYS"
        self.acq_rate = "TbT"
        self.nrpoints_before = 50
        self.nrpoints_after = 750
        self.restore_init_state = True
        self.do_pulse_evg = False

        self.mags_strength_rtol = 0.05
        self._pingers2kick = ""  # 'h', 'v' or 'hv'
        self.hkick = 0  # [mrad]
        self.vkick = 0  # [mrad]
        self.trigpingh_delay_raw = 36802990  # defined @ 2024-05-21 mach. study
        self.trigpingv_delay_raw = 36802937  # defined @ 2024-05-21 mach. study
        self.magnets_timeout = 120.0

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
        stg += ftmp("mags_strength_rtol", self.mags_strength_rtol, "")
        if self.hkick is None:
            stg += stmp("hkick", "same", "(current value will not be changed)")
        else:
            stg += ftmp("hkick", self.hkick, "[mrad]")

        dly = self.trigpingh_delay_raw
        if dly is None:
            stg += stmp(
                "trigpingh_delay_raw",
                "same",
                "(current value will not be changed)",
            )
        else:
            stg += dtmp("trigpingh_delay_raw", dly, "")
        if self.vkick is None:
            stg += stmp("vkick", "same", "(current value will not be changed)")
        else:
            stg += ftmp("vkick", self.vkick, "[mrad]")
        dly = self.trigpingv_delay_raw
        if dly is None:
            stg += stmp(
                "trigpingv_delay_raw",
                "same",
                "(current value will not be changed)",
            )
        else:
            stg += dtmp("trigpingv_delay_raw", dly, "")
        stg += stmp("restore_init_state", self.restore_init_state, "")
        return stg

    @property
    def pingers2kick(self):
        """."""
        return self._pingers2kick

    @pingers2kick.setter
    def pingers2kick(self, val):
        """Must be 'h', 'v' or 'vh'/'hv'."""
        self._pingers2kick = str(val).lower()


class MeasureTbTData(_ThreadBaseClass, _AcqBPMsSignals):
    """."""

    PINGERH_TRIGGER = "SI-01SA:TI-InjDpKckr"
    PINGERV_TRIGGER = "SI-19C4:TI-PingV"

    def __init__(self, isonline=False):
        """."""
        _ThreadBaseClass.__init__(
            self,
            params=TbTDataParams(),
            target=self._do_measurement,
            isonline=isonline
        )
        _AcqBPMsSignals.__init__(
            self,
            params=self.params,
            isonline=isonline,
            ispost_mortem=False
        )

        self._meas_finished_ok = True
        self._init_timing_state = dict()
        self._init_magnets_state = dict()

    @property
    def init_timing_state(self):
        """."""
        return self._init_timing_state

    @property
    def init_magnets_state(self):
        """."""
        return self._init_magnets_state

    def create_devices(self):
        """."""
        _AcqBPMsSignals.create_devices(self)
        self.devices["pingh"] = _PowerSupplyPU(
            _PowerSupplyPU.DEVICES.SI_INJ_DPKCKR
        )
        self.devices["trigpingh"] = _Trigger(self.PINGERH_TRIGGER)
        self.devices["pingv"] = _PowerSupplyPU(
            _PowerSupplyPU.DEVICES.SI_PING_V
        )
        self.devices["trigpingv"] = _Trigger(self.PINGERV_TRIGGER)

    def get_timing_state(self):
        """."""
        # BPMs trigger and EVG timing state
        state_dict = _AcqBPMsSignals.get_timing_state(self)

        # Pingers trigger timing state
        trigs = self.devices["trigpingh"], self.devices["trigpingv"]
        keys = "h", "v"

        for key, trig in zip(keys, trigs) :
            state_dict[f"trigping{key}_state"] = trig.state  # enable state
            state_dict[f"trigping{key}_source"] = trig.source_str
            state_dict[f"trigping{key}_delay_raw"] = trig.delay_raw

        return state_dict

    def prepare_timing(self, state=None):
        """."""
        print("Configuring BPMs timing...")
        _AcqBPMsSignals.prepare_timing(self, state)  # BPM trigger timing

        state = dict() if state is None else state
        print("Configuring magnets timing...")
        prms = self.params
        pingers2kick = prms.pingers2kick
        trigpingh = self.devices["trigpingh"]
        trigpingv = self.devices["trigpingv"]

        if ("h" in pingers2kick) or (state.get("trigpingh_state", 0) == 1):
            trigh_ok = trigpingh.cmd_enable(timeout=prms.timeout)
            trigpingh.source = state.get("trigpingh_source", prms.timing_event)
            dly = state.get("trigpingh_delay_raw", prms.trigpingh_delay_raw)
            if dly is not None:
                trigpingh.delay_raw = dly
        else:
            trigh_ok = trigpingh.cmd_disable(timeout=prms.timeout)
        print(f"\tpingh trigger configured: {trigh_ok}")

        if ("v" in pingers2kick) or (state.get("trigpingv_state", 0) == 1):
            trigv_ok = trigpingv.cmd_enable(timeout=prms.timeout)
            trigpingv.source = state.get("trigpingv_source", prms.timing_event)
            dly = state.get("trigpingv_delay_raw", prms.trigpingv_delay_raw)
            if dly is not None:
                trigpingv.delay_raw = dly
        else:
            trigv_ok = trigpingv.cmd_disable(timeout=prms.timeout)
        print(f"\tpingv trigger configured: {trigv_ok}")

        return trigh_ok and trigv_ok

    def get_magnets_strength(self):
        """."""
        pingh_str = self.devices["pingh"].strength
        pingv_str = self.devices["pingv"].strength

        return pingh_str, pingv_str

    def get_magnets_state(self):
        """."""
        state = dict()
        pings = self.devices["pingh"], self.devices["pingv"]
        keys = "h", "v"
        pingh_str, pingv_str = self.get_magnets_strength()
        for key, ping in zip(keys, pings):
            ping_str = pingh_str if key == "h" else pingv_str
            state[f"ping{key}_pwr"] = ping.pwrstate
            state[f"ping{key}_pulse"] = ping.pulse
            state[f"ping{key}_strength"] = ping_str
            state[f"ping{key}_voltage"] = ping.voltage
        return state

    def set_magnets_state(self, state, wait_mon=True):
        """Set magnets strengths, pwr and pulse states."""
        timeout = self.params.magnets_timeout
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]

        print("Setting magnets states...")
        pingh_ok, pingv_ok = True, True
        # make sure mags are on to set their strengths
        if "h" in self.params.pingers2kick:
            pingh_ok = pingh.cmd_turn_on(timeout=timeout)
        if "v" in self.params.pingers2kick:
            pingv_ok = pingv.cmd_turn_on(timeout=timeout)

        if not pingh_ok or not pingv_ok:
            print("\tpingers could not be turned-on for setting strengths.")
            print(f"\t\tpingh-on: {pingh_ok}; pingv_on: {pingv_ok}")
            return False

        # turn-off pulse before changing strengths
        # prevent accidental activation
        pingh_ok = pingh.cmd_turn_off_pulse(timeout=timeout)
        pingv_ok = pingv.cmd_turn_off_pulse(timeout=timeout)

        if not pingh_ok or not pingv_ok:
            msg = "\tpingers pulse-status could not be turned-off "
            msg += "for securely setting the strengths."
            print(msg)
            print(f"\t\tpingh pulse: {pingh_ok}; pingv pulse: {pingv_ok}")
            return False

        pingh_ok, pingv_ok = self.set_magnets_strength(
            hkick=state.get("pingh_strength", None),
            vkick=state.get("pingv_strength", None),
            wait_mon=wait_mon
        )

        if not pingh_ok or not pingv_ok:
            print("\tpingers strenghts could not be set.")
            return False

        if not state.get("pingh_pwr", True):
            pingh_ok = pingh.cmd_turn_off(timeout=self.params.magnets_timeout)

        if not state.get("pingv_pwr", True):
            pingv_ok = pingv.cmd_turn_off(timeout=self.params.magnets_timeout)

        if not pingh_ok or not pingv_ok:
            print("\tfailed at setting magnets power-state")
            return False

        if state.get("pingh_pulse", False):
            pingh_ok = pingh.cmd_turn_on_pulse(timeout)
        else:
            pingh_ok = pingh.cmd_turn_off_pulse(timeout)
        if not pingh_ok:
            print("\tpingh pulse-status could not be set")

        if state.get("pingv_pulse", False):
            pingv_ok = pingv.cmd_turn_on_pulse(timeout)
        else:
            pingv_ok = pingv.cmd_turn_off_pulse(timeout)
        if not pingv_ok:
            print("\tpingv pulse-status could not be set")

        return pingh_ok and pingv_ok

    def set_magnets_strength(
        self, hkick=None, vkick=None, magnets_timeout=None, wait_mon=True
    ):
        """Set pingers strengths, check if was set & indicate which failed."""
        pingh, pingv = self.devices["pingh"], self.devices["pingv"]
        pingh_ok, pingv_ok = True, True

        # first, strengths are only set. No waiting reaching the desired values
        # (i.e. timeout = 0)
        if hkick is not None:
            print(f"\tSetting pingh strength to {hkick:.3f} mrad...")
            pingh.strength = hkick

        if vkick is not None:
            print(f"\tSetting pingv strength to {vkick:.3f} mrad...")
            pingv.strength = vkick

        if not wait_mon:
            return pingh_ok, pingv_ok

        # once strenghts are set and wait_mon is True
        # we wait the magnets reach the values set

        magnets_timeout = magnets_timeout or self.params.magnets_timeout
        t0 = _time.time()

        if hkick is not None:
            print("\t\twaiting pingh reach strength set")
            pingh_ok = pingh.set_strength(
                hkick,
                tol=self.params.mags_strength_rtol * abs(hkick),
                timeout=magnets_timeout,
                wait_mon=wait_mon,
            )

        elapsed_time = _time.time() - t0
        magnets_timeout -= elapsed_time

        if vkick is not None:
            print("\t\twaiting pingv reach strength set")
            pingv_ok = pingv.set_strength(
                vkick,
                tol=self.params.mags_strength_rtol * abs(vkick),
                timeout=magnets_timeout,
                wait_mon=wait_mon,
            )

        bad_pingers = []
        if not pingh_ok:
            bad_pingers.append("pingh")
        if not pingv_ok:
            bad_pingers.append("pingv")

        if bad_pingers:
            msg = "\tSome magnet strengths were not set.\n"
            msg += f"\t\tBad pingers: {', '.join(bad_pingers)}"
            print(msg)

        print("\tStrengths all set.")
        return pingh_ok, pingv_ok

    def prepare_magnets(self):
        """Create magnets state dict from params."""
        print("Preparing magnets...")
        pingers2kick = self.params.pingers2kick
        state = dict()
        if "h" in pingers2kick:
            state["pingh_pwr"] = 1  # always make sure its on
            state["pingh_pulse"] = 1
            state["pingh_strength"] = self.params.hkick
        else:
            # state["pingh_pwr"] = 0  # but will not be turning-off.
            state["pingh_pulse"] = 0  # only changing pulse-sts

        if "v" in pingers2kick:
            state["pingv_pwr"] = 1
            state["pingv_pulse"] = 1
            state["pingv_strength"] = self.params.vkick
        else:
            # state["pingv_pwr"] = 0
            state["pingv_pulse"] = 0

        return self.set_magnets_state(state)

    def _do_measurement(self):
        """."""
        init_timing_state = self._init_timing_state
        init_magnets_state = self._init_magnets_state

        timing_ok = self.prepare_timing()
        if not timing_ok:
            print("Failed at configuring timing! Exiting.")
            self._meas_finished_ok = False
            self._restore_and_exit(timing_state=init_timing_state)

        print("Timing was succesfully configured.")

        if self._stopevt.is_set():
            print("Measurement stopped.")
            self._meas_finished_ok = False
            self._restore_and_exit(timing_state=init_timing_state)

        mags_ok = self.prepare_magnets()  # gets strengths from params
        if not mags_ok:
            print("Failed at configuring magnets! Exiting.")
            self._meas_finished_ok = False
            self._restore_and_exit(init_timing_state, init_magnets_state)

        print("Magnets were succesfully configured.")

        if self._stopevt.is_set():
            print("Measurement stopped.")
            self._meas_finished_ok = False
            self._restore_and_exit(init_timing_state, init_magnets_state)

        try:
            self.acquire_data()
            print("Acquisition was successful.")
        except Exception as e:
            self._meas_finished_ok = False
            print(f"An error occurred during acquisition: {e}")
            self._restore_and_exit(init_timing_state, init_magnets_state)

        if self.params.restore_init_state:
            self._restore_and_exit(init_timing_state, init_magnets_state)

        print("Measurement finished.")

    def acquire_data(self):
        """."""
        curr0 = self.devices["currinfo"].current
        _AcqBPMsSignals.acquire_data(self)
        self.data["current_before"] = curr0

    def get_data(self):
        """."""
        data = _AcqBPMsSignals.get_data(self)
        data["magnets_state"] = self.get_magnets_state()
        data["current_after"] = data.pop("stored_current")
        data["init_magnets_state"] = self._init_magnets_state
        data["init_timing_state"] = self._init_timing_state
        return data

    def update_initial_states(self):
        """."""
        if not self.connected:
            raise RuntimeError("Not connected yet!")
        self._init_timing_state = self.get_timing_state()
        self._init_magnets_state = self.get_magnets_state()

    def _restore_and_exit(self, timing_state=None, magnets_state=None):
        """Restore timing and magnets state and exit the measurement."""
        print("Restoring machine initial state.")

        if timing_state is not None:
            timing_ok = self.prepare_timing(timing_state)
            print(f"\tTiming restored: {timing_ok}")

        if magnets_state is not None:
            mags_ok = self.set_magnets_state(magnets_state, wait_mon=False)
            # no need to wait_mon if exiting anyways. wait_mon is critical to
            # prevent meas starting before magnets reach the required strentgh
            print(f"\tMagnets restored: {mags_ok}")

    def check_machine_restored(self):
        """."""
        if self.ismeasuring:
            print("Measurement not finished.")
            return False

        magnets_state = self.get_magnets_state()
        timing_state = self.get_timing_state()
        mags = self._init_magnets_state == magnets_state
        timing = self._init_timing_state == timing_state

        print(f"Magnets restored:\t{mags}")
        print(f"Timing restored:\t{timing}")

        return mags and timing

    def check_measurement_finished_ok(self):
        """Check if measument finished without errors.

        Returns:
            bool: True if measurement finished without errors.

        """
        if self.ismeasuring:
            print("Measurment not finished.")
            return False
        return self._meas_finished_ok

    def check_measurement_quality(self):
        """."""
        raise NotImplementedError

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
    PINGH_CALIBRATION = 1.5416651659146232  # pingers calibration factors
    PINGV_CALIBRATION = 1.02267573  # should be in the mags exc. curve
    SYNCH_TUNE = 0.004713  # check this

    def __init__(self, filename="", isonline=False):
        """Analysis of linear optics using Turn-by-turn data."""
        super().__init__(isonline=isonline)

        self._fname = None
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
        self._bpms_names = None

        if filename:
            self.load_and_apply(filename)

    def __str__(self):
        """."""
        stg = ""
        data = self.data
        if not data:
            return stg

        stg += "\n"
        stg += "Measurement data:\n"

        ftmp = "{0:26s} = {1:9.6f}  {2:s}\n".format
        stmp = "{0:26s} = {1:9}  {2:s}\n".format
        dtmp = "{0:26s} = {1:9d}  {2:s}\n".format
        gtmp = "{0:<15s} = {1:}  {2:}\n".format

        stg += gtmp("timestamp", self.timestamp, "")  # TODO: convert tmstp
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

        stg += stmp("event_mode", data["timing_state"]["event_mode"], "")
        stg += dtmp(
            "event_delay_raw",
            int(data["timing_state"]["event_delay_raw"]),
            ""
        )
        stg += "\n"
        stg += "BPMs state\n"
        stg += "\n"

        stg += stmp("acq_rate", data["acq_rate"], "")
        stg += stmp("nrsamples_pre", data["nrsamples_pre"], "")
        stg += stmp("nrsamples_post", data["nrsamples_post"], "")
        stg += stmp("switching_mode", data["switching_mode"], "")
        stg += stmp("switching_frequency", data["switching_frequency"], "")
        stg += stmp(
            "trigbpm_source", data["timing_state"]["trigbpm_source"], ""
        )
        stg += dtmp(
            "trigbpm_nrpulses",
            data["timing_state"]["trigbpm_nrpulses"],
            "",
        )
        stg += dtmp(
            "trigbpm_delay_raw",
            int(data["timing_state"]["trigbpm_delay_raw"]),
            ""
        )

        stg += "\n"
        stg += "Pingers state\n"
        stg += "\n"

        stg += stmp(
            "trigpingh_state", data["timing_state"]["trigpingh_state"], ""
        )
        stg += stmp(
            "trigpingh_source",
            data["timing_state"]["trigpingh_source"],
            "",
        )
        # stg += dtmp(
        #     "trigpingh_delay_raw",
        #     data["timing_state"]["trigpingh_delay_raw"],
        #     ""
        # )
        stg += stmp("pingh_pwr", data["magnets_state"]["pingh_pwr"], "")

        stg += stmp(
            "pingh_pulse", data["magnets_state"]["pingh_pulse"], ""
        )
        stg += ftmp("hkick", data["magnets_state"]["pingh_strength"], "mrad")
        stg += "\n"

        stg += stmp(
            "trigpingv_state", data["timing_state"]["trigpingv_state"], ""
        )
        stg += stmp(
            "trigpingv_source",
            data["timing_state"]["trigpingv_source"],
            "",
        )
        # stg += dtmp(
        #     "trigpingv_delay_raw",
        #     data["timing_state"]["trigpingv_delay_raw"],
        #     ""
        # )
        stg += stmp("pingv_pwr", data["magnets_state"]["pingv_pwr"], "")

        stg += stmp(
            "pingv_pulse", data["magnets_state"]["pingv_pulse"], ""
        )
        stg += ftmp("vkick", data["magnets_state"]["pingv_strength"], "mrad")

        return stg

    @property
    def fname(self):
        """."""
        return self._fname

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
        try:
            keys = super().load_and_apply(fname)
        except Exception:
            print('Problem loading file {fname}.')
            return
        self._fname = fname

        if keys:
            print("The following keys were not used:")
            print("     ", str(keys))

        # make often used data attributes
        data = self.data
        epoch_tmstp = data.get("timestamp", None)
        timestamp = (
            _datetime.datetime.fromtimestamp(epoch_tmstp)
            if epoch_tmstp is not None
            else None
        )
        self.timestamp = timestamp
        trajx = data.get("orbx", None).copy() * 1e-3
        trajy = data.get("orby", None).copy() * 1e-3
        trajsum = data.get("sumdata", None).copy()

        if trajsum is not None:
            self.trajsum = trajsum

        # zero mean in samples dimension
        if trajx is not None:
            trajx -= trajx.mean(axis=0)[None, :]
            self.trajx = trajx
        if trajy is not None:
            trajy -= trajy.mean(axis=0)[None, :]
            self.trajy = trajy

        self.nrsamples_pre = data.get("nrsamples_pre", None)
        self.nrsamples_post = data.get("nrsamples_post", None)
        self.tunex = data.get("tunex", None)
        self.tuney = data.get("tuney", None)
        self.acq_rate = data.get("acq_rate", None)
        self.rf_freq = data.get("rf_frequency", None)
        self.sampling_freq = data.get("sampling_frequency", None)
        self.switching_freq = data.get("switching_frequency", None)

    def linear_optics_analysis(
        self, method="PCA", compare_meas2model=True, **kwargs
    ):
        """Linear optics (beta-beating & phase adv. errors) analysis.

        Determines beta-functions and phase-advances on BPMs via sinusoidal
        fitting of TbT data in the harmonic motion timescale, as well as via
        spatio-temporal modal analysis using Principal Components Analysis
        (PCA) and Independent Components Analysis (ICA).
        """
        if method.lower() == "fitting":
            # TODO: avoid recalculating if already available
            self.harmonic_analysis(calc_optics=True, **kwargs)
            data = self.fitting_data
        elif method.lower() == "pca":
            self.principal_components_analysis(calc_optics=True)
            data = self.pca_data
        elif method.lower() == "ica":
            self.independent_components_analysis(calc_optics=True, **kwargs)
            data = self.ica_data

        betay = data.get("betay", None)
        betax = data.get("betax", None)
        phasex = data.get("phasex", None)
        phasey = data.get("phasey", None)

        betax_model = self.model_optics.get("betax", None)
        betay_model = self.model_optics.get("betay", None)
        phasex_model = self.model_optics.get("phasex", None)
        phasey_model = self.model_optics.get("phasey", None)

        if betax is not None:
            self.plot_betabeat_and_phase_error(
                betax_model,
                betax,
                phasex_model,
                phasex,
                title=f"{method} Optics Analysis: betax & phasex",
                compare_meas2model=compare_meas2model,
                bpms2use=self.bpms2use,
            )
        if betay is not None:
            self.plot_betabeat_and_phase_error(
                betay_model,
                betay,
                phasey_model,
                phasey,
                title=f"{method} Optics Analysis: betay & phasey",
                compare_meas2model=compare_meas2model,
                bpms2use=self.bpms2use,
            )

    def harmonic_analysis(self, calc_optics=True, guess_tunes=False):
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
            residue = traj - final_fit

            # store fitting data
            fitting_data["tune" + label] = tune
            fitting_data["amp" + label] = amps
            fitting_data["phase" + label] = phases_fit
            fitting_data["tune_err" + label] = params_error[0]
            fitting_data["amp" + label + "_err"] = params_error[1 : nbpms + 1]
            fitting_data["phase" + label + "_err"] = params_error[-nbpms:]
            fitting_data["traj" + label + "_init_fit"] = initial_fit
            fitting_data["traj" + label + "_final_fit"] = final_fit
            fitting_data["fitting" + label + "_residue"] = residue

            if not calc_optics:
                self.fitting_data = fitting_data
                return

            if self.model_optics is None:
                self._get_nominal_optics(tunes=(tunex + 49.0, tuney + 14.0))

            beta_model = self.model_optics["beta" + label]

            # fit beta-function & action from fitted amplitudes & nominal beta
            beta_fit, action = self.calc_beta_and_action(amps, beta_model)
            fitting_data["beta" + label] = beta_fit
            # TODO: propagate amplitude errors to beta errors
            fitting_data["action" + label] = action
            # TODO: propagate amplitude errors to action error

        self.fitting_data = fitting_data

    def principal_components_analysis(self, calc_optics=True):
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
        pca_data = dict()

        traj = _np.concatenate((self.trajx, self.trajy), axis=1)
        # perform PCA via SVD of history matrix
        umat, svals, vtmat = self.calc_svd(traj, full_matrices=False)

        # collect source signals and mixing matrix
        signals = umat * _np.sqrt(traj.shape[0] - 1)  # data signals
        mixing_matrix = vtmat.T @ _np.diag(svals)
        mixing_matrix /= _np.sqrt(traj.shape[0] - 1)

        # calulate tunes of the source signals
        spec0, tunes0 = self.calc_spectrum(signals[:, 0], axis=0)
        spec1, tunes1 = self.calc_spectrum(signals[:, 1], axis=0)
        spec2, tunes2 = self.calc_spectrum(signals[:, 2], axis=0)
        spec3, tunes3 = self.calc_spectrum(signals[:, 3], axis=0)

        tune0 = tunes0[_np.argmax(_np.abs(spec0))]
        tune1 = tunes1[_np.argmax(_np.abs(spec1))]
        tune2 = tunes2[_np.argmax(_np.abs(spec2))]
        tune3 = tunes3[_np.argmax(_np.abs(spec3))]

        # identify which signal is which (x or y)
        xidcs, yidcs = self.identify_modes_planes(
            tunes=[tune0, tune1, tune2, tune3],
            tunex=self.tunex,
            tuney=self.tuney,
        )

        # extract betatron modes fom mixing matrix
        xspatial_modes = mixing_matrix[:, xidcs]
        xtemp_modes = signals[:, xidcs]

        yspatial_modes = mixing_matrix[:, yidcs]
        ytemp_modes = signals[:, yidcs]

        # determine sin and cos modes temporal modes
        # sin/cos temporal modes correspond to cos/sin spatial modes
        # spatial modes are used for calc. optics
        # as in Xiaobiao's book , eqs. 5.31-5.32 (PCA), 5.55 (ICA)
        sinx_idx, cosx_idx = self.identify_sin_cos_modes(
            sources=xtemp_modes,
            plane="hor",
        )
        siny_idx, cosy_idx = self.identify_sin_cos_modes(
            sources=ytemp_modes,
            plane="ver",
        )

        sin_temp_modex = xtemp_modes[:, sinx_idx]  # cos/sin spatial mode
        cos_temp_modex = xtemp_modes[:, cosx_idx]  # corresponds to
        sin_spatial_modex = xspatial_modes[:, cosx_idx]  # sin/cos temp.
        cos_spatial_modex = xspatial_modes[:, sinx_idx]

        sin_temp_modey = ytemp_modes[:, siny_idx]
        cos_temp_modey = ytemp_modes[:, cosy_idx]
        sin_spatial_modey = yspatial_modes[:, cosy_idx]
        cos_spatial_modey = yspatial_modes[:, siny_idx]

        pca_data["singular_values"] = svals
        pca_data["source_signals"] = signals
        pca_data["mixing_matrix"] = mixing_matrix
        pca_data["xidcs"] = xidcs
        pca_data["yidcs"] = yidcs
        pca_data["sin_spatial_modex"] = sin_spatial_modex
        pca_data["cos_spatial_modex"] = cos_spatial_modex
        pca_data["sin_temp_modex"] = sin_temp_modex
        pca_data["cos_temp_modex"] = cos_temp_modex
        pca_data["sin_spatial_modey"] = sin_spatial_modey
        pca_data["cos_spatial_modey"] = cos_spatial_modey
        pca_data["sin_temp_modey"] = sin_temp_modey
        pca_data["cos_temp_modey"] = cos_temp_modey

        if not calc_optics:
            self.pca_data = pca_data
            return

        if self.model_optics is None:
            tunes = self.tunex + 49.0, self.tuney + 14.0
            self._get_nominal_optics(tunes)

        betax_model = self.model_optics["betax"]
        betay_model = self.model_optics["betay"]

        # calculate beta function & phase from spatial betatron modes
        betax, phasex = self.get_beta_and_phase_from_betatron_modes(
            sin_spatial_modex[:160], cos_spatial_modex[:160], betax_model
        )
        betay, phasey = self.get_beta_and_phase_from_betatron_modes(
            sin_spatial_modey[160:], cos_spatial_modey[160:], betay_model
        )

        # calculate signal variance
        signalx = _np.sqrt(_np.sum(svals[xidcs] ** 2))
        signaly = _np.sqrt(_np.sum(svals[yidcs] ** 2))
        # and signal noise
        noise = _np.sqrt(_np.sum(svals[3:] ** 2))
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

        pca_data["betax"] = betax
        pca_data["betay"] = betay
        pca_data["phasex"] = phasex
        pca_data["phasey"] = phasey
        pca_data["snrx"] = snrx
        pca_data["snry"] = snry
        pca_data["betax_err"] = betax_error
        pca_data["betay_err"] = betay_error
        pca_data["phasex_err"] = phasex_error
        pca_data["phasex_err"] = phasey_error
        self.pca_data = pca_data

    def identify_modes_planes(self, tunes, tunex, tuney):
        """Identify the x and y betatron modes idcs in PCA/ICA decomposition.

        When calculating the mixing matrix via PCA or ICA, the hor. and ver.
        betatron modes need the be determined. PCA sorts the modes with
        increasing variance (singular values), while ICA sorts the modes
        arbitrairly. By calculating the tune of the source signals
        corresponding to a given betatron mode and comparing it to the
        reference horizontal and vertical tunes, the hortizontal and vertical
        beatron modes can be identified.

        Args:
            tunes (list, tuple): calculated tunes for the candidate modes
            tunex (float): hor. tune
            tuney (float): ver. tune

        Returns:
            xidcs, yidcs: arrays containig the hor./ver. modes indices for PCA/
                ICA decompositions.
        """
        tunes = _np.array(tunes)[None, :]
        diff = _np.array([tunex, tuney])[:, None] - tunes
        idcs = _np.argmin(_np.abs(diff), axis=0).astype(bool)
        xidcs = _np.argwhere(~idcs).squeeze()
        yidcs = _np.argwhere(idcs).squeeze()
        return xidcs, yidcs

    def identify_sin_cos_modes(self, sources=None, plane="hor"):
        """Identify which TEMPORAl modes are sin/cos modes."""
        if plane.lower() == "hor":
            init, fin = self.trajx_turns_slice
            tune = self.tunex
        else:
            init, fin = self.trajy_turns_slice
            tune = self.tuney

        n = _np.arange(fin-init)
        sin = _np.sin(2 * _np.pi * tune * n)[:, None]
        cos = _np.cos(2 * _np.pi * tune * n)[:, None]
        ref = _np.concatenate((sin, cos), axis=1)
        corr = sources[init:fin].T @ ref
        idcs = _np.abs(corr).argmax(axis=0)
        sin_mode_idx = idcs[0]
        cos_mode_idx = idcs[1]
        return sin_mode_idx, cos_mode_idx

    def independent_components_analysis(
        self,
        n_components=4,
        calc_optics=True,
        **kwargs
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

        traj = _np.concatenate((self.trajx, self.trajy), axis=1)

        ica = _FastICA(
            n_components=n_components, **kwargs
        )

        # collect source signals & mixing matrix
        signals = ica.fit_transform(traj)
        mixing_matrix = ica.mixing_

        # determine betatron modes from mixing matrix
        # largest variance should be contained in the betatron modes
        # 4 modes = 2 modes for hor. & ver. planes, each
        idcs = _np.argsort(_np.std(mixing_matrix, axis=0))[-4:]
        betatron_temp_modes = signals[:, idcs]
        betatron_spatial_modes = mixing_matrix[:, idcs]

        # calulate tunes of the source signals
        spec0, tunes0 = self.calc_spectrum(betatron_temp_modes[:, 0], axis=0)
        spec1, tunes1 = self.calc_spectrum(betatron_temp_modes[:, 1], axis=0)
        spec2, tunes2 = self.calc_spectrum(betatron_temp_modes[:, 2], axis=0)
        spec3, tunes3 = self.calc_spectrum(betatron_temp_modes[:, 3], axis=0)

        tune0 = tunes0[_np.argmax(_np.abs(spec0))]
        tune1 = tunes1[_np.argmax(_np.abs(spec1))]
        tune2 = tunes2[_np.argmax(_np.abs(spec2))]
        tune3 = tunes3[_np.argmax(_np.abs(spec3))]

        # identify which signal is which (x or y)
        xidcs, yidcs = self.identify_modes_planes(
            tunes=[tune0, tune1, tune2, tune3],
            tunex=self.tunex,
            tuney=self.tuney,
        )

        # extract betatron modes fom mixing matrix
        xspatial_modes = betatron_spatial_modes[:, xidcs]
        xtemp_modes = betatron_temp_modes[:, xidcs]

        yspatial_modes = betatron_spatial_modes[:, yidcs]
        ytemp_modes = betatron_temp_modes[:, yidcs]

        # determine sin and cos modes temporal modes
        # sin/cos temporal modes correspond to cos/sin spatial modes
        # spatial modes are used for calc. optics
        # as in Xiaobiao's book , eqs. 5.31-5.32 (PCA), 5.55 (ICA)
        sinx_idx, cosx_idx = self.identify_sin_cos_modes(
            sources=xtemp_modes, plane="hor",
        )
        siny_idx, cosy_idx = self.identify_sin_cos_modes(
            sources=ytemp_modes, plane="ver",
        )

        sin_temp_modex = xtemp_modes[:, sinx_idx]  # cos/sin spatial mode
        cos_temp_modex = xtemp_modes[:, cosx_idx]  # corresponds to
        sin_spatial_modex = xspatial_modes[:, cosx_idx]  # sin/cos temp.
        cos_spatial_modex = xspatial_modes[:, sinx_idx]

        sin_temp_modey = ytemp_modes[:, siny_idx]
        cos_temp_modey = ytemp_modes[:, cosy_idx]
        sin_spatial_modey = yspatial_modes[:, cosy_idx]
        cos_spatial_modey = yspatial_modes[:, siny_idx]

        ica_data["source_signals"] = signals
        ica_data["mixing_matrix"] = mixing_matrix
        ica_data["betatron_temp_modes"] = betatron_temp_modes
        ica_data["betatron_spatial_modes"] = betatron_spatial_modes
        ica_data["xidcs"] = xidcs
        ica_data["yidcs"] = yidcs
        ica_data["sin_spatial_modex"] = sin_spatial_modex
        ica_data["cos_spatial_modex"] = cos_spatial_modex
        ica_data["sin_temp_modex"] = sin_temp_modex
        ica_data["cos_temp_modex"] = cos_temp_modex
        ica_data["sin_spatial_modey"] = sin_spatial_modey
        ica_data["cos_spatial_modey"] = cos_spatial_modey
        ica_data["sin_temp_modey"] = sin_temp_modey
        ica_data["cos_temp_modey"] = cos_temp_modey

        if not calc_optics:
            self.ica_data = ica_data
            return

        if self.model_optics is None:
            tunes = self.tunex + 49.0, self.tuney + 14.0
            self._get_nominal_optics(tunes)

        betax_model = self.model_optics["betax"]
        betay_model = self.model_optics["betay"]

        # calculate beta function & phase from spatial betatron modes
        betax, phasex = self.get_beta_and_phase_from_betatron_modes(
            sin_spatial_modex[:160], cos_spatial_modex[:160], betax_model
        )
        betay, phasey = self.get_beta_and_phase_from_betatron_modes(
            sin_spatial_modey[160:], cos_spatial_modey[160:], betay_model
        )

        # save results
        ica_data["betax"] = betax
        ica_data["betay"] = betay
        ica_data["phasex"] = phasex
        ica_data["phasey"] = phasey
        # TODO: error bars can be calculated similarly to PCA's case
        # however, variance is not explictly available as singular values
        # it must be extracted from the spatial modes.
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

        axs[0].plot(tunesx, _np.abs(trajx_spec)[:, bpm_index])
        axs[1].plot(tunesy, _np.abs(trajy_spec)[:, bpm_index])

        if not title:
            title = "kicked beam trajectories spectrum \n"
            title += f"kicks = ({self.params.hkick},{self.params.vkick}) mrad"

        axs[0].set_title(title)

        axs[0].set_ylabel(r"$x$ [mm]")
        axs[1].set_ylabel(r"$y$ [mm]")
        axs[1].set_xlabel("tune")
        fig.tight_layout()

        return fig, axs

    def plot_trajs(self, bpm_index=0, timescale=2, compare_fit=False):
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
            Defaults to 2.
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

        nr_pre = self.data["nrsamples_pre"]
        nr_post = self.data["nrsamples_post"]

        if not timescale:
            nmax_x, nmax_y = int(1 / self.tunex), int(1 / self.tuney)
            begin = max(nr_pre - 5, 0)
            slicex = (begin, nr_pre + nmax_x + 1)
            slicey = (begin, nr_pre + nmax_y + 1)
            slicesum = (begin, nr_pre + max(nmax_x, nmax_y) + 1)

        nmax = int(1 / self.SYNCH_TUNE)
        if timescale == 1:
            slicex = slicey = slicesum = (0, nr_pre + nmax + 1)

        elif timescale == 2:
            slicex = slicey = slicesum = (0, nr_pre + nr_post + 1)

        trajx = self.trajx[:, bpm_index]
        trajy = self.trajy[:, bpm_index]
        trajsum = self.trajsum[:, bpm_index]

        if self._bpms_names is None:
            self._bpms_names = SOFBFactory.create("SI").bpm_names

        fig, ax = _mplt.subplots(1, 3, figsize=(15, 5))
        name = self._bpms_names[bpm_index]
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
                "x-",
                mfc="none",
                color="blue",
                label="fit",
            )
            ax[0].plot(
                _np.arange(init, end + 1, 1),
                res[:, bpm_index],
                "x-",
                mfc="none",
                color="green",
                label="residue",
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
                "x-",
                mfc="none",
                color="red",
                label="fit",
            )
            ax[1].plot(
                _np.arange(init, end + 1, 1),
                res[:, bpm_index],
                "x-",
                mfc="none",
                color="green",
                label="residue",
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

    def plot_modal_analysis(self, method="PCA", **kwargs):
        """."""
        if self.pca_data is None:
            self.principal_components_analysis(calc_optics=False)
        if method.lower() == "pca":
            data = self.pca_data
            u = data["source_signals"]
        elif method.lower() == "ica":
            if self.ica_data is None:
                self.independent_components_analysis(
                    calc_optics=False, **kwargs
                )
            data = self.ica_data
            u = data["betatron_temp_modes"]
        else:
            raise ValueError("Unknown modal analysis method.")

        s = self.pca_data["singular_values"]
        s_xidcs = self.pca_data["xidcs"]
        s_yidcs = self.pca_data["yidcs"]

        xidcs = data["xidcs"]
        yidcs = data["yidcs"]

        sin_temp_modex = data["sin_temp_modex"]
        sin_temp_modey = data["sin_temp_modey"]
        cos_temp_modex = data["cos_temp_modex"]
        cos_temp_modey = data["cos_temp_modey"]

        sin_spatial_modex = data["sin_spatial_modex"]
        sin_spatial_modey = data["sin_spatial_modey"]
        cos_spatial_modex = data["cos_spatial_modex"]
        cos_spatial_modey = data["cos_spatial_modey"]

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

        svals.plot(s, "o", color="k", mfc="none")
        idx = _np.arange(len(s))
        sel = _np.array(s_xidcs)
        svals.plot(
            idx[sel], s[sel], "o", color="b",
            mfc="none", label="hor. signals singular values"
        )
        sel = _np.array(s_yidcs)
        svals.plot(
            idx[sel], s[sel], "o", color="r",
            mfc="none", label='ver. signals singular values')
        svals.set_title("singular values spectrum")
        svals.set_yscale("log")
        svals.legend()

        variance = _np.cumsum(s) / _np.sum(s)
        var.plot(variance, "o", color="k", mfc="none")
        idx = _np.arange(len(variance))
        sel = _np.array(xidcs)
        var.plot(idx[sel], variance[sel], "o", color="b", mfc="none")

        sel = _np.array(yidcs)
        var.plot(idx[sel], variance[sel], "o", color="r", mfc="none")

        var.set_title("explained variance")
        var.set_xlabel("rank")

        source1.plot(sin_temp_modex, label="hor. sin mode")
        source1.plot(cos_temp_modex, label="hor. cos mode")
        source1.set_title("temporal modes - hor. source signals")
        source1.set_xlabel("turns index")
        source1.legend()

        source2.plot(sin_temp_modey, label="ver. sin mode")
        source2.plot(cos_temp_modey, label="ver. cos mode")
        source2.set_title("temporal modes - ver. source signals")
        source2.set_xlabel("turns index")
        source2.legend()

        spatial1.plot(cos_spatial_modex, label="hor. cos mode")
        spatial1.plot(sin_spatial_modex, label="hor. sin mode")
        spatial1.set_title("spatial modes - hor. source signals")
        spatial1.set_xlabel("BPMs index (H/V)")
        spatial1.legend()

        spatial2.plot(cos_spatial_modey, label="ver. cos mode")
        spatial2.plot(sin_spatial_modey, label="ver. sin mode")
        spatial2.set_title("spatial modes - ver. source signals")
        spatial2.set_xlabel("BPMs index (H/V)")
        spatial2.legend()

        freq, fourier = _np.fft.rfftfreq(n=u.shape[0]), _np.fft.rfft(u, axis=0)

        specx = _np.abs(fourier)[:, _np.array(xidcs)]
        spec1.plot(
            freq, specx[:, 0], "o-", color="b",
            mfc="none", label="hor. cos mode"
        )
        spec1.plot(
            freq, specx[:, 1], "x-", color="b",
            label="hor. sin mode"
        )
        spec1.set_title("temporal modes spectrum - hor. source signals")
        spec1.set_xlabel("fractional tune")
        spec1.legend()
        peak_idx = _np.argmax(specx[:, 0])
        peak_val = specx[peak_idx, 0]
        peak_freq = freq[peak_idx]
        spec1.annotate(
            f"hor. tune = {peak_freq:.4f}",
            xy=(peak_freq, peak_val),
            xytext=(peak_freq + 0.05, 0.9 * peak_val),
            arrowprops=dict(arrowstyle="->", color="k"),
            bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=0.5),
            fontsize=10,
        )

        specy = _np.abs(fourier)[:, _np.array(yidcs)]
        spec2.plot(
            freq, specy[:, 0], "o-", color="r",
            mfc="none", label="ver. cos mode",
        )
        spec2.plot(
            freq, specy[:, 1], "x-", color="r",
            label="ver. sin mode"
        )
        spec2.set_title("temporal modes spectrum - ver. source signals")
        spec2.set_xlabel("fractional tune")
        spec2.legend()
        peak_idx = _np.argmax(specy[:, 0])
        peak_val = specy[peak_idx, 0]
        peak_freq = freq[peak_idx]
        spec2.annotate(
            f"ver. tune = {peak_freq:.4f}",
            xy=(peak_freq, peak_val),
            xytext=(peak_freq + 0.05, 0.9 * peak_val),
            arrowprops=dict(arrowstyle="->", color="k"),
            bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="k", lw=0.5),
            fontsize=10,
        )
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
                self.fitting_data["fitting" + label + "_residue"].std(
                    axis=axis
                ),
                "o-",
                mfc="none",
            )
            _mplt.xlabel(f"{along} index")
            _mplt.ylabel("residue [mm]")

    def plot_betabeat_and_phase_error(
        self,
        beta_model,
        beta_meas,
        phase_model,
        phase_meas,
        title=None,
        compare_meas2model=False,
        bpms2use=None,
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
            ax_beta.plot(beta_model, "o-", label="Model", mfc="none")
            ax_beta.plot(beta_meas, "o--", label="Meas", mfc="none")
            ax_beta.set_ylabel("beta function")
            ax_beta.legend()

        # Beta beating plot
        ax_beat = axs[0, 1] if compare_meas2model else axs[0]
        beta_beat = (beta_model - beta_meas) / beta_model
        beta_beat *= 100  # [%]
        ax_beat.plot(
            beta_beat,
            "o-",
            label=f"rms beat = {_np.nanstd(beta_beat):.2f} %",
            mfc="none",
        )
        ax_beat.set_ylabel("beta beating [%]")
        ax_beat.legend()

        # Phase advance plot
        if compare_meas2model:
            model_phase_advance = _np.diff(phase_model)
            meas_phase_advance = _np.abs(_np.diff(phase_meas))

            ax_phase = axs[1, 0]
            ax_phase.plot(model_phase_advance, "o-", label="Model", mfc="none")
            ax_phase.plot(meas_phase_advance, "o--", label="Meas", mfc="none")
            ax_phase.set_ylabel("BPMs phase advance [rad]")
            ax_phase.legend()

        # Phase advance error plot
        ax_phase_err = axs[1, 1] if compare_meas2model else axs[1]
        ph_err = model_phase_advance - meas_phase_advance
        ax_phase_err.plot(
            ph_err,
            "o-",
            label=f"rms err={_np.nanstd(ph_err):.2f}",
            mfc="none",
        )
        ax_phase_err.set_ylabel("BPMs phase advance error [rad]")
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
            f=self.harmonic_tbt_model,
            xdata=xdata,
            ydata=bpmdata,
            p0=params_guess,
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
            phase_meas = _np.unwrap(phase_meas)
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
        beta = sin_mode**2 + cos_mode**2
        beta /= _np.std(beta) / _np.std(beta_model)
        phase = _np.arctan2(sin_mode, cos_mode)
        phase = _np.unwrap(phase)
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
