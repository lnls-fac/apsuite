"""Main module."""

import time as _time
import datetime as _datetime
from functools import reduce as _red
import operator as _opr
from copy import deepcopy as _dcopy

import numpy as _np

from mathphys.functions import (
    get_namedtuple as _get_namedtuple,
    load as _load,
    save as _save,
)

from siriuspy.devices import (
    SOFB as _SOFB,
    StrengthConv as _StrengthConv,
    CurrInfoSI as _CurrInfoSI,
    FamBPMs as _FamBPMs,
    PowerSupply as _PowerSupply,
    RFGen as _RFGen,
    Trigger as _Trigger,
    Event as _Event,
    EVG as _EVG,
    Tune as _Tune,
)
from siriuspy.search import LLTimeSearch as _LLTime
from siriuspy.sofb.csdev import SOFBFactory as _SOFBFactory
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient

from apsuite.utils import (
    ParamsBaseClass as _ParamsBaseClass,
    ThreadedMeasBaseClass as _BaseClass,
)

from apsuite.commisslib.meas_ac_orm import MeasACORM as _MeasACORM
from apsuite.commisslib.measure_bba import BBAParams as _BBAParams


class ACBBAParams(_ParamsBaseClass):
    """Parameters for AC-BBA."""

    BPMNAMES = _BBAParams.BPMNAMES[:2]
    QUADNAMES = _BBAParams.QUADNAMES[:2]
    QUAD_MODULATION_MODE = _get_namedtuple("QuadModulationMode", ["AC", "DC"])

    def __init__(self):
        """."""
        super().__init__()
        self.timeout_bpms = 60  # [s]
        self.timeout_correctors = 20  # [s]

        self.quad_modulation_mode = ACBBAParams.QUAD_MODULATION_MODE.DC
        self.quad_delta_kl = 0.02  # [1/m]
        self.wait_quadrupole = 0.3  # [s]

        self.cv_freq = 17.0  # [Hz]
        self.ch_freq = 23.0  # [Hz]
        self.qn_freq = 0.0  # [Hz]
        self.qs_freq = 0.0  # [Hz]

        self.excit_time = 4  # [s]
        self.corrs_delay = 5e-3  # [s]

        self.ch_kick = 5  # [urad]
        self.cv_kick = 5  # [urad]

        self.dorbx = 100.0  # [um]
        self.dorby = 100.0  # [um]

        self.use_qs_excitation = False
        self.use_qn_excitation = False

        self.measure_bpms_noise = True
        self.acq_rate = "FAcq"
        self.orm_name = "ref_respmat"

        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 5  # [um]
        self.correct_orbit_each_step = True

    def __str__(self):
        """."""
        st = "BBA-AC measurement parameters."
        return st


class DoACBBA(_BaseClass):
    """AC Beam Based Alignment measurement."""

    STATUS = _get_namedtuple("Status", ["Fail", "Success"])

    TIMING_STATE_OPTIONS = (
        "trigbpms_source",
        "trigbpms_nr_pulses",
        "trigbpms_delay_raw",
        "trigcorrs_source",
        "trigcorrs_nr_pulses",
        "trigcorrs_delay_raw",
        "trigcorrs_delta_delay_raw",
        "evt_mode",
        "evt_delay_raw",
    )

    def __init__(self, isonline=True, func=None, LLTime=_LLTime):
        """."""
        self.params = ACBBAParams()
        super().__init__(
            params=self.params, target=self._do_acbba, isonline=isonline
        )

        self.verbose = True

        self.data["bpmnames"] = list(ACBBAParams.BPMNAMES)
        self.data["quadnames"] = list(ACBBAParams.QUADNAMES)
        self.data["scancenterx"] = _np.zeros(len(ACBBAParams.BPMNAMES))
        self.data["scancentery"] = _np.zeros(len(ACBBAParams.BPMNAMES))

        self._bpms2dobba = self.data["bpmnames"]
        self.sofb_data = None
        self.configdb = None
        self._orm = None

        if self.isonline:
            self.sofb_data = _SOFBFactory.create("SI")
            self.configdb = _ConfigDBClient(config_type="si_bbadata")
            self._create_devices()
        else:
            if func is not None:
                func(self)
            if LLTime is not None:
                self._LLTime = LLTime

        self.data["log"] = [(_time.time(), "Started.")]

    @property
    def bpms2dobba(self):
        """List of BPMs to perform BBA."""
        return self._bpms2dobba.copy()

    @bpms2dobba.setter
    def bpms2dobba(self, bpmnames):
        """List of BPMs to perform BBA."""
        if not isinstance(bpmnames, (list, tuple)):
            raise TypeError("bpmnames must be a list (or tuple) of strings.")
        for bpm in bpmnames:
            if bpm not in self.data["bpmnames"]:
                msg = f'Invalid BPM: {bpm}. Check "ACBBAParams.BPMNAMES".'
                raise ValueError(msg)
        self._bpms2dobba = bpmnames

    def _create_devices(self):
        """Create and connect to devices."""
        # BPMs
        self.bpms = _FamBPMs(mturn_signals2acq="XY", props2init="acq")
        self.devices["fambpms"] = self.bpms

        # Quadrupoles
        props = ["PwrState-Sts", "KL-SP", "KL-RB", "KLRef-Mon"]
        for qname in self.data["quadnames"]:
            if qname in self.devices:
                continue
            self.devices[qname] = _PowerSupply(qname, props2init=props)

        # Kick converters
        sofbdata = self.sofb_data
        self.devices.update({
            n + ":StrengthConv": _StrengthConv(n, "Ref-Mon")
            for n in (sofbdata.ch_names + sofbdata.cv_names)
        })

        # Correctors
        props = [
            "Kick-SP",
            "OpMode-Sel",
            "OpMode-Sts",
            "Current-SP",
            "Current-RB",
            "Current-Mon",
            "CurrentRef-Mon",
            "CycleType-Sel",
            "CycleFreq-SP",
            "CycleAmpl-SP",
            "CycleOffset-SP",
            "CycleAuxParam-SP",
            "CycleAuxParam-RB",
            "CycleNrCycles-SP",
            "CycleAmpl-RB",
            "CycleOffset-RB",
            "CycleFreq-RB",
            "CycleNrCycles-RB",
            "CycleType-Sts",
            "CycleEnbl-Mon",
            "ParamPWMFreq-Cte",
        ]
        self.devices.update({
            name: _PowerSupply(name, props2init=props)
            for name in (sofbdata.ch_names + sofbdata.cv_names)
        })

        # SOFB
        self.devices["sofb"] = _SOFB(_SOFB.DEVICES.SI)

        # CurrInfo
        self.devices["currinfo"] = _CurrInfoSI()

        # RF generator
        props = ["GeneralFreq-SP", "GeneralFreq-RB"]
        self.devices["rfgen"] = _RFGen(props2init=props)

        # Tune
        self.devices["tune"] = _Tune(_Tune.DEVICES.SI)

        # BPMs Trigger
        props = [
            "Src-Sts",
            "NrPulses-RB",
            "DelayRaw-RB",
            "Src-Sel",
            "NrPulses-SP",
            "DelayRaw-SP",
        ]
        self.devices["trigbpms"] = _Trigger("SI-Fam:TI-BPM", props2init=props)

        # Correctors Trigger
        props = [
            "Src-Sts",
            "NrPulses-RB",
            "DelayRaw-RB",
            "DeltaDelayRaw-RB",
            "Src-Sel",
            "NrPulses-SP",
            "DelayRaw-SP",
            "LowLvlTriggers-Cte",
            "DeltaDelayRaw-SP",
        ]
        self.devices["trigcorrs"] = _Trigger(
            "SI-Glob:TI-Mags-Corrs", props2init=props
        )
        # Event to start synchronous acquisition:
        props = [
            "Mode-Sts",
            "DelayRaw-RB",
            "Mode-Sel",
            "DelayRaw-SP",
            "ExtTrig-Cmd",
        ]
        self.devices["evt"] = _Event("Study", props2init=props)
        props = ["ContinuousEvt-Sts", "UpdateEvt-Cmd"]
        self.devices["evg"] = _EVG(props2init=props)

    def _setup_orm(self):
        """Get the orbit response matrix from configdb server."""
        name = self.params.orm_name
        self._orm = _np.array(self.configdb.get_config_value(name))

    def _do_acbba(self):
        """."""
        # Initialize data
        self.data["measure"] = dict()
        self._setup_orm()

        # Start
        tini = _datetime.datetime.fromtimestamp(_time.time())
        msg = f"Starting measurement at {tini.strftime('%Y-%m-%d %Hh%Mm%Ss')}"
        self._log("\n" + msg)

        if self.devices["sofb"].autocorrsts:
            msg = "SOFB feedback is enabled. Please desable it first."
            self._log(msg)
            return self.STATUS.Fail

        # Get initial timing state
        self._log("Getting Timing state... ", end="")
        timing_state = self.get_timing_state()
        self._log("Done!")

        # Measure BPMs noise
        if self.params.measure_bpms_noise:
            msg = "Measuring BPMs noise:"
            self._log(msg)
            measnoise_ok, noise_data = self._do_measure_bpms_noise(tab=1)
            self.data["bpms_noise"] = noise_data
            if not measnoise_ok:
                msg = "Problem measuring BPMs noise."
                self._log(msg)

        # Set/Check if Correctors are in SlowRef mode
        msg = "Setting Correctors OpMode to SlowRef... "
        self._log(msg, end="")
        corrs_opmode_ok = self._change_corrs_opmode(
            "slowref", self.sofb_data.ch_names + self.sofb_data.cv_names
        )
        if not corrs_opmode_ok:
            msg = "Fail: Could not set OpMode to SlowRef. Exiting."
            self._log(msg)
            return self.STATUS.Fail
        self._log("Done!")

        # Do AC-BBA for each BPM
        nr_bpms = len(self._bpms2dobba)
        msg = f"Running AC-BBA for {nr_bpms:03d} BPMs:"
        self._log(msg)
        stsok = True
        for i, bpm in enumerate(self._bpms2dobba):
            if self._stopevt.is_set():
                msg = "Stopped!"
                self._log(msg, tab=1)
                stsok = False
                break
            if not self.havebeam:
                msg = "Beam was lost!"
                self._log(msg, tab=1)
                stsok = False
                break
            msg = f'Doing AC-BBA for BPM "{bpm}" ({i + 1:03d}/{nr_bpms:03d}):'
            self._log(msg, tab=1)
            sts, data_acq = self._do_acbba_single_bpm(bpm, tab=2)
            self.data["measure"][bpm] = data_acq
            if sts == self.STATUS.Fail:
                stsok = False
                msg = "Fail!"
                self._log(msg, tab=1)
                break
            self._log("Done!", tab=1)
        self._log(f"{'Done' if stsok else 'Fail'}!")

        # Restore Correctors opmode to SlowRef
        msg = "Restoring Correctors OpMode to SlowRef... "
        self._log(msg, end="")
        corrs_opmode_ok = self._change_corrs_opmode(
            "slowref",
            self.sofb_data.ch_names + self.sofb_data.cv_names,
            timeout=self.params.timeout_correctors,
        )
        if not corrs_opmode_ok:
            msg = "Fail: Could restore OpMode to SlowRef. Exiting."
            self._log(msg)
            stsok = False
        else:
            self._log("Done!")

        # Restore timing state
        self._log("Restoring Timing state... ", end="")
        self.set_timing_state(timing_state)
        self._log("Done!")

        # Correct orbit before ending
        if self.havebeam:
            self._log("Correcting Orbit... ", end="")
            self.correct_orbit()
            self._log("Ok!")

        # Finish
        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini).split(".")[0]
        msg = f"Measurement finished! ET: {dtime}"
        self._log(msg)
        return self.STATUS.Success if stsok else self.STATUS.Fail

    def _do_acbba_single_bpm(self, bpmname, **kw):
        tab = kw.pop("tab", 0)

        if bpmname not in self.data["bpmnames"]:
            msg = f"Invalid BPM: {bpmname}."
            self._log(msg, tab=tab)
            return self.STATUS.Fail, None

        # correct orbit
        if self.params.correct_orbit_each_step:
            self._log("Correcting Orbit... ", end="", tab=tab)
            self.correct_orbit()
            self._log("Ok!")

        quadname = self.data["quadnames"][self.data["bpmnames"].index(bpmname)]
        quadmode = self.params.quad_modulation_mode

        chname, cvname, *_ = self._get_correctors_for_bpm(bpmname, self._orm)

        stren_ini = self.get_quad_strength(quadname)
        delta_kl = self.params.quad_delta_kl

        data = {
            "quadname": quadname,
            "chname": chname,
            "cvname": cvname,
            "pos": None,
            "neg": None,
            "quadmode": quadmode,
            "quad_stren_ini": stren_ini,
            "quad_delta_kl": delta_kl,
        }

        if quadmode == self.params.QUAD_MODULATION_MODE.AC:
            msg = "Quadrupole AC modulation mode is not implemented yet."
            self._log(msg, tab=tab)
            return self.STATUS.Fail, data
        elif quadmode == self.params.QUAD_MODULATION_MODE.DC:
            msg = "Setup: Quadrupole modulation mode: DC."
            self._log(msg, tab=tab)

            # Set Quad KL = KL0 + dKL
            msg = "Step: Positive quadrupole modulation."
            self._log(msg, tab=tab)
            msg = f'Changing quadrupole "{quadname}" strength... '
            self._log(msg, tab=tab + 1, end="")
            sts = self.set_quad_strength(
                quadname, stren_ini + delta_kl / 2, tab=tab + 1
            )
            if sts == self.STATUS.Fail:
                self.set_quad_strength(
                    quadname, stren_ini, ignore_timeout=True
                )
                return sts, data
            self._log("Done!")

            sts, data_pos = self._acquire_data(chname, cvname, tab=tab + 1)
            data["pos"] = data_pos
            if sts == self.STATUS.Fail:
                return sts, data

            # Set Quad KL = KL0 - dKL
            msg = "Step: Negative quadrupole modulation."
            self._log(msg, tab=tab)
            msg = f'Changing quadrupole "{quadname}" strength... '
            self._log(msg, tab=tab + 1, end="")
            sts = self.set_quad_strength(
                quadname, stren_ini - delta_kl / 2, tab=tab + 1
            )
            if sts == self.STATUS.Fail:
                self.set_quad_strength(
                    quadname, stren_ini, ignore_timeout=True
                )
                return sts, data
            self._log("Done!")

            sts, data_neg = self._acquire_data(chname, cvname, tab=tab + 1)
            data["neg"] = data_neg
            if sts == self.STATUS.Fail:
                return sts, data

            # Restore Quad KL = KL0
            msg = "Step: Restoring quadrupole strength."
            self._log(msg, tab=tab)
            msg = f'Changing quadrupole "{quadname}" strength... '
            self._log(msg, tab=tab + 1, end="")
            sts = self.set_quad_strength(quadname, stren_ini, tab=tab + 1)
            if sts == self.STATUS.Fail:
                self.set_quad_strength(
                    quadname, stren_ini, ignore_timeout=True
                )
                return sts, data
            self._log("Done!")

        else:
            msg = "Invalid Quadrupole modulation mode. Skipping..."
            self._log(msg, tab=tab)
            return self.STATUS.Fail, data

        return self.STATUS.Success, data

    def _acquire_data(self, ch_name, cv_name, **kw):
        """."""
        tab = kw.pop("tab", 0)
        corr_names = [ch_name, cv_name]
        corr_freqs = [self.params.ch_freq, self.params.cv_freq]
        corr_kicks = [self.params.ch_kick, self.params.cv_kick]
        excit_time = self.params.excit_time
        tout_bpms = self.params.timeout_bpms
        tout_corrs = self.params.timeout_correctors

        # Configure BPMs and Timing
        t00 = _time.time()
        msg = "Configuring BPMs and Timing... "
        self._log(msg, tab=tab, end="")
        nr_points = self._get_acq_nr_points()
        ret = self._config_bpms(nr_points, rate=self.params.acq_rate)
        sts = self._check_bpms_configok(ret)
        if sts == self.STATUS.Fail:
            return sts, None
        self._config_timing(
            self.params.corrs_delay,
            chs=[[ch_name]],
            cvs=[[cv_name]],
            nr_points=nr_points,
        )
        msg = f"Done! ET: {_time.time() - t00:.2f}s"
        self._log(msg)

        # Configure correctors
        t01 = _time.time()
        msg = "Configuring correctors... "
        self._log(msg, tab=tab, end="")
        self._config_correctors(corr_names, corr_kicks, corr_freqs, excit_time)
        msg = f"Done! ET: {_time.time() - t01:.2f}s"
        self._log(msg)

        # Configure correctors opmode to Cycle
        t02 = _time.time()
        msg = f"Changing Corrs. ({ch_name}, {cv_name}) OpMode to Cycle... "
        self._log(msg, tab=tab, end="")
        if not self._change_corrs_opmode("cycle", corr_names, tab=tab):
            msg = "Fail! Could not set OpMode to Cycle."
            self._log(msg, tab=tab)
            return self.STATUS.Fail, None
        msg = f"Done! ET: {_time.time() - t02:.2f}s"
        self._log(msg)

        # Trigger Event
        t03 = _time.time()
        msg = "Sending timing signal... "
        self._log(msg, tab=tab, end="")
        self.bpms.reset_mturn_initial_state()
        self.devices["evt"].cmd_external_trigger()
        msg = f"Done! ET: {_time.time() - t03:.2f}s"
        self._log(msg)

        # Wait BPMs to update with new data
        t04 = _time.time()
        msg = "Waiting BPMs to update... "
        self._log(msg, tab=tab, end="")
        ret = self.bpms.wait_update_mturn(timeout=tout_bpms)
        sts = self._check_if_bpms_updated(ret)
        sts_str = "Done" if sts == self.STATUS.Success else "Fail"
        msg = f"{sts_str}! ET: {_time.time() - t04:.2f}s"
        self._log(msg, tab=tab)

        # Save data
        t05 = _time.time()
        msg = "Saving data... "
        self._log(msg, end="", tab=tab)
        data = self.get_general_data()
        data.update(self.get_bpms_data())
        msg = f"Done! ET: {_time.time() - t05:.2f}s"
        self._log(msg)

        # Restore Correctors opmode to SlowRef
        t06 = _time.time()
        msg = f"Restoring Corrs. ({ch_name}, {cv_name}) OpMode to SlowRef... "
        self._log(msg, tab=tab, end="")
        if not self._wait_cycle_to_finish(corr_names, timeout=tout_corrs):
            msg = "Fail! Cycle still not finished."
            self._log(msg)
            return self.STATUS.Fail, data
        if not self._change_corrs_opmode("slowref", corr_names, tab=tab):
            msg = "Fail! Could restore OpMode to SlowRef."
            self._log(msg, tab=tab)
            return self.STATUS.Fail, data
        msg = f"Done! ET: {_time.time() - t06:.2f}s"
        self._log(msg)

        return self.STATUS.Success, data

    def _check_bpms_configok(self, ret):
        bpmnames = self.data["bpmnames"]
        if ret < 0:
            idx = -int(ret) - 1
            # msg = f"BPM {idx:d} did not finish last acquisition."
            bpmname = bpmnames[idx]
            msg = f'"{bpmname}" did not finish last acquisition.'
            self._log(msg)
        elif ret > 0:
            idx = int(ret) - 1
            # msg = f"BPM {idx:d} is not ready for acquisition."
            bpmname = bpmnames[idx]
            msg = f'"{bpmname}" is not ready for acquisition.'
            self._log(msg)
        return self.STATUS.Fail if ret else self.STATUS.Success

    def _check_if_bpms_updated(self, ret):
        """."""
        if ret != 0:
            if ret > 0:
                tag = self.bpms.bpm_names[int(ret) - 1]
                pos = self.bpms.mturn_signals2acq[int((ret % 1) * 10) - 1]
                msg = f'Problem: BPM "{tag}" did not update, signal {pos}.'
            elif ret == -1:
                msg = "Problem: Initial timestamps were not defined."
            elif ret == -2:
                msg = "Problem: signals size changed."
            self._log(msg)
            return self.STATUS.Fail
        return self.STATUS.Success

    def _get_correctors_for_bpm(self, bpmname, orm=None):
        """Choose a CH and a CV that most affect the target BPM."""
        if orm is None:
            orm = self._orm
        if orm is None:
            raise RuntimeError("Orbit Response Matrix not loaded.")

        sofb = self.sofb_data
        bpmnames = self.data["bpmnames"]
        if bpmname not in bpmnames:
            raise ValueError("Invalid BPM! Check ACBBAParams.BPMNAMES")
        bpm_idx = bpmnames.index(bpmname)

        orm_xx = orm[: sofb.nr_bpms, : sofb.nr_ch]
        orm_yy = orm[sofb.nr_bpms :, sofb.nr_ch : sofb.nr_chcv]

        ch_idx = int(_np.argmax(_np.abs(orm_xx[bpm_idx, :])))
        cv_idx = int(_np.argmax(_np.abs(orm_yy[bpm_idx, :])))

        ch_name = sofb.ch_names[ch_idx]
        cv_name = sofb.cv_names[cv_idx]
        return ch_name, cv_name, ch_idx, cv_idx + sofb.nr_ch

    def _do_measure_bpms_noise(self, **kw):
        tab = kw.pop("tab", 0)
        tini = _datetime.datetime.fromtimestamp(_time.time())

        stsok = True

        t00 = _time.time()
        msg = "Configuring BPMs and Timing... "
        self._log(msg, tab=tab, end="")
        nr_points = self._get_acq_nr_points()
        ret = self._config_bpms(nr_points, rate=self.params.acq_rate)
        sts = self._check_bpms_configok(ret)
        self._config_timing()
        if sts == self.STATUS.Success:
            msg = f"Done! ET: {_time.time() - t00:.2f}s"
            self._log(msg)
        else:
            stsok = False

        t01 = _time.time()
        msg = "Sending Trigger signal... "
        self._log(msg, tab=tab, end="")
        self.bpms.reset_mturn_initial_state()
        self.devices["evt"].cmd_external_trigger()
        msg = f"Done! ET: {_time.time() - t01:.2f}s"
        self._log(msg)

        t02 = _time.time()
        msg = "Waiting for BPMs to update... "
        self._log(msg, tab=tab, end="")
        ret = self.bpms.wait_update_mturn(timeout=self.params.timeout_bpms)
        sts = self._check_if_bpms_updated(ret)
        if sts == self.STATUS.Success:
            sts_str = "Done"
        else:
            stsok = False
            sts_str = "Fail"
        self._log(f"{sts_str}! ET: {_time.time() - t02:.2f}s")

        _time.sleep(0.5)
        data = self.get_general_data()
        data.update(self.get_bpms_data())
        data["ch_freq"] = self.params.ch_freq
        data["cv_freq"] = self.params.cv_freq

        tfin = _datetime.datetime.fromtimestamp(_time.time())
        dtime = str(tfin - tini).split(".")[0]
        msg = f"{'Done' if stsok else 'Fail'}! ET: {dtime}"
        self._log(msg)
        return stsok, data

    def _get_acq_nr_points(self):
        freq = self.devices["rfgen"].frequency
        rate = self.params.acq_rate
        n_pts = self.params.excit_time
        n_pts += self.params.corrs_delay * 2
        n_pts *= self.bpms.get_sampling_frequency(freq, acq_rate=rate)
        n_pts = int(_np.ceil(n_pts))
        return n_pts

    def get_bpms_data(self):
        """Get all BPM related data relevant for the measurements.

        Returns:
            dict: BPMs data.

        """
        orbx, orby = self.bpms.get_mturn_signals()
        bpm0 = self.bpms.devices[0]
        rf_freq = self.devices["rfgen"].frequency

        data = dict()
        data["orbx"] = orbx
        data["orby"] = orby
        data["rf_frequency"] = rf_freq
        data["acq_rate"] = bpm0.acq_channel_str
        data["sampling_frequency"] = self.bpms.get_sampling_frequency(rf_freq)
        data["nrsamples_pre"] = bpm0.acq_nrsamples_pre
        data["nrsamples_post"] = bpm0.acq_nrsamples_post
        data["trig_delay_raw"] = self.devices["trigbpms"].delay_raw
        data["switching_mode"] = bpm0.switching_mode_str
        data["switching_frequency"] = self.bpms.get_switching_frequency(
            rf_freq
        )
        return data

    def get_general_data(self):
        """Get general purpose data.

        Returns:
            dict: general purpose data.

        """
        data = dict()
        data["timestamp"] = _time.time()
        data["stored_current"] = self.devices["currinfo"].current
        data["tunex"] = self.devices["tune"].tunex
        data["tuney"] = self.devices["tune"].tuney
        return data

    def _config_bpms(self, nr_points, rate=None):
        if rate is None:
            rate = self.params.acq_rate
        return self.bpms.config_mturn_acquisition(
            acq_rate=rate,
            nr_points_before=0,
            nr_points_after=nr_points,
            repeat=False,
            external=True,
        )

    def get_timing_state(self):
        """Get the timing state."""
        state = dict()
        for opt in DoACBBA.TIMING_STATE_OPTIONS:
            devname, *state_opt = opt.split("_")
            state_opt = "_".join(state_opt)
            device = self.devices.get(devname, None)
            if device is not None:
                state[opt] = 0
                state[opt] = getattr(device, state_opt)
        return state

    def set_timing_state(self, state):
        """Set the timing state."""
        for opt in DoACBBA.TIMING_STATE_OPTIONS:
            if opt not in state.keys():
                continue
            devname, *state_opt = opt.split("_")
            device = self.devices.get(devname, None)
            state_opt = "_".join(state_opt)
            if device is not None:
                setattr(device, state_opt, state[opt])
                continue
        _time.sleep(0.1)
        self.devices["evg"].cmd_update_events()

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
        state["trigbpms_source"] = "Study"
        state["trigbpms_nr_pulses"] = 1
        state["trigbpms_delay_raw"] = 0.0

        state["evt_mode"] = "External"
        state["evt_delay_raw"] = 0

        state["trigcorrs_source"] = "Study"
        state["trigcorrs_nr_pulses"] = 1

        rf_freq = self.devices["rfgen"].frequency
        ftim = rf_freq / 4  # timing base frequency
        dly = int(cm_dly * ftim)
        if chs is None or cvs is None or nr_points is None:
            state["trigcorrs_delay_raw"] = dly
            self.set_timing_state(state)
            return

        state["trigcorrs_delay_raw"] = 0
        nr_runs = len(chs)
        # Calculate delta_delay for correctors to be as close as possible to a
        # multiple of the the sampling period to ensure repeatability of
        # experiment along runs excited during single acquisition:
        fsamp = self.bpms.get_sampling_frequency(rf_freq, self.params.acq_rate)
        runs_delta_dly = _np.arange(nr_runs, dtype=float)
        runs_delta_dly *= nr_points / fsamp
        runs_delta_dlyr = _np.round(runs_delta_dly * ftim)

        # get low level trigger names to be configured in each run of the
        # acquisition:
        ll_trigs = []
        for ch, cv in zip(chs, cvs):  # noqa: B905
            llt = set()
            for c in ch + cv:
                trig = self._LLTime.get_trigger_name(c + ":BCKPLN")
                llt.add(trig)
            ll_trigs.append(llt)

        # check if correctors controlled by the same trigger are requested to
        # be triggered in different times during the same acquisition
        if len(_red(_opr.or_, ll_trigs)) != _red(_opr.add, map(len, ll_trigs)):
            raise ValueError("Impossible trigger configuration requested.")

        trigcorr = self.devices["trigcorrs"]
        delta_delay_raw = _np.zeros(trigcorr.delta_delay_raw.size)
        low_level = trigcorr.low_level_triggers
        for llts, ddlyr in zip(ll_trigs, runs_delta_dlyr):  # noqa: B905
            # Find all low level triggers of this sector and set their delay:
            for llt in llts:
                if llt not in low_level:
                    raise ValueError(f"Trigger {llt:s} is not valid.")
                delta_delay_raw[low_level.index(llt)] = ddlyr + dly
        state["trigcorr_delta_delay_raw"] = delta_delay_raw
        self.set_timing_state(state)

    def _log(self, msg, *args, **kwargs):
        """."""
        if "tab" in kwargs:
            tab = kwargs.pop("tab")
            msg = "  " * tab + msg
        if self.verbose:
            print(msg, *args, **kwargs)
        self.data["log"].append((_time.time(), msg))

    @property
    def havebeam(self):
        """."""
        haveb = self.devices["currinfo"]
        return haveb.connected and haveb.storedbeam

    def _config_correctors(self, corr_names, kicks, freqs, excit_time):
        """."""
        for i, cmn in enumerate(corr_names):
            cmo = self.devices[cmn]
            conv = self.devices[cmn + ":StrengthConv"].conv_strength_2_current
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
            fsamp = cmo["ParamPWMFreq-Cte"]
            params = cmo.cycle_aux_param
            params[1] = freqs[i] / fsamp * 360
            params[1] *= 0.1
            cmo.cycle_aux_param = params

    def _change_corrs_opmode(self, mode, corr_names=None, timeout=None, **kw):
        """."""
        tab = kw.pop("tab", 0)
        if timeout is None:
            timeout = self.params.timeout_correctors

        opm_sel = _PowerSupply.OPMODE_SEL
        opm_sts = _PowerSupply.OPMODE_STS
        mode_sel = opm_sel.Cycle if mode == "cycle" else opm_sel.SlowRef
        mode_sts = opm_sts.Cycle if mode == "cycle" else opm_sts.SlowRef

        if corr_names is None:
            corr_names = self.sofb_data.ch_names + self.sofb_data.cv_names

        for cmn in corr_names:
            cmo = self.devices[cmn]
            cmo.opmode = mode_sel

        for cmn in corr_names:
            dt_ = _time.time()
            cmo = self.devices[cmn]
            if not cmo.wait("OpMode-Sts", mode_sts, timeout=timeout):
                msg = "\nERR:" + cmn + " did not change to " + mode
                self._log(msg, tab=tab)
                return False
            dt_ -= _time.time()
            timeout = max(timeout + dt_, 0)
            cmo.current = cmo.current
        return True

    def _wait_cycle_to_finish(self, corr_names=None, timeout=None):
        """."""
        if timeout is None:
            timeout = self.params.timeout_correctors
        if corr_names is None:
            corr_names = self.sofb_data.ch_names + self.sofb_data.cv_names

        for cmn in corr_names:
            cmo = self.devices[cmn]
            if not cmo.wait_cycle_to_finish(timeout=timeout):
                return False
        return True

    def get_quad_strength(self, quadname):
        """."""
        if quadname not in self.data["quadnames"]:
            raise ValueError(f"Invalid quadrupole: {quadname}.")
        quad = self.devices[quadname]
        return float(quad.strength)

    def set_quad_strength(
        self, quadname, strength, ignore_timeout=False, **kw
    ):
        """."""
        tab = kw.pop("tab", 0)
        if quadname not in self.data["quadnames"]:
            raise ValueError(f"Invalid quadrupole: {quadname}.")
        quad = self.devices[quadname]
        quad.strength = float(strength)

        if ignore_timeout:
            return self.STATUS.Success

        if not quad.wait_float(
            "KLRef-Mon",
            strength,
            rel_tol=0.0,
            abs_tol=0.05 * self.params.quad_delta_kl,
            timeout=self.params.wait_quadrupole,
        ):
            msg = f'Could not change quadrupole "{quadname}" strength!'
            msg += f"\nTryed to set KL = {strength}, "
            msg += f"current KL = {quad.strength} [1/m]."
            self._log(msg, tab=tab)
            return self.STATUS.Fail
        return self.STATUS.Success

    def correct_orbit(self):
        """."""
        if not self.havebeam:
            return
        sofb = self.devices["sofb"]
        sofb.correct_orbit_manually(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr,
        )

    def _process_data_single_bpm(self, bpmname):
        if bpmname not in self.data["measure"]:
            return

        meas = self.data["measure"][bpmname]
        data_pos = meas["pos"]
        data_neg = meas["neg"]
        quadmode = meas["quadmode"]

        if data_pos is None or data_neg is None:
            return

        bpmnames = self.data["bpmnames"]
        bpmidx = bpmnames.index(bpmname)

        fs = float(data_pos["sampling_frequency"])
        dt = 1.0 / fs
        fh = float(data_pos.get("ch_freq", self.params.ch_freq))
        fv = float(data_pos.get("cv_freq", self.params.cv_freq))

        freqs = _np.array([fh, fv], dtype=float)

        orbx_pos = _np.asarray(data_pos["orbx"], dtype=float)
        orby_pos = _np.asarray(data_pos["orby"], dtype=float)
        orbx_neg = _np.asarray(data_neg["orbx"], dtype=float)
        orby_neg = _np.asarray(data_neg["orby"], dtype=float)

        npts = min(
            orbx_pos.shape[0],
            orbx_neg.shape[0],
            orby_pos.shape[0],
            orby_neg.shape[0],
        )
        orbx_pos = orbx_pos[:npts]
        orby_pos = orby_pos[:npts]
        orbx_neg = orbx_neg[:npts]
        orby_neg = orby_neg[:npts]

        tim = _np.arange(npts) * dt

        nr_cycles = _np.array(
            [
                int(round(self.params.excit_time * fh)),
                int(round(self.params.excit_time * fv)),
            ],
            dtype=int,
        )

        mat = self.fitting_matrix(tim, freqs, num_cycles=nr_cycles)
        u, s, vt = _np.linalg.svd(mat, full_matrices=False)
        pinv = vt.T / s @ u.T

        dcx_pos = _np.mean(orbx_pos, axis=0)
        dcy_pos = _np.mean(orby_pos, axis=0)
        dcx_neg = _np.mean(orbx_neg, axis=0)
        dcy_neg = _np.mean(orby_neg, axis=0)

        cosx_pos, sinx_pos, _ = self.fit_fourier_components(
            orbx_pos - dcx_pos, freqs, dt, pinv=pinv
        )
        cosy_pos, siny_pos, _ = self.fit_fourier_components(
            orby_pos - dcy_pos, freqs, dt, pinv=pinv
        )
        cosx_neg, sinx_neg, _ = self.fit_fourier_components(
            orbx_neg - dcx_neg, freqs, dt, pinv=pinv
        )
        cosy_neg, siny_neg, _ = self.fit_fourier_components(
            orby_neg - dcy_neg, freqs, dt, pinv=pinv
        )

        amp_x_pos, ph_x_pos = self.fit_calc_amp_and_phase(cosx_pos, sinx_pos)
        amp_y_pos, ph_y_pos = self.fit_calc_amp_and_phase(cosy_pos, siny_pos)
        amp_x_neg, ph_x_neg = self.fit_calc_amp_and_phase(cosx_neg, sinx_neg)
        amp_y_neg, ph_y_neg = self.fit_calc_amp_and_phase(cosy_neg, siny_neg)

        phref_h_pos = ph_x_pos[bpmidx]
        phref_v_pos = ph_y_pos[bpmidx]
        phref_h_neg = ph_x_neg[bpmidx]
        phref_v_neg = ph_y_neg[bpmidx]

        if quadmode == self.params.QUAD_MODULATION_MODE.DC:
            sgn_xh_pos = _np.sign(_np.cos(ph_x_pos[0] - phref_h_pos))
            sgn_yh_pos = _np.sign(_np.cos(ph_y_pos[0] - phref_h_pos))
            sgn_xv_pos = _np.sign(_np.cos(ph_x_pos[1] - phref_v_pos))
            sgn_yv_pos = _np.sign(_np.cos(ph_y_pos[1] - phref_v_pos))
            sgn_xh_neg = _np.sign(_np.cos(ph_x_neg[0] - phref_h_neg))
            sgn_yh_neg = _np.sign(_np.cos(ph_y_neg[0] - phref_h_neg))
            sgn_xv_neg = _np.sign(_np.cos(ph_x_neg[1] - phref_v_neg))
            sgn_yv_neg = _np.sign(_np.cos(ph_y_neg[1] - phref_v_neg))

            sgn_xh_pos[sgn_xh_pos == 0] = 1.0
            sgn_yh_pos[sgn_yh_pos == 0] = 1.0
            sgn_xv_pos[sgn_xv_pos == 0] = 1.0
            sgn_yv_pos[sgn_yv_pos == 0] = 1.0
            sgn_xh_neg[sgn_xh_neg == 0] = 1.0
            sgn_yh_neg[sgn_yh_neg == 0] = 1.0
            sgn_xv_neg[sgn_xv_neg == 0] = 1.0
            sgn_yv_neg[sgn_yv_neg == 0] = 1.0

            sxh_pos = amp_x_pos[0] * sgn_xh_pos
            syh_pos = amp_y_pos[0] * sgn_yh_pos
            sxv_pos = amp_x_pos[1] * sgn_xv_pos
            syv_pos = amp_y_pos[1] * sgn_yv_pos
            sxh_neg = amp_x_neg[0] * sgn_xh_neg
            syh_neg = amp_y_neg[0] * sgn_yh_neg
            sxv_neg = amp_x_neg[1] * sgn_xv_neg
            syv_neg = amp_y_neg[1] * sgn_yv_neg

            d_x = dcx_pos - dcx_neg
            d_y = dcy_pos - dcy_neg
            d_xh = sxh_pos - sxh_neg
            d_yh = syh_pos - syh_neg
            d_xv = sxv_pos - sxv_neg
            d_yv = syv_pos - syv_neg
        elif quadmode == self.params.QUAD_MODULATION_MODE.AC:
            msg = "AC quadrupole modulation mode not implemented yet."
            raise NotImplementedError(msg)
        else:
            quadmode_str = self.params.QUAD_MODULATION_MODE._field[quadmode]
            msg = f"Invalid quadrupole modulation mode: {quadmode_str}"
            raise ValueError(msg)

        y_h = d_x * d_yv - d_xv * d_y
        y_v = d_xh * d_y - d_x * d_yh
        x_h = d_xh * d_yv - d_xv * d_yh
        x_v = d_xh * d_yv - d_xv * d_yh

        m_h = _np.polyfit(x_h, y_h, 1)[0]
        m_v = _np.polyfit(x_v, y_v, 1)[0]

        x0_pos = (
            dcx_pos[bpmidx] + sxh_pos[bpmidx] * m_h + sxv_pos[bpmidx] * m_v
        )
        y0_pos = (
            dcy_pos[bpmidx] + syv_pos[bpmidx] * m_v + syh_pos[bpmidx] * m_h
        )

        x0_neg = (
            dcx_neg[bpmidx] + sxh_neg[bpmidx] * m_h + sxv_neg[bpmidx] * m_v
        )
        y0_neg = (
            dcy_neg[bpmidx] + syv_neg[bpmidx] * m_v + syh_neg[bpmidx] * m_h
        )

        x0 = 0.5 * (x0_pos + x0_neg)
        y0 = 0.5 * (y0_pos + y0_neg)

        self.analysis[bpmname] = dict(
            x0=x0,
            y0=y0,
            m_h=m_h,
            m_v=m_v,
            d_x=d_x,
            d_y=d_y,
            d_xh=d_xh,
            d_yh=d_yh,
            d_xv=d_xv,
            d_yv=d_yv,
            amp_x_pos=amp_x_pos,
            amp_y_pos=amp_y_pos,
            amp_x_neg=amp_x_neg,
            amp_y_neg=amp_y_neg,
            ph_x_pos=ph_x_pos,
            ph_y_pos=ph_y_pos,
            ph_x_neg=ph_x_neg,
            ph_y_neg=ph_y_neg,
        )

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
