"""Main module."""

import datetime as _datetime
import operator as _opr
import time as _time
from copy import deepcopy as _dcopy
from functools import reduce as _red

import numpy as _np
from mathphys.functions import (
    get_namedtuple as _get_namedtuple,
    load as _load,
    save as _save,
)
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient
from siriuspy.devices import (
    EVG as _EVG,
    SOFB as _SOFB,
    CurrInfoSI as _CurrInfoSI,
    Event as _Event,
    FamBPMs as _FamBPMs,
    PowerSupply as _PowerSupply,
    RFGen as _RFGen,
    StrengthConv as _StrengthConv,
    Trigger as _Trigger,
    Tune as _Tune,
)
from siriuspy.search import LLTimeSearch as _LLTime
from siriuspy.sofb.csdev import SOFBFactory as _SOFBFactory

from apsuite.commisslib.meas_ac_orm import MeasACORM as _MeasACORM
from apsuite.commisslib.measure_bba import BBAParams as _BBAParams
from apsuite.utils import (
    ParamsBaseClass as _ParamsBaseClass,
    ThreadedMeasBaseClass as _BaseClass,
)


class ACBBAParams(_ParamsBaseClass):
    """Parameters for AC-BBA."""

    BPMNAMES = _BBAParams.BPMNAMES
    QUADNAMES = _BBAParams.QUADNAMES
    QUAD_MODULATION_MODE = _get_namedtuple('QuadModulationMode', ['AC', 'DC'])

    def __init__(self):
        """."""
        super().__init__()
        self.timeout_bpms = 60  # [s]
        self.timeout_magnets = 20  # [s]

        self._quad_modulation_mode = ACBBAParams.QUAD_MODULATION_MODE.DC
        self.quad_delta_kl = 0.01  # [1/m]
        self.wait_quadrupole = 2.0  # [s]

        self.cv_freq = 17.0  # [Hz]
        self.ch_freq = 23.0  # [Hz]
        self.q_freq = 5.0  # [Hz]

        self.excit_time = 4  # [s]
        self.corrs_delay = 5e-3  # [s]

        self.ch_kick = 5  # [urad]
        self.cv_kick = 5  # [urad]
        # self.dorbx = 100.0  # [um]
        # self.dorby = 100.0  # [um]
        # self.use_normalized_kicks = False

        self.measure_bpms_noise = True
        self.acq_rate = 'FAcq'
        self.orm_name = 'ref_respmat'

        self.sofb_maxcorriter = 5
        self.sofb_maxorberr = 5  # [um]
        self.correct_orbit_each_step = True

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += 'AC-BBA Parameters:\n'
        stg += ftmp('timeout_bpms', self.timeout_bpms, '[s]')
        stg += ftmp('timeout_magnets', self.timeout_magnets, '[s]')
        stg += stmp('quad_modulation_mode', self.quad_modulation_mode_str, '')
        stg += ftmp('quad_delta_kl', self.quad_delta_kl, '[1/m]')
        stg += ftmp('wait_quadrupole', self.wait_quadrupole, '[s]')
        stg += ftmp('cv_freq', self.cv_freq, '[Hz]')
        stg += ftmp('ch_freq', self.ch_freq, '[Hz]')
        stg += ftmp('q_freq', self.q_freq, '[Hz]')
        stg += ftmp('excit_time', self.excit_time, '[s]')
        stg += ftmp('corrs_delay', self.corrs_delay, '[s]')
        stg += ftmp('ch_kick', self.ch_kick, '[urad]')
        stg += ftmp('cv_kick', self.cv_kick, '[urad]')
        stg += ftmp('dorbx', self.dorbx, '[um]')
        stg += ftmp('dorby', self.dorby, '[um]')

        stg += stmp('measure_bpms_noise', str(self.measure_bpms_noise), '')
        stg += stmp('acq_rate', self.acq_rate, '')
        stg += stmp('orm_name', self.orm_name, '')
        stg += dtmp('sofb_maxcorriter', self.sofb_maxcorriter, '')
        stg += ftmp('sofb_maxorberr', self.sofb_maxorberr, '[um]')
        stg += stmp(
            'correct_orbit_each_step', str(self.correct_orbit_each_step), ''
        )
        return stg

    @property
    def quad_modulation_mode(self):
        """Quadrupole Modulation Mode (int)."""
        return self._quad_modulation_mode

    @property
    def quad_modulation_mode_str(self):
        """Quadrupole Modulation Mode (str)."""
        return self.QUAD_MODULATION_MODE._fields[self._quad_modulation_mode]

    @quad_modulation_mode.setter
    def quad_modulation_mode(self, value):
        fields = self.QUAD_MODULATION_MODE._fields
        if isinstance(value, str) and value.upper() in fields:
            self._quad_modulation_mode = fields.index(value)
        elif value in self.QUAD_MODULATION_MODE:
            self._quad_modulation_mode = int(value)
        else:
            raise ValueError(
                "Invalid Quadrupole Modulation Mode! Select " +
                "(int) 0 or 1 | (str) 'AC' or 'DC'"
            )


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

        "trigquads_source",
        "trigquads_nr_pulses",
        "trigquads_delay_raw",
        "trigquads_delta_delay_raw",

        "trigskews_source",
        "trigskews_nr_pulses",
        "trigskews_delay_raw",
        "trigskews_delta_delay_raw",

        "evt_mode",
        "evt_delay_raw",
    )

    def __init__(self, isonline=True):
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
        self._bpms_corrs_mapping = {}
        self.sofb_data = None
        self.configdb_orm = None
        self._orm = None

        if self.isonline:
            self.sofb_data = _SOFBFactory.create("SI")
            self.configdb_orm = _ConfigDBClient(config_type="si_orbcorr_respm")
            self._create_devices()

        self.data["log"] = [(_time.time(), "Started.")]

    # ----- Imported methods -----

    fitting_matrix = staticmethod(_MeasACORM.fitting_matrix)
    fit_fourier_components = classmethod(_MeasACORM.fit_fourier_components)
    fit_calc_amp_and_phase = staticmethod(_MeasACORM.fit_calc_amp_and_phase)

    # ----- Properties -----

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

    @property
    def bpms_corrs_mapping(self):
        """Mapping of which CH and CV to excite for each BPM."""
        return self._bpms_corrs_mapping.copy()

    @bpms_corrs_mapping.setter
    def bpms_corrs_mapping(self, value):
        for bpm, (ch, cv) in enumerate(value.items()):
            if bpm not in self.data["bpmnames"]:
                raise ValueError(f"Invalid BPM: {bpm}!")
            if ch not in self.sofb_data.ch_names:
                raise ValueError(f"Invalid CH: {ch} for BPM {bpm}!")
            if cv not in self.sofb_data.cv_names:
                raise ValueError(f"Invalid CV: {cv} for BPM {bpm}!")
        self._bpms_corrs_mapping = value.copy()

    @property
    def havebeam(self):
        """."""
        cinfo = self.devices["currinfo"]
        return cinfo.connected and cinfo.storedbeam

    # ----- Setup devices -----

    def _create_devices(self):
        """Create and connect to devices."""
        # BPMs
        self.bpms = _FamBPMs(mturn_signals2acq="XY", props2init="acq")
        self.devices["fambpms"] = self.bpms

        # Quadrupoles
        props = [
            "PwrState-Sts",
            "KL-SP",
            "KL-RB",
            "KLRef-Mon",
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
        for qname in self.data["quadnames"]:
            if qname in self.devices:
                continue
            self.devices[qname] = _PowerSupply(qname, props2init=props)

        # Correctors
        sofbdata = self.sofb_data
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

        # Strength converters
        self.devices.update({
            n + ":StrengthConv": _StrengthConv(n, "Ref-Mon") for n in (
                sofbdata.ch_names + sofbdata.cv_names + self.data["quadnames"]
            )
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

        # Correctors, Quads and Skews Triggers
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
        self.devices["trigquads"] = _Trigger(
            "SI-Glob:TI-Mags-QTrims", props2init=props
        )
        self.devices["trigskews"] = _Trigger(
            "SI-Glob:TI-Mags-Skews", props2init=props
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
        self._orm = _np.array(self.configdb_orm.get_config_value(name))

    # ----- SOFB utils -----

    def correct_orbit(self):
        """."""
        if not self.havebeam:
            return
        sofb = self.devices["sofb"]
        sofb.correct_orbit_manually(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr,
        )

    # ----- Quadrupole handling -----

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

    def get_quad_strength_limits(self, quadname, margin=0.0005):
        """."""
        if quadname not in self.data["quadnames"]:
            raise ValueError(f"Invalid quadrupole: {quadname}.")
        quad = self.devices[quadname]
        pv = quad.pv_object("KL-SP")
        upp = pv.upper_disp_limit
        low = pv.lower_disp_limit
        # Limits are interchanged in some quads:
        lolim = min(upp, low) + margin
        hilim = max(upp, low) - margin
        return _np.array([lolim, hilim], dtype=float)

    def check_isvalid_dkl(
            self,
            bpm_names=None,
            strengths=None,
            strength_limits=None,
            margin=0.0005,
            return_valid=False,
        ):
        """."""
        quad_names = self.data['quadnames']
        bpms = bpm_names if bpm_names else self.data['bpmnames']
        quads = [quad_names[self.data['bpmnames'].index(bpm)] for bpm in bpms]

        if strengths is None:
            strengths = [self.get_quad_strength(quad) for quad in quads]

        if strength_limits is None:
            strength_limits = [
                self.get_quad_strength_limits(quad, margin) for quad in quads
            ]

        if len(bpms) != len(strengths):
            msg = 'Size mismatch between the group and init_strengths: '
            msg += f'{len(bpms)} != {len(strengths)}!'
            raise ValueError(msg)

        if len(strengths) != len(strength_limits):
            msg = 'Size mismatch between init_strengths and strength_limits: '
            msg += f'{len(strengths)} != {len(strength_limits)}!'
            raise ValueError(msg)

        ok = True
        valid = strengths.copy()
        for idx, quad in enumerate(quads):
            stren = strengths[idx]
            dkl = abs(self.params.quad_delta_kl)
            lolim, hilim = strength_limits[idx]
            clow, chigh = lolim + dkl / 2, hilim - dkl / 2
            if clow > chigh:
                self._log(f'ERR: {quad}, dKL = {dkl:.3g} too high!')
                valid[idx] = None
            else:
                valid[idx] = _np.clip(stren, clow, chigh)
            if valid[idx] != stren:
                msg = f'WARN: {quad}, '
                msg += f'KL = {stren:.3g}, '
                msg += f'dKL = {dkl:.3g}, '
                msg += f'limits = ({lolim:.3g}, {hilim:.3g}). '
                msg += f'Change KL to: {valid[idx]}'
                self._log(msg)
                ok = False
        if return_valid:
            return ok, valid
        return ok

    # ----- A -----

    # ----- A -----

    # ----- A -----

    def _do_acbba(self):
        """."""
        # Initial checkings
        if not all([
            self.check_isvalid_dkl(bpm) for bpm in self._bpms2dobba
        ]):
            self._log("Adjust quad strength or change dKL first.")
            return

        # Initialize data
        self.data["measure"] = dict()
        self._setup_orm()

        # Start
        tini = _datetime.datetime.fromtimestamp(_time.time())
        msg = f"Starting measurement at {tini.strftime('%Y-%m-%d %Hh%Mm%Ss')}"
        self._log("\n" + msg)

        sofb = self.devices['sofb']
        if sofb.autocorrsts:
            msg = "SOFB feedback is enabled. Please disable it first."
            self._log(msg)
            return self.STATUS.Fail

        if sofb.synckicksts != sofb._data.CorrSync.Off:
            msg = "SOFB correctors synchronization is On. Please turn it Off."
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
            "ch_freq": self.params.ch_freq,
            "cv_freq": self.params.cv_freq,
            "q_freq": self.params.q_freq,
            "data_dc_kl_pos": None,
            "data_dc_kl_neg": None,
            "data_ac": None,
            "quadmode": quadmode,
            "quad_stren_ini": stren_ini,
            "quad_delta_kl": delta_kl,
        }

        if quadmode == self.params.QUAD_MODULATION_MODE.AC:
            msg = "Setup: Quadrupole modulation mode: AC."
            self._log(msg, tab=tab)
            sts, data_ac = self._acquire_data(
                chname,
                cvname,
                quad_name=quadname,
                delta_kl=delta_kl,
                tab=tab + 1
            )
            data["data_ac"] = data_ac
            if sts == self.STATUS.Fail:
                return sts, data

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

    def _acquire_data(self, ch_name, cv_name, quad_name=None, **kw):
        """."""
        tab = kw.pop("tab", 0)

        magnets = [ch_name, cv_name]
        strengths = [self.params.ch_kick, self.params.cv_kick]
        excit_freqs = [self.params.ch_freq, self.params.cv_freq]
        excit_time = self.params.excit_time
        tout_bpms = self.params.timeout_bpms
        tout_mags = self.params.timeout_magnets

        if quad_name is not None:
            magnets += [quad_name]
            excit_freqs += [self.params.q_freq]
            strengths += [self.params.quad_delta_kl/2]

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
            quads=[[quad_name]],
            nr_points=nr_points,
        )
        msg = f"Done! ET: {_time.time() - t00:.2f}s"
        self._log(msg)

        # Configure correctors
        t01 = _time.time()
        msg = "Configuring correctors... "
        self._log(msg, tab=tab, end="")
        self._config_magnets(magnets, strengths, excit_freqs, excit_time)
        msg = f"Done! ET: {_time.time() - t01:.2f}s"
        self._log(msg)

        # Configure correctors opmode to Cycle
        t02 = _time.time()
        msg = f"Changing Mags. ({', '.join(magnets)}) OpMode to Cycle... "
        self._log(msg, tab=tab, end="")
        if not self._change_mags_opmode("cycle", magnets, tab=tab):
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
        msg = f"Restoring Mags. ({', '.join(magnets)}) OpMode to SlowRef... "
        self._log(msg, tab=tab, end="")
        if not self._wait_cycle_to_finish(magnets, timeout=tout_mags):
            msg = "Fail! Cycle still not finished."
            self._log(msg)
            return self.STATUS.Fail, data
        if not self._change_mags_opmode("slowref", magnets, tab=tab):
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

        sofb = self.sofb_data
        bpmnames = self.data["bpmnames"]

        if bpmname in self._bpms_corrs_mapping:
            ch_name, cv_name = self._bpms_corrs_mapping[bpmname]
            ch_idx = sofb.ch_names.index(ch_name)
            cv_idx = sofb.cv_names.index(cv_name)
            return ch_name, cv_name, ch_idx, cv_idx + sofb.nr_ch

        if orm is None:
            orm = self._orm
        if orm is None:
            raise RuntimeError("Orbit Response Matrix not loaded.")

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

    def _config_timing(
        self,
        cm_dly=0,
        chs=None,
        cvs=None,
        quads=None,
        nr_points=None
    ):
        state = dict()
        state["trigbpms_source"] = "Study"
        state["trigbpms_nr_pulses"] = 1
        state["trigbpms_delay_raw"] = 0.0

        state["evt_mode"] = "External"
        state["evt_delay_raw"] = 0

        state["trigcorrs_source"] = "Study"
        state["trigcorrs_nr_pulses"] = 1

        if quads is not None:
            state["trigquads_source"] = "Study"
            state["trigquads_nr_pulses"] = 1

            state["trigskews_source"] = "Study"
            state["trigskews_nr_pulses"] = 1

        rf_freq = self.devices["rfgen"].frequency
        ftim = rf_freq / 4  # timing base frequency
        dly = int(cm_dly * ftim)
        if chs is None or cvs is None or nr_points is None:
            state["trigcorrs_delay_raw"] = dly
            state["trigquads_delay_raw"] = dly
            state["trigskews_delay_raw"] = dly
            self.set_timing_state(state)
            return

        state["trigcorrs_delay_raw"] = 0
        state["trigquads_delay_raw"] = 0
        state["trigskews_delay_raw"] = 0
        # nr_runs = len(chs)
        # # Calculate delta_delay for correctors to be as close as possible to a
        # # multiple of the the sampling period to ensure repeatability of
        # # experiment along runs excited during single acquisition:
        # fsamp = self.bpms.get_sampling_frequency(rf_freq, self.params.acq_rate)
        # runs_delta_dly = _np.arange(nr_runs, dtype=float)
        # runs_delta_dly *= nr_points / fsamp
        # runs_delta_dlyr = _np.round(runs_delta_dly * ftim)

        # # get low level trigger names to be configured in each run of the
        # # acquisition:
        # ll_trigs_corrs = []
        # ll_trigs_quads = []
        # ll_trigs_skews = []
        # if quads is None:
        #     quads = _np.full_like(chs, False, dtype=bool).tolist()
        # for ch, cv, quad in zip(chs, cvs, quads):  # noqa: B905
        #     llt_corrs = set()
        #     llt_quads = set()
        #     llt_skews = set()
        #     for c in ch + cv + quad:
        #         if not c:
        #             continue
        #         trig = _LLTime.get_trigger_name(c + ":BCKPLN")
        #         if c.dev in ["CH", "CV"]:
        #             llt_corrs.add(trig)
        #         elif c.dev == "QS":
        #             llt_skews.add(trig)
        #         else:
        #             llt_quads.add(trig)
        #     ll_trigs_corrs.append(llt_corrs)
        #     ll_trigs_quads.append(llt_quads)
        #     ll_trigs_skews.append(llt_skews)

        # # check if correctors controlled by the same trigger are requested to
        # # be triggered in different times during the same acquisition
        # for ll_trigs in [ll_trigs_corrs, ll_trigs_quads, ll_trigs_skews]:
        #     if len(_red(_opr.or_, ll_trigs)) != \
        #         _red(_opr.add, map(len, ll_trigs)):
        #         raise ValueError("Impossible trigger configuration requested.")

        # for trig_type, ll_trigs in zip(  # noqa: B905
        #     ["trigcorrs", "trigquads", "trigskews"],
        #     [ll_trigs_corrs, ll_trigs_quads, ll_trigs_skews]
        #     ):
        #     trig = self.devices[trig_type]
        #     delta_delay_raw = _np.zeros(trig.delta_delay_raw.size)
        #     low_level = trig.low_level_triggers
        #     for llts, ddlyr in zip(ll_trigs, runs_delta_dlyr):  # noqa: B905
        #         # Find all ll triggers of this sector and set their delay:
        #         for llt in llts:
        #             if llt not in low_level:
        #                 raise ValueError(f"Trigger {llt:s} is not valid.")
        #             delta_delay_raw[low_level.index(llt)] = ddlyr + dly
        #     state[f"{trig_type}_delta_delay_raw"] = delta_delay_raw

        self.set_timing_state(state)

    def _log(self, msg, *args, **kwargs):
        """."""
        if "tab" in kwargs:
            tab = kwargs.pop("tab")
            msg = "  " * tab + msg
        if self.verbose:
            print(msg, *args, **kwargs)
        self.data["log"].append((_time.time(), msg))

    def _config_magnets(self, magnets, strengths, freqs, excit_time):
        """."""
        for i, cmn in enumerate(magnets):
            cmo = self.devices[cmn]
            conv = self.devices[cmn + ":StrengthConv"].conv_strength_2_current
            cmo.cycle_type = cmo.CYCLETYPE.Sine
            cmo.cycle_freq = freqs[i]
            cmo.cycle_ampl = conv(strengths[i])
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

    def _change_mags_opmode(self, mode, magnets=None, timeout=None, **kw):
        """."""
        tab = kw.pop("tab", 0)
        if timeout is None:
            timeout = self.params.timeout_magnets

        opm_sel = _PowerSupply.OPMODE_SEL
        opm_sts = _PowerSupply.OPMODE_STS
        mode_sel = opm_sel.Cycle if mode == "cycle" else opm_sel.SlowRef
        mode_sts = opm_sts.Cycle if mode == "cycle" else opm_sts.SlowRef

        quadmod_mode = self.params.quad_modulation_mode
        if magnets is None:
            magnets = self.sofb_data.ch_names + self.sofb_data.cv_names
            if quadmod_mode == self.params.QUAD_MODULATION_MODE.AC:
                magnets += self.data["quadnames"]

        for magname in magnets:
            mag = self.devices[magname]
            mag.opmode = mode_sel

        for magname in magnets:
            dt_ = _time.time()
            mag = self.devices[magname]
            if not mag.wait("OpMode-Sts", mode_sts, timeout=timeout):
                msg = "\nERR:" + mag + " did not change to " + mode
                self._log(msg, tab=tab)
                return False
            dt_ -= _time.time()
            timeout = max(timeout + dt_, 0)
            mag.current = mag.current
        return True

    def _wait_cycle_to_finish(self, magnets=None, timeout=None):
        """."""
        if timeout is None:
            timeout = self.params.timeout_magnets

        quadmod_mode = self.params.quad_modulation_mode
        if magnets is None:
            magnets = self.sofb_data.ch_names + self.sofb_data.cv_names
            if quadmod_mode == self.params.QUAD_MODULATION_MODE.AC:
                magnets += self.data["quadnames"]

        t0 = _time.time()
        for magname in magnets:
            mag = self.devices[magname]
            dt = timeout - (_time.time() - t0)
            if dt < 0 or not mag.wait_cycle_to_finish(timeout=dt):
                return False
        return True

    def _process_data_single_bpm(self, bpmname, phase_adjust=0):
        if bpmname not in self.data["measure"]:
            return

        meas = self.data["measure"][bpmname]
        quadmode = meas["quadmode"]

        bpmnames = self.data["bpmnames"]
        bpmidx = bpmnames.index(bpmname)

        if quadmode == self.params.QUAD_MODULATION_MODE.DC:

            data_pos = meas["pos"]
            data_neg = meas["neg"]

            if data_pos is None or data_neg is None:
                return

            fs = float(data_pos["sampling_frequency"])
            dt = 1.0 / fs
            fh = float(data_pos.get("ch_freq", self.params.ch_freq))
            fv = float(data_pos.get("cv_freq", self.params.cv_freq))

            freqs = _np.array([fh, fv], dtype=float)

            orbx_pos = _np.asarray(data_pos["orbx"], dtype=float)
            orby_pos = _np.asarray(data_pos["orby"], dtype=float)
            orbx_neg = _np.asarray(data_neg["orbx"], dtype=float)
            orby_neg = _np.asarray(data_neg["orby"], dtype=float)

            npts = orbx_pos.shape[0]
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
            amp_x_neg, ph_x_neg = self.fit_calc_amp_and_phase(cosx_neg, sinx_neg)

            amp_y_pos, ph_y_pos = self.fit_calc_amp_and_phase(cosy_pos, siny_pos)
            amp_y_neg, ph_y_neg = self.fit_calc_amp_and_phase(cosy_neg, siny_neg)

            phref_h_pos = ph_x_pos[:, bpmidx]
            phref_h_neg = ph_x_neg[:, bpmidx]
            phref_v_pos = ph_y_pos[:, bpmidx]
            phref_v_neg = ph_y_neg[:, bpmidx]

            f = phase_adjust
            sgn_xh_pos = _np.sign(_np.cos(ph_x_pos[0] - phref_h_pos[0] * f))
            sgn_xv_pos = _np.sign(_np.cos(ph_x_pos[1] - phref_v_pos[1] * f))
            sgn_xh_neg = _np.sign(_np.cos(ph_x_neg[0] - phref_h_neg[0] * f))
            sgn_xv_neg = _np.sign(_np.cos(ph_x_neg[1] - phref_v_neg[1] * f))
            sgn_yh_pos = _np.sign(_np.cos(ph_y_pos[0] - phref_h_pos[0] * f))
            sgn_yv_pos = _np.sign(_np.cos(ph_y_pos[1] - phref_v_pos[1] * f))
            sgn_yh_neg = _np.sign(_np.cos(ph_y_neg[0] - phref_h_neg[0] * f))
            sgn_yv_neg = _np.sign(_np.cos(ph_y_neg[1] - phref_v_neg[1] * f))

            sgn_xh_pos[sgn_xh_pos == 0] = 1.0
            sgn_xv_pos[sgn_xv_pos == 0] = 1.0
            sgn_xh_neg[sgn_xh_neg == 0] = 1.0
            sgn_xv_neg[sgn_xv_neg == 0] = 1.0

            sgn_yh_pos[sgn_yh_pos == 0] = 1.0
            sgn_yv_pos[sgn_yv_pos == 0] = 1.0
            sgn_yh_neg[sgn_yh_neg == 0] = 1.0
            sgn_yv_neg[sgn_yv_neg == 0] = 1.0

            sxh_pos = amp_x_pos[0] * sgn_xh_pos
            sxv_pos = amp_x_pos[1] * sgn_xv_pos
            sxh_neg = amp_x_neg[0] * sgn_xh_neg
            sxv_neg = amp_x_neg[1] * sgn_xv_neg

            syh_pos = amp_y_pos[0] * sgn_yh_pos
            syv_pos = amp_y_pos[1] * sgn_yv_pos
            syh_neg = amp_y_neg[0] * sgn_yh_neg
            syv_neg = amp_y_neg[1] * sgn_yv_neg

            d_x = dcx_pos - dcx_neg
            d_y = dcy_pos - dcy_neg
            d_xh = sxh_pos - sxh_neg
            d_yh = syh_pos - syh_neg
            d_xv = sxv_pos - sxv_neg
            d_yv = syv_pos - syv_neg

            y_h = -(d_x * d_yv - d_xv * d_y)
            y_v = -(d_xh * d_y - d_x * d_yh)
            x_h = d_xh * d_yv - d_xv * d_yh
            x_v = d_xh * d_yv - d_xv * d_yh

            m_h = _np.polyfit(x_h, y_h, 1)[0]
            m_v = _np.polyfit(x_v, y_v, 1)[0]

            x0_pos = (
                dcx_pos[bpmidx] + sxh_pos[bpmidx] * m_h + sxv_pos[bpmidx] * m_v
            )
            x0_neg = (
                dcx_neg[bpmidx] + sxh_neg[bpmidx] * m_h + sxv_neg[bpmidx] * m_v
            )

            y0_pos = (
                dcy_pos[bpmidx] + syv_pos[bpmidx] * m_v + syh_pos[bpmidx] * m_h
            )
            y0_neg = (
                dcy_neg[bpmidx] + syv_neg[bpmidx] * m_v + syh_neg[bpmidx] * m_h
            )

            x0 = 0.5 * (x0_pos + x0_neg)
            y0 = 0.5 * (y0_pos + y0_neg)

            self.analysis[bpmname] = dict(
                tim=tim,
                x0=x0,
                y0=y0,
                x0pos=x0_pos,
                x0neg=x0_neg,
                y0pos=y0_pos,
                y0neg=y0_neg,
                dcx_pos=dcx_pos[bpmidx],
                dcy_pos=dcy_pos[bpmidx],
                dcx_neg=dcx_neg[bpmidx],
                dcy_neg=dcy_neg[bpmidx],
                sxh_pos=sxh_pos[bpmidx],
                sxh_neg=sxh_neg[bpmidx],
                sxv_pos=sxv_pos[bpmidx],
                sxv_neg=sxv_neg[bpmidx],
                syh_pos=syh_pos[bpmidx],
                syh_neg=syh_neg[bpmidx],
                syv_pos=syv_pos[bpmidx],
                syv_neg=syv_neg[bpmidx],
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

        elif quadmode == self.params.QUAD_MODULATION_MODE.AC:
            data = meas["zer"]

            if data is None:
                return

            fs = float(data["sampling_frequency"])
            dt = 1.0 / fs
            fh = float(data.get("ch_freq", self.params.ch_freq))
            fv = float(data.get("cv_freq", self.params.cv_freq))
            fq = float(data.get("q_freq", self.params.q_freq))

            # freqs = _np.array([fh, fv, fq], dtype=float)
            freqs = _np.array([fh - fq, fh + fq, fv - fq, fv + fq], dtype=float)

            orbx = _np.asarray(data["orbx"], dtype=float)
            orby = _np.asarray(data["orby"], dtype=float)

            npts = orbx.shape[0]
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

            dcx = _np.mean(orbx, axis=0)
            dcy = _np.mean(orby, axis=0)

            cosx, sinx, _ = self.fit_fourier_components(
                orbx - dcx, freqs, dt, pinv=pinv
            )
            cosy, siny, _ = self.fit_fourier_components(
                orby - dcy, freqs, dt, pinv=pinv
            )

            amp_x, ph_x = self.fit_calc_amp_and_phase(cosx, sinx)
            amp_y, ph_y = self.fit_calc_amp_and_phase(cosy, siny)

            self.analysis[bpmname] = dict(
                tim=tim,
                dcx=dcx,
                dcy=dcy,
                amp_x=amp_x,
                amp_y=amp_y,
                ph_x=ph_x,
                ph_y=ph_y,
            )

        else:
            quadmode_str = self.params.QUAD_MODULATION_MODE._field[quadmode]
            msg = f"Invalid quadrupole modulation mode: {quadmode_str}"
            raise ValueError(msg)
