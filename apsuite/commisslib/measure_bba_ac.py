"""Main module."""

import datetime as _datetime
# import operator as _opr  # unused (temporary?)
import time as _time
from copy import deepcopy as _dcopy
# from functools import reduce as _red  # unused (temporary?)

import numpy as _np
from mathphys.functions import get_namedtuple as _get_namedtuple
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
# from siriuspy.search import LLTimeSearch as _LLTime  # unused (temporary?)
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
        self.dorbx = 100.0  # [um]
        self.dorby = 100.0  # [um]
        self.use_normalized_kicks = True

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
        stg += stmp('use_normalized_kicks', str(self.use_normalized_kicks), '')
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

        self.data["log"] = list()

    # ----- Imported methods -----
    # public
    fitting_matrix = staticmethod(_MeasACORM.fitting_matrix)
    fit_fourier_components = _MeasACORM.fit_fourier_components
    fit_calc_amp_and_phase = staticmethod(_MeasACORM.fit_calc_amp_and_phase)

    # ----- Properties -----
    # public
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
        for bpm, (ch, cv) in value.items():
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
    # private
    def _create_devices(self):
        """Create and connect to devices."""
        # BPMs
        self.bpms = _FamBPMs(mturn_signals2acq="XY", props2init="acq")
        self.devices["fambpms"] = self.bpms

        # Quadrupoles
        ac_props = [
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
        props = ["PwrState-Sts", "KL-SP", "KL-RB", "KLRef-Mon"] + ac_props
        for qname in self.data["quadnames"]:
            if qname in self.devices:
                continue
            self.devices[qname] = _PowerSupply(qname, props2init=props)

        # Correctors
        sofbdata = self.sofb_data
        props = ["Kick-SP"] + ac_props
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
    # public
    def correct_orbit(self):
        """."""
        if not self.havebeam:
            return
        sofb = self.devices["sofb"]
        sofb.correct_orbit_manually(
            nr_iters=self.params.sofb_maxcorriter,
            residue=self.params.sofb_maxorberr,
        )

    # ----- Quadrupole utils -----
    # public
    def get_quad_strength(self, quadname):
        """."""
        if quadname not in self.data["quadnames"]:
            raise ValueError(f"Invalid quadrupole: {quadname}.")
        quad = self.devices[quadname]
        return float(quad.strength)

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

    # ----- Config Timing -----
    # public
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

    # private
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
        # # Calculate delta_delay for correctors to be as close as possible to
        # # a multiple of the the sampling period to ensure repeatability of
        # # experiment along runs excited during single acquisition:
        # fsamp = self.bpms.get_sampling_frequency(
        #     rf_freq, self.params.acq_rate
        # )
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
        #         msg = "Impossible trigger configuration requested."
        #         raise ValueError(msg)

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

    # ----- Config BPMs -----
    # private
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

        if bpmname not in bpmnames:
            raise ValueError("Invalid BPM! Check ACBBAParams.BPMNAMES")
        bpm_idx = bpmnames.index(bpmname)

        if orm is None:
            orm = self._orm
        if orm is None:
            raise RuntimeError("Orbit Response Matrix not loaded.")

        if bpmname in self._bpms_corrs_mapping:
            ch_name, cv_name = self._bpms_corrs_mapping[bpmname]
            ch_idx = sofb.ch_names.index(ch_name)
            cv_idx = sofb.cv_names.index(cv_name)
            r_xx = orm[bpm_idx, ch_idx]
            r_yy = orm[bpm_idx + sofb.nr_bpms, cv_idx + sofb.nr_ch]
            return ch_name, cv_name, ch_idx, cv_idx + sofb.nr_ch, r_xx, r_yy

        orm_xx = orm[: sofb.nr_bpms, : sofb.nr_ch]
        orm_yy = orm[sofb.nr_bpms :, sofb.nr_ch : sofb.nr_chcv]

        ch_idx = int(_np.argmax(_np.abs(orm_xx[bpm_idx, :])))
        cv_idx = int(_np.argmax(_np.abs(orm_yy[bpm_idx, :])))

        ch_name = sofb.ch_names[ch_idx]
        cv_name = sofb.cv_names[cv_idx]

        r_xx = orm[bpm_idx, ch_idx]
        r_yy = orm[bpm_idx + sofb.nr_bpms, cv_idx + sofb.nr_ch]
        return ch_name, cv_name, ch_idx, cv_idx + sofb.nr_ch, r_xx, r_yy

    # ----- Config Magnets -----
    # private
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

    # ----- Acquire data -----
    # public
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

    def get_magnets_data(self, magnets):
        """Get magnets related data.

        Args:
            magnets (list, tuple): magnets used in excitation

        Returns:
            dict: BPMs data.

        """
        data = dict()
        for mag in magnets:
            magnet = self.devices[mag]
            conv = self.devices[mag + ':StrengthConv'].conv_current_2_strength
            data[mag] = {
                "cycle_num_cycles": magnet.cycle_num_cycles,
                "cycle_ampl": magnet.cycle_ampl,
                "cycle_ampl_strength": conv(magnet.cycle_ampl),
                "cycle_type": magnet.cycle_type_str,
                "cycle_freq": magnet.cycle_freq,
                "cycle_offset": magnet.cycle_offset,
                "cycle_aux_param": _dcopy(magnet.cycle_aux_param)
            }
        for trigname in ("trigcorrs", "trigquads", "trigskews"):
            trig = self.devices[trigname]
            data[trigname + "_delay_raw"] = trig.delay_raw
            data[trigname + "_delta_delay_raw"] = trig.delta_delay_raw
        return data

    # private
    def _get_acq_nr_points(self):
        freq = self.devices["rfgen"].frequency
        rate = self.params.acq_rate
        n_pts = self.params.excit_time
        n_pts += self.params.corrs_delay * 2
        n_pts *= self.bpms.get_sampling_frequency(freq, acq_rate=rate)
        n_pts = int(_np.ceil(n_pts))
        return n_pts

    def _acquire_data(self, ch_name, cv_name, quad_name=None, **kw):
        """."""
        tab = kw.pop("tab", 0)
        r_xx = kw.pop("r_xx", None)
        r_yy = kw.pop("r_yy", None)

        ch_kick, cv_kick = self.params.ch_kick, self.params.cv_kick

        if r_xx is not None and self.params.use_normalized_kicks:
            ch_kick = self.params.dorbx / abs(r_xx)

        if r_yy is not None and self.params.use_normalized_kicks:
            cv_kick = self.params.dorby / abs(r_yy)

        magnets = [ch_name, cv_name]
        strengths = [ch_kick, cv_kick]
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
        data.update(self.get_magnets_data(magnets))
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

    def _measure_bpms_noise(self, **kw):
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

    # ----- Process data -----
    # private
    @staticmethod
    def _process_excit(
        orb_data, excit_time=0, idx_lim=(0, -1)
    ):
        """Process orbit excitation data for AC-BBA analysis.

        Args:
            orb_data (dict): Acquired orbit data.
            fh (float): CH excitation frequency [Hz].
            fv (float): CV excitation frequency [Hz].
            fq (float): quadrupole excitation frequency [Hz].
            fs (float): orbit sampling frequency [Hz].
            excit_time (float, int): excitation period [s].
            idx_lim (tuple, optional): Index limits for data slicing.
                Defaults to (0, -1).

        Returns:
            dict: Processed excitation data containing orbit, DC,
                amplitude, phase, and frequencies.

        Raises:
            ValueError: If any of the excitation frequencies are zero.

        """
        fh, fv, fq = 0.0, 0.0, None
        for key in orb_data:
            if isinstance(key, str) and 'PS' in key:
                freq = orb_data[key]['cycle_freq']
                if 'CH' in key:
                    fh = freq
                elif 'CV' in key:
                    fv = freq
                else:
                    fq = freq
        freqs = [fh, fv]
        if fq is not None:
            freqs += [fq, fh + fq, fh - fq, fv + fq, fv - fq]
        freqs = _np.asarray(freqs, dtype=float)

        if _np.any(freqs == 0):
            raise ValueError('Invalid freq!')

        orbx = _np.asarray(orb_data['orbx'], dtype=float)[slice(*idx_lim), :]
        orby = _np.asarray(orb_data['orby'], dtype=float)[slice(*idx_lim), :]

        dcx = _np.mean(orbx, axis=0)
        dcy = _np.mean(orby, axis=0)

        fs = orb_data.get('sampling_frequency', orbx.shape[0] / excit_time)
        dt = 1.0 / fs
        npts = orbx.shape[0]
        tim = _np.arange(npts) * dt
        nr_cycles = _np.asarray(
            [round(excit_time * f) for f in freqs], dtype=int)

        mat = _MeasACORM.fitting_matrix(tim, freqs, num_cycles=nr_cycles)
        u, s, vt = _np.linalg.svd(mat, full_matrices=False)
        pinv = vt.T / s @ u.T

        cosx, sinx, _ = _MeasACORM.fit_fourier_components(
            orbx - dcx, freqs, dt, pinv=pinv
        )
        cosy, siny, _ = _MeasACORM.fit_fourier_components(
            orby - dcy, freqs, dt, pinv=pinv
        )

        amp_x, ph_x = _MeasACORM.fit_calc_amp_and_phase(cosx, sinx)
        amp_y, ph_y = _MeasACORM.fit_calc_amp_and_phase(cosy, siny)

        return {
            'orb': _np.array([orbx, orby]),
            'dc': _np.array([dcx, dcy]),
            'amp': _np.array([amp_x, amp_y]),
            'phi': _np.array([ph_x, ph_y]),
            'freqs': freqs,
        }

    @staticmethod
    def _get_m_coefs(d_x, d_y, d_xh, d_yh, d_xv, d_yv):
        y_h = d_xv * d_y - d_yv * d_x
        y_v = d_yh * d_x - d_xh * d_y
        x_ = d_xh * d_yv - d_xv * d_yh

        m_v = _np.polyfit(x_, y_v, 1)[0]
        m_h = _np.polyfit(x_, y_h, 1)[0]

        return _np.vstack((m_h, m_v))

    @staticmethod
    def _find_intersec(data, bpmidx, idx_lim=(0, -1)):
        """Find the intersection of orbits at two quadrupole states (pos, neg).

        Args:
            data (dict): Orbit data for both quadrupole states.
            bpmidx (int): Index of the BPM for which to find the intersection.
            idx_lim (tuple, optional): Index limits for data slicing.
                Defaults to (0, -1).

        Returns:
            np.ndarray: Intersection points.

        """
        quadname = data['quadname']
        op = _np.array([
            _np.asarray(data['data_dc_kl_pos']['orbx'], dtype=float)[slice(*idx_lim), :],
            _np.asarray(data['data_dc_kl_pos']['orby'], dtype=float)[slice(*idx_lim), :],
        ])
        on = _np.array([
            _np.asarray(data['data_dc_kl_neg']['orbx'], dtype=float)[slice(*idx_lim), :],
            _np.asarray(data['data_dc_kl_neg']['orby'], dtype=float)[slice(*idx_lim), :],
        ])
        offs = []
        arrs = []
        idcs = _np.full((160,), True)
        idcs[bpmidx] = False
        for pl in [0, 1]:
            x = _np.array([op[pl, :, bpmidx], on[pl, :, bpmidx]])
            y = _np.array([op[pl, :, idcs], on[pl, :, idcs]])
            if quadname.dev == 'QS':
                y = _np.array([op[pl - 1, :, idcs], on[pl - 1, :, idcs]])

            a1, b1 = _np.polyfit(x[0, :], y[0, :, :].T, 1)
            a2, b2 = _np.polyfit(x[1, :], y[1, :, :].T, 1)
            da = a1 - a2
            db = b1 - b2
            offs.append(_np.dot(da, -db) / _np.dot(da, da))
            arrs.append(-db / da)
        return _np.asarray(offs)

    def _process_data_single_bpm(
            self, bpmname, idx_lim=(0, -1), calc_mode='old'
        ):
        """Process AC-BBA acquired orbit data for a given BPM.

        Args:
            data (dict): Acquired orbit data.
            bpmname (str): Name of the BPM.
            idx_lim (tuple, optional): Index limits for data slicing.
                Defaults to (0, -1).
            calc_mode (str, optional): AC mode calculation method.
                Defaults to 'old'.

        Returns:
            dict: Processed data for the specified BPM, including the 
                calculated offsets.
        """
        idx = self.data['bpmnames'].index(bpmname)
        data = self.data['measure'][bpmname]
        mode = data['quadmode']
        excit_time = data.get('excit_time', self.params.excit_time)

        if mode == self.params.QUAD_MODULATION_MODE.DC:
            f_h = data.get('ch_freq', 0)
            f_v = data.get('cv_freq', 0)
            if f_h != f_v:
                r_pos = DoACBBA._process_excit(
                    data['data_dc_kl_pos'], excit_time, idx_lim
                )
                r_neg = DoACBBA._process_excit(
                    data['data_dc_kl_neg'], excit_time, idx_lim
                )
                dc_pos = r_pos['dc']
                dc_neg = r_neg['dc']
                projs_pos = r_pos['amp']
                projs_neg = r_neg['amp']
                projs_pos *= _np.sign(_np.cos(r_pos['phi']))
                projs_neg *= _np.sign(_np.cos(r_neg['phi']))
                d_x, d_y = dc_pos - dc_neg
                (d_xh, d_xv), (d_yh, d_yv) = projs_pos - projs_neg
                m_ = DoACBBA._get_m_coefs(d_x, d_y, d_xh, d_yh, d_xv, d_yv)
                offset_pos = (
                    dc_pos[:, idx].reshape(2, 1) + projs_pos[:, :, idx] @ m_
                ).ravel()
                offset_neg = (
                    dc_neg[:, idx].reshape(2, 1) + projs_neg[:, :, idx] @ m_
                ).ravel()
                offset = 0.5 * (offset_pos + offset_neg)
            else:
                offset = DoACBBA._find_intersec(data, idx, idx_lim)

        elif mode == self.params.QUAD_MODULATION_MODE.AC:
            r = DoACBBA._process_excit(
                data['data_ac'], excit_time, mode, idx_lim
            )
            dc = r['dc']
            projs = r['amp']
            projs *= _np.sign(_np.cos(r['phi']))
            d_x = projs[0, 2, :]  # 2 = fq
            d_y = projs[1, 2, :]  # 2 = fq

            # Old calc
            if calc_mode == 'old':
                d_xh = _np.sum(
                    projs[0, [3, 4], :], axis=0
                )  # 3: fh + fq, 4: fh - fq
                d_xv = _np.sum(
                    projs[0, [5, 6], :], axis=0
                )  # 5: fv + fq, 6: fv - fq
                d_yh = _np.sum(
                    projs[1, [3, 4], :], axis=0
                )  # 3: fh + fq, 4: fh - fq
                d_yv = _np.sum(
                    projs[1, [5, 6], :], axis=0
                )  # 5: fv + fq, 6: fv - fq

            # New calc
            else:
                d_xh = (
                    2 * _np.take_along_axis(
                        projs[0, [3, 4], :],
                        _np.argmax(
                            _np.abs(projs[0, [3, 4], :]), axis=0, keepdims=True
                        ),
                        axis=0,
                    )[0]
                )  # 3 = fh + fq, 4 = fh - fq
                d_xv = (
                    2 * _np.take_along_axis(
                        projs[0, [5, 6], :],
                        _np.argmax(
                            _np.abs(projs[0, [5, 6], :]), axis=0, keepdims=True
                        ),
                        axis=0,
                    )[0]
                )  # 5 = fv + fq, 6 = fv - fq
                d_yh = (
                    2 * _np.take_along_axis(
                        projs[1, [3, 4], :],
                        _np.argmax(
                            _np.abs(projs[1, [3, 4], :]), axis=0, keepdims=True
                        ),
                        axis=0,
                    )[0]
                )  # 3 = fh + fq, 4 = fh - fq
                d_yv = (
                    2 * _np.take_along_axis(
                        projs[1, [5, 6], :],
                        _np.argmax(
                            _np.abs(projs[1, [5, 6], :]), axis=0, keepdims=True
                        ),
                        axis=0,
                    )[0]
                )  # 5 = fv + fq, 6 = fv - fq

            m_ = DoACBBA._get_m_coefs(d_x, d_y, d_xh, d_yh, d_xv, d_yv)
            offset = (
                dc[:, idx].reshape(2, 1)
                + projs[:, [0, 1], idx] @ m_  # 0 = fh, 1 = fv
            ).ravel()

        return offset

    # public
    def process_data(self, idx_lim=(0, -1), calc_mode='old'):
        """Process data and calculate all BPMs "BPM-to-quadrupole" offsets.

        Args:
            idx_lim (tuple, optional): _description_. Defaults to (0, -1).
            calc_mode (str, optional): _description_. Defaults to 'old'.
        """
        nbpms = len(_BBAParams.BPMNAMES)
        if 'orbx' not in self.analysis:
            self.analysis['orbx'] = _np.zeros(nbpms)
            self.analysis['orby'] = self.analysis['orbx'].copy()

        for bpm in self.data['measure']:
            idx = _BBAParams.BPMNAMES.index(bpm)
            x0, y0 = self._process_data_single_bpm(
                bpm, idx_lim=idx_lim, calc_mode=calc_mode
            )
            self.analysis['orbx'][idx] = x0
            self.analysis['orby'][idx] = y0

    # ----- Do AC-BBA -----
    # private
    def _do_acbba(self):
        """."""
        # Initial checkings
        if not self.check_isvalid_dkl(self._bpms2dobba):
            self._log("Adjust quad strength or change dKL first.")
            return

        # Initialize data
        self.data["log"] = list()
        self.data["measure"] = dict()
        self._setup_orm()

        # Start
        tini = _datetime.datetime.fromtimestamp(_time.time())
        msg = f"Starting measurement at {tini.strftime('%Y-%m-%d %Hh%Mm%Ss')}"
        self._log("\n" + msg)

        sofb = self.devices['sofb']
        sofb_checks = (
            bool(sofb.autocorrsts),
            sofb.synckicksts not in [
                sofb._data.CorrSync.Off,
                sofb._data.CorrSync.RmpBO
            ],
        )
        if any(sofb_checks):
            msg = (
                "SOFB feedback is enabled. Please disable it.",
                "SOFB Corr. Sync. must be Off or RmpBO."
            )[sofb_checks.index(True)]
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
            measnoise_ok, noise_data = self._measure_bpms_noise(tab=1)
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

        chname, cvname, _chi, _cvi, r_xx, r_yy = \
            self._get_correctors_for_bpm(bpmname, self._orm)

        stren_ini = self.get_quad_strength(quadname)
        delta_kl = self.params.quad_delta_kl

        data = {
            "quadname": quadname,
            "chname": chname,
            "cvname": cvname,
            "r_xx": r_xx,
            "r_yy": r_yy,
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
                tab=tab + 1,
                r_xx=r_xx,
                r_yy=r_yy,
            )
            data["data_ac"] = data_ac
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

            sts, data_pos = self._acquire_data(
                chname, cvname,
                tab=tab + 1,
                r_xx=r_xx,
                r_yy=r_yy
            )
            data["data_dc_kl_pos"] = data_pos
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

            sts, data_neg = self._acquire_data(
                chname,
                cvname,
                tab=tab + 1,
                r_xx=r_xx,
                r_yy=r_yy,
            )
            data["data_dc_kl_neg"] = data_neg
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

            return sts, data

        else:
            msg = "Invalid Quadrupole modulation mode. Skipping..."
            self._log(msg, tab=tab)
            return self.STATUS.Fail, data

    # ----- Logging -----
    # private
    def _log(self, msg, *args, **kwargs):
        """."""
        if "tab" in kwargs:
            tab = kwargs.pop("tab")
            msg = "  " * tab + msg
        if self.verbose:
            print(msg, *args, **kwargs)
        self.data["log"].append((_time.time(), msg))
