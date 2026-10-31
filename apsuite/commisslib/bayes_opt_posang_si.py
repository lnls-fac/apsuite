"""."""

import logging as _log
from datetime import timedelta as _timedelta
import time as _time
from threading import Thread as _Thread, Event as _Event
import numpy as _np

from siriuspy.devices import PosAng, CurrInfoSI, EVG, PowerSupplyPU, InjCtrl
from siriuspy.clientarch import PVDataSet, Time

from ..optimization.bayesian_optimizer import (
    BayesianOptimizerGPy as _BayesianOptimizerGPy,
    BayesianOptimizerGPyParams as _Params,
)


class OptimizePosAngParams(_Params):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.limit_lower = _np.array([-2.0, -2.0, -2.0, -2.0, -1.265])
        self.limit_upper = _np.array([+2.0, +2.0, +2.0, +2.0, -1.035])
        # order: posx, angx, posy, angy, kick_nlk
        # nlk lims defined by calculating 10% variation range from current
        # strength.
        self.max_number_evals = _np.inf
        self.max_number_iters = _np.inf
        self.acq_func_type = self.AcqFuncType.UpperConfidenceBound
        self.beta_param = 0.01  # UCB exploitation/exploration trade-off
        self.automatic_relevance_determination = True
        self.gp_optimize_every_n_iters = 1  # GP hyperparams fit frequency
        self.num_restarts_gp = 2  # GP hyperparams fit restarts
        self.num_restarts_acq = 2  # Acq. func. optimization restarts
        self.seed = 0
        self.trust_region_mode = self.TrustRegionMode.FixedTrustRegion
        self.trust_region_scale = 0.02
        # --- warm-start data ---
        self.nr_points_warm_start_data = 10  # inj data from last 10 pulses
        self.initial_sampling_mode = self.InitialSamplingMode.ProvidedData
        self.initial_data_positions = None
        self.initial_data_objectives = None
        self.trigger_injection = False  # will not pulse EVG
        # will wait for inj pulses (during accumulation or top-up)

    def __str__(self):
        """."""
        stg = super().__str__()
        stg += self._TMPD(
            'nr_points_warm_start_data', self.nr_points_warm_start_data, ''
        )
        return stg


class OptimizePosAng(_BayesianOptimizerGPy):
    """Bayesian optimization of PosAng (posx, angx, posy, angy) + NLK."""

    def __init__(self, isonline=True, use_thread=True):
        """."""
        super().__init__(
            use_thread=use_thread,
            params=OptimizePosAngParams(),
            isonline=isonline,
        )
        if self.isonline:
            self._create_devices()

        if self.isonline and not self.params.trigger_injection:
            self.news = _Event()  # signals whether injection pulse happened
            # inj_charge_pv = self.devices['currinfo'].pv_object('InjCharge-Mon')
            # inj_current_pv = self.devices['currinfo'].pv_object('InjCurr-Mon')
            inj_efficiency_pv = self.devices['currinfo'].pv_object(
                'InjEff-Mon'
            )
            self._monitor_pv = inj_efficiency_pv
            # self._monitor_pv = inj_charge_pv  # alternatives to be tested
            # self._monitor_pv = inj_current_pv  # may be useful to select only
            # valid, significant pulses with charge or current above certain
            # threshold.
            self._monitor_pv.auto_monitor = True
            self._thread_update = None  # Thread launched to monitor when
            # injection pulses happen

    def _initialization(self):
        """."""
        if not self.isonline:
            raise RuntimeError(
                'Cannot start optimization offline. Set self.isonline to True'
            )

        if not self.params.nr_points_warm_start_data:
            raise ValueError(
                'nr_points_warm_start_data is zero. '
                + 'No baseline surrogate model for optimization!'
            )  # TODO: if no data, proceed w/ initial sampling

        self.pulse_interval = (
            self.devices['injctrl'].topup_period * 60
            if self.devices['injctrl'].topup_state_str != 'Off'
            else 3  # hard-coded accumulation period
        )

        _log.debug('Collecting data for model warm-start')
        self._collect_warm_start_data()

        if not self.params.initial_position.size:
            _log.warning(
                'Initial position not specified. '
                + 'Using current TS pos, ang and NLK kicks.'
            )
            self.params.initial_position = self.get_positions()

        _log.debug('Acquiring machine initial state...')
        self.initial_state = self._get_state()

        if self.params.trigger_injection:
            _log.debug('Preparing EVG...')
            self._prepare_evg()
        else:
            self._monitor_pv.add_callback(self._monitor_pv_update)


        # _log.info('Updating PosAng reference...')
        # self.devices['posang'].cmd_update_reference()
        # warm start will not work if reference is update
        # considering working w/ correctors & septa strengths directly

        self.data['timestamp'] = _time.time()
        return True

    def _collect_warm_start_data(self):
        currinfo_devname = self.devices['currinfo'].devname
        posang_devname = self.devices['posang'].devname
        nlk_devname = self.devices['nlk'].devname

        posx_pvname = posang_devname + ':' + 'DeltaPosX-SP'
        posy_pvname = posang_devname + ':' + 'DeltaPosY-SP'
        angx_pvname = posang_devname + ':' + 'DeltaAngX-SP'
        angy_pvname = posang_devname + ':' + 'DeltaAngY-SP'
        nlk_kick_pvname = nlk_devname + ':' + 'Kick-SP'
        injeff_pvname = currinfo_devname + ':' + 'InjEff-Mon'

        pvnames = [
            posx_pvname,
            angx_pvname,
            posy_pvname,
            angy_pvname,
            nlk_kick_pvname,
            injeff_pvname,
        ]

        pvdata = PVDataSet(pvnames=pvnames)
        now = Time.now()
        time_stop = now
        time_start = now - _timedelta(
            seconds=self.pulse_interval * self.params.nr_points_warm_start_data
            + 1
        )
        pvdata.time_start = time_start
        pvdata.time_stop = time_stop
        pvdata.update()

        data_positions = []
        for pvname in pvnames[:-1]:
            data_positions.append(pvdata[pvname].value)
        data_positions = _np.array(data_positions).reshape(
            -1, len(pvnames[:-1])
        )
        data_objectives = -_np.array(pvdata[injeff_pvname].value)

        self.params.initial_data_positions = data_positions
        self.params.initial_data_objectives = data_objectives

    def _prepare_evg(self):
        """."""
        evg = self.devices['evg']
        # configure to inject on first bucket just once
        evg.bucketlist = [1]
        evg.nrpulses = 1
        evg.cmd_update_events()
        _time.sleep(1)

    def _finalization(self):
        """."""
        if not self.params.trigger_injection:
            _log.debug('Clearing callbacks and news event')
            self._monitor_pv.clear_callbacks()
            self.news.clear()

        _log.debug('Restoring machine initial state...')
        self._set_state(self.initial_state)
        return super()._finalization()

    def objective_function(self, pos=None):
        """."""
        if pos is None:
            pos = self.get_positions()
        self.set_positions(pos)
        injeff = self._inject()
        return -injeff

    def get_positions(self):
        """."""
        posang = self.devices['posang']
        posx = posang.delta_posx
        angx = posang.delta_angx
        posy = posang.delta_posy
        angy = posang.delta_angy
        kick_nlk = self.devices['nlk'].strength
        return _np.array([posx, angx, posy, angy, kick_nlk])

    def set_positions(self, pos):
        """."""
        if pos is None:
            _log.warning('pos is None. Not setting positions.')
            return

        posang, nlk = self.devices['posang'], self.devices['nlk']

        _log.debug('Varying positions, angles and NLK strength...')
        posx, angx, posy, angy, kick_nlk = pos
        posang.delta_posx = posx
        posang.delta_angx = angx
        posang.delta_posy = posy
        posang.delta_angy = angy
        nlk.set_strength(kick_nlk, wait_mon=False)  # wait mon?
        _log.debug('\tDone!')

    def _get_state(self):
        injc = self.devices['injctrl']
        evg = self.devices['evg']

        state = dict()
        state['pos'] = self.get_positions()
        state['topup_standby_ts_injsept_g1'] = injc.topup_standby_ts_injsept_g1
        state['topup_standby_ts_injsept_g2'] = injc.topup_standby_ts_injsept_g2
        state['topup_standby_ts_injsept_f'] = injc.topup_standby_ts_injsept_f
        state['topup_standby_si_injnlkckr'] = injc.topup_standby_si_injnlkckr
        state['evg_nrpulses'] = evg.nrpulses
        state['evg_bucketlist'] = evg.bucketlist

        return state

    def _set_state(self, state):
        pos = state.get('pos', None)
        self.set_positions(pos)

        injc = self.devices['injctrl']
        evg = self.devices['evg']

        injc.topup_standby_ts_injsept_g1 = state['topup_standby_ts_injsept_g1']
        injc.topup_standby_ts_injsept_g2 = state['topup_standby_ts_injsept_g2']
        injc.topup_standby_ts_injsept_f = state['topup_standby_ts_injsept_f']
        injc.topup_standby_si_injnlkckr = state['topup_standby_si_injnlkckr']
        evg.nrpulses = state['evg_nrpulses']
        evg.bucketlist = state['evg_bucketlist']

    def _create_devices(self):
        self.devices['posang'] = PosAng(PosAng.DEVICES.TS)
        self.devices['nlk'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_NLKCKR
        )
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['injctrl'] = InjCtrl()

    def _inject(self):
        evg, currinfo = self.devices['evg'], self.devices['currinfo']

        if not self.params.trigger_injection:
            _log.warning('Not injecting. Waiting for an external pulse.')
            while not self.news.wait(self.pulse_interval):
                _log.warning('\tTimed out waiting for injection pulse...')
                if self._stopevt.is_set():
                    _log.warning('Stopped by user. Exiting.')
                    return
            self.news.clear()
            _log.debug('\tInjection pulse received.')

        else:
            _log.debug('Injecting...')
            evg.cmd_turn_on_injection()
            evg.wait_injection_finish()
            # TODO: wait accum period
            _log.debug('\tdone!')

        return currinfo.injeff

    def _monitor_pv_update(self, pvname, value, **kwargs):
        self._thread_update = _Thread(target=self._update_flag, daemon=True)
        self._thread_update.start()

    def _update_flag(self):
        _time.sleep(0.05)
        # if self._monitor_pv > threshold_val:
        # possible condition for recognizing valid pulses
        self.news.set()
