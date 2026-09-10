"""."""
import time as _time
import logging as _log
from threading import Event
import numpy as _np

from siriuspy.epics import PV, CAThread as _Thread
from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoBO, EVG, \
    EGTriggerPS, LILLRF, InjCtrl, PosAng, DCCT, Trigger, ASLLRF

from ..optimization.rcds import RCDS as _RCDS, RCDSParams as _RCDSParams


class OptimizeInjBOParams(_RCDSParams):
    """."""
    KNOB_DEFS = {

        # Linac lenses
        'li_lens1': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.pvs['li_lens1'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_lens1'], 'value', v
                ),
        },

        'li_lens2': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.pvs['li_lens2'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_lens2'], 'value', v
                ),
        },

        'li_lens3': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.pvs['li_lens3'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_lens3'], 'value', v
                ),
        },

        'li_lens4': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.pvs['li_lens4'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_lens4'], 'value', v
                ),
        },

        # Linac solenoids
        'li_slnd1': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd1'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd1'], 'value', v
                ),
        },

        'li_slnd2': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd2'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd2'], 'value', v
                ),
        },

        'li_slnd3': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd3'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd3'], 'value', v
                ),
        },

        'li_slnd4': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd4'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd4'], 'value', v
                ),
        },

        'li_slnd5': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd5'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd5'], 'value', v
                ),
        },

        'li_slnd6': {
            'lower': 0.0, # [A] 
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd6'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd6'], 'value', v
                ),
        },

        'li_slnd7': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd7'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd7'], 'value', v
                ),
        },

        'li_slnd8': {
            'lower': 0.0, # [A] 
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd8'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd8'], 'value', v
                ),
        },

        'li_slnd9': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd9'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd9'], 'value', v
                ),
        },

        'li_slnd10': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd10'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd10'], 'value', v
                ),
        },

        'li_slnd11': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd11'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd11'], 'value', v
                ),
        },

        'li_slnd12': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd12'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd12'], 'value', v
                ),
        },

        'li_slnd13': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd13'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd13'], 'value', v
                ),
        },

        'li_slnd14': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd14'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd14'], 'value', v
                ),
        },

        'li_slnd15': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd15'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd15'], 'value', v
                ),
        },

        'li_slnd16': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd16'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd16'], 'value', v
                ),
        },

        'li_slnd17': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd17'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd17'], 'value', v
                ),
        },

        'li_slnd18': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd18'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd18'], 'value', v
                ),
        },

        'li_slnd19': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd19'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd19'], 'value', v
                ),
        },

        'li_slnd20': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd20'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd20'], 'value', v
                ),
        },

        'li_slnd21': {
            'lower': 0.0, # [A]
            'upper': 35.0,
            'get': lambda self: self.pvs['li_slnd21'].value,
            'set': lambda self, v: setattr(
                self.pvs['li_slnd21'], 'value', v
                ),
        },

        # Linac quadrupoles
        'li_qf1': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.devices['li_qf1'].current,
            'set': lambda self, v: setattr(
                self.devices['li_qf1'], 'current', v
                ),
        },

        'li_qf2': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.devices['li_qf2'].current,
            'set': lambda self, v: setattr(
                self.devices['li_qf2'], 'current', v
                ),
        },

        'li_qf3': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.devices['li_qf3'].current,
            'set': lambda self, v: setattr(
                self.devices['li_qf3'], 'current', v
                ),
        },

        'li_qd1': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.devices['li_qd1'].current,
            'set': lambda self, v: setattr(
                self.devices['li_qd1'], 'current', v
                ),
        },

        'li_qd2': {
            'lower': -5.0, # [A]
            'upper': 5.0,
            'get': lambda self: self.devices['li_qd2'].current,
            'set': lambda self, v: setattr(
                self.devices['li_qd2'], 'current', v
                ),
        },

        # TB quadrupoles
        'tb_qf1': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qf1'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qf1'], 'current', v
                ),
        },

        'tb_qd1': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qd1'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qd1'], 'current', v
                ),
        },

        'tb_qf2a': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qf2a'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qf2a'], 'current', v
                ),
        },

        'tb_qd2a': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qd2a'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qd2a'], 'current', v
                ),
        },

        'tb_qf2b': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qf2b'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qf2b'], 'current', v
                ),
        },

        'tb_qd2b': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qd2b'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qd2b'], 'current', v
                ),
        },

        'tb_qf3': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qf3'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qf3'], 'current', v
                ),
        },

        'tb_qd3': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qd3'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qd3'], 'current', v
                ),
        },

        'tb_qf4': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qf4'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qf4'], 'current', v
                ),
        },

        'tb_qd4': {
            'lower': -10.0, # [A]
            'upper': 10.0,
            'get': lambda self: self.devices['tb_qd4'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_qd4'], 'current', v
                ),
        },

        # Position / angle
        'posx': {
            'lower': -2.0, # [mm]
            'upper': 2.0,
            'get': lambda self: self.devices['pos_ang'].delta_posx,
            'set': lambda self, v: setattr(
                self.devices['pos_ang'], 'delta_posx', v
                ),
        },

        'angx': {
            'lower': -1.0, # [mrad]
            'upper': 1.0,
            'get': lambda self: self.devices['pos_ang'].delta_angx,
            'set': lambda self, v: setattr(
                self.devices['pos_ang'], 'delta_angx', v
                ),
        },

        'posy': {
            'lower': -2.0, # [mm]
            'upper': 2.0,
            'get': lambda self: self.devices['pos_ang'].delta_posy,
            'set': lambda self, v: setattr(
                self.devices['pos_ang'], 'delta_posy', v
                ),
        },

        'angy': {
            'lower': -1.0, # [mrad]
            'upper': 1.0,
            'get': lambda self: self.devices['pos_ang'].delta_angy,
            'set': lambda self, v: setattr(
                self.devices['pos_ang'], 'delta_angy', v
                ),
        },

        'injkckr': {
            'lower': -25.0, # 
            'upper': -19.0,
            'get': lambda self: self.devices['injkckr'].strength,
            'set': lambda self, v: setattr(
                self.devices['injkckr'], 'strength', v
                ),
        },

        # TB correctors / septum
        'tb_ch1': {
            'lower': -10.0,
            'upper': 10.0,
            'get': lambda self: self.devices['tb_ch1'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_ch1'], 'current', v
                ),
        },

        'tb_injsept': {
            'lower': -776.69,
            'upper': 0.0,
            'get': lambda self: self.devices['tb_injsept'].strength,
            'set': lambda self, v: setattr(
                self.devices['tb_injsept'], 'strength', v
                ),
        },

        'tb_cv1': {
            'lower': -10.0,
            'upper': 10.0,
            'get': lambda self: self.devices['tb_cv1'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_cv1'], 'current', v
                ),
        },

        'tb_cv2': {
            'lower': -10.0,
            'upper': 10.0,
            'get': lambda self: self.devices['tb_cv2'].current,
            'set': lambda self, v: setattr(
                self.devices['tb_cv2'], 'current', v
                ),
        },

        # Linac LLRF
        'shb_amp': {
            'lower': 20,
            'upper': 40,
            'get': lambda self: self.devices[
                'li_llrf'].dev_shb.amplitude,
            'set': lambda self, v: setattr(
                self.devices['li_llrf'].dev_shb, 'amplitude', v
                ),
        },

        'kly1_amp': {
            'lower': 85,
            'upper': 91,
            'get': lambda self: self.devices[
                'li_llrf'].dev_klystron1.amplitude,
            'set': lambda self, v: setattr(
                self.devices['li_llrf'].dev_klystron1, 'amplitude', v
                ),
        },

        'kly2_amp': {
            'lower': 70,
            'upper': 76,
            'get': lambda self: self.devices[
                'li_llrf'].dev_klystron2.amplitude,
            'set': lambda self, v: setattr(
                self.devices['li_llrf'].dev_klystron2, 'amplitude', v
                ),
        },

        'shb_phs': {
            'lower': 160,
            'upper': 180,
            'get': lambda self: self.devices[
                'li_llrf'].dev_shb.phase,
            'set': lambda self, v: setattr(
                self.devices['li_llrf'].dev_shb, 'phase', v
                ),
        },

        'kly1_phs': {
            'lower': -180,
            'upper': -150,
            'get': lambda self: self.devices[
                'li_llrf'].dev_klystron1.phase,
            'set': lambda self, v: setattr(
                self.devices['li_llrf'].dev_klystron1, 'phase', v
                ),
        },

        'kly2_phs': {
            'lower': -20,
            'upper': 0,
            'get': lambda self: self.devices[
                'li_llrf'].dev_klystron2.phase,
            'set': lambda self, v: setattr(
                self.devices['li_llrf'].dev_klystron2, 'phase', v
                ),
        },

        # Booster RF
        'borf_amp': {
            'lower': 30,
            'upper': 80,
            'get': lambda self: self.devices[
                'bo_llrf'].voltage_bottom,
            'set': lambda self, v: setattr(
                self.devices['bo_llrf'], 'voltage_bottom', v
                ),
        },

        'borf_phs': {
            'lower': 90,
            'upper': 160,
            'get': lambda self: self.devices[
                'bo_llrf'].phase_bottom,
            'set': lambda self, v: setattr(
                self.devices['bo_llrf'], 'phase_bottom', v
                ),
        },
    }
    
    KNOBS = list(KNOB_DEFS.keys())
    
    def __init__(self):
        """."""
        super().__init__()
        self._knobs = self.KNOBS
        
        self.limit_lower = _np.array([
            self.KNOB_DEFS[knob]['lower']
            for knob in self._knobs
        ])
        
        self.limit_upper = _np.array([
            self.KNOB_DEFS[knob]['upper']
            for knob in self._knobs
        ])

        self.initial_position = _np.array([
            0.5 * (
                self.KNOB_DEFS[knob]['lower']
                + self.KNOB_DEFS[knob]['upper']
            )
            for knob in self._knobs
        ])
        
        self.initial_search_directions = _np.eye(
            len(self._knobs), dtype=float
        )
        
        self.curr_wfm_index = 100
        self.nrpulses = 5
        self.use_median = False
        self.wait_between_injections = 3  # [s]
        self.trigger_injection = False

        self.pos0 = None

    def __str__(self):
        """."""
        stg = '-----  RCDS Parameters  -----\n\n'
        stg += super().__str__()
        stg += '\n\n-----  OptimizeInjBO Parameters  -----\n\n'
        stg += self._TMPS('trigger_injection', str(self.trigger_injection), '')
        stg += self._TMPD('curr_wfm_index', self.curr_wfm_index, '')
        stg += self._TMPD('nrpulses', self.nrpulses, '')
        stg += self._TMPD(
            'wait_between_injections', self.wait_between_injections, '[s]'
        )
        stg += self._TMPS('use_median', str(self.use_median), '')
        stg += self._TMPS('knobs', ', '.join(self._knobs), '')

        return stg

    @property
    def knobs(self):
        """Define the knobs and limits appropriately."""
        return self._knobs

    @knobs.setter
    def knobs(self, knobs):
        """Define the knobs and limits appropriately."""
        for kn in knobs:
            if kn not in self.KNOBS:
                raise ValueError(f'Knob {kn} is not a valid knob.')

        self._knobs = knobs
        self.limit_lower = _np.array([
            self.KNOB_DEFS[k]['lower'] for k in knobs
        ])

        self.limit_upper = _np.array([
            self.KNOB_DEFS[k]['upper'] for k in knobs
        ])

        self.initial_position = _np.array([
            0.5 * (
                self.KNOB_DEFS[k]['lower']
                + self.KNOB_DEFS[k]['upper']
            )
            for k in knobs
        ])

        self.initial_search_directions = _np.eye(
            len(knobs)
        )


class OptimizeInjBO(_RCDS):
    """."""

    def __init__(self, isonline=True, use_thread=True):
        """."""
        _RCDS.__init__(self, isonline=isonline, use_thread=use_thread)
        self.params = OptimizeInjBOParams()
        self.data['positions'] = []
        self.data['currents'] = []

        if self.isonline:
            self._create_devices()

        self.news = Event()  # signals whether a new injection pulse happened
        self._curr150mev_pv = self.devices['currinfo'].pv_object(
            'Current150MeV-Mon'
        )  # 150MeV current PV: will be monitored to singal an injection pulse
        self._curr150mev_pv.auto_monitor = True
        self._thread_update = None  # Thread to monitor injection pulses
        # initialized to None because not needed yet

    def prepare_evg(self):
        """Prepare EVG for optimization."""
        evg = self.devices['evg']
        # configure to inject on first bucket just once
        evg.bucketlist = [1]
        evg.nrpulses = 1
        evg.cmd_update_events()
        _time.sleep(1)

    def objective_function(self, pos=None, apply=True):
        """."""
        pos0 = self.get_current_position()

        if pos is not None:
            self.set_position_to_machine(pos)
            self.data['positions'].append(pos)
            _time.sleep(2)  # revisit this once position setter is refactored
        else:
            self.data['positions'].append(pos0)
        self.news.clear()

        injcurrs = list()
        obj = None
        for i in range(self.params.nrpulses):
            inj = self.inject_beam_and_get_current()
            if inj is None:
                break
            injcurrs.append(inj)
            _time.sleep(self.params.wait_between_injections)
        else:
            injcurrs = _np.array(injcurrs)
            self.data['currents'].append(injcurrs)
            func = _np.median if self.params.use_median else _np.mean
            obj = - func(injcurrs)

        if pos is not None and not apply:
            self.set_position_to_machine(pos0)

        return obj

    def inject_beam_and_get_current(self):
        """Inject beam and get injected current, if desired."""
        idx = self.params.curr_wfm_index
        if not self.params.trigger_injection:
            while not self.news.wait(15):  # revisit this wait time
                _log.warning('Timed out waiting for injection.')
                if self._stopevt.is_set():
                    _log.warning('Stopped by user. Exiting')
                    return
            self.news.clear()
            return self.devices['dcct'].current_fast[idx]

        inj0 = self.devices['dcct'].current_fast[idx]
        self.devices['evg'].cmd_turn_on_injection(wait_rb=True)
        self.devices['evg'].wait_injection_finish()
        for _ in range(50):
            inj = self.devices['dcct'].current_fast[idx]
            if inj0 != inj:
                break
            _time.sleep(0.1)
        else:
            _log.warning('Timed out waiting current to update.')
        return inj

    def measure_objective_function_noise(self, nr_evals, pos=None):
        """."""
        self._curr150mev_pv.add_callback(self._curr150mev_update)

        if pos is None:
            pos = self.get_current_position()
        obj = []
        for i in range(nr_evals):
            obj.append(self.objective_function(pos))
            _log.info(f'{i+1:02d}/{nr_evals:02d}  --> obj. = {obj[-1]:.3f}')
        noise_level = _np.std(obj)
        self.params.noise_level = noise_level
        self.data['measured_objfuncs_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level

        self._curr150mev_pv.clear_callbacks()
        return noise_level, obj

    def get_current_position(self):
        """Return the values of the knobs.

        Returns:
            numpy.ndarray (N,): vector of knobs values.

        """
        
        return _np.array([
            self.params.KNOB_DEFS[knob]['get'](self)
            for knob in self.params.knobs
            ])

    def set_position_to_machine(self, pos):
        """."""
        if len(pos) != len(self.params.knobs):
            raise ValueError(
                'Length of pos must match number of knobs selected.')

        for knob, value in zip(self.params.knobs, pos):
            self.params.KNOB_DEFS[knob]['set'](self, value)

        self.wait_set_pos(pos, timeout=10)

    def _create_devices(self):
        # knobs devices

        # lenses
        self.pvs["li_lens1"] = PV("LI-01:PS-Lens-1:Current-SP")
        self.pvs["li_lens2"] = PV("LI-01:PS-Lens-2:Current-SP")
        self.pvs["li_lens3"] = PV("LI-01:PS-Lens-3:Current-SP")
        self.pvs["li_lens4"] = PV("LI-01:PS-Lens-4:Current-SP")

        # solenoids
        self.pvs["li_slnd1"] = PV("LI-01:PS-Slnd-1:Current-SP")
        self.pvs["li_slnd2"] = PV("LI-01:PS-Slnd-2:Current-SP")
        self.pvs["li_slnd3"] = PV("LI-01:PS-Slnd-3:Current-SP")
        self.pvs["li_slnd4"] = PV("LI-01:PS-Slnd-4:Current-SP")
        self.pvs["li_slnd5"] = PV("LI-01:PS-Slnd-5:Current-SP")
        self.pvs["li_slnd6"] = PV("LI-01:PS-Slnd-6:Current-SP")
        self.pvs["li_slnd7"] = PV("LI-01:PS-Slnd-7:Current-SP")
        self.pvs["li_slnd8"] = PV("LI-01:PS-Slnd-8:Current-SP")
        self.pvs["li_slnd9"] = PV("LI-01:PS-Slnd-9:Current-SP")
        self.pvs["li_slnd10"] = PV("LI-01:PS-Slnd-10:Current-SP")
        self.pvs["li_slnd11"] = PV("LI-01:PS-Slnd-11:Current-SP")
        self.pvs["li_slnd12"] = PV("LI-01:PS-Slnd-12:Current-SP")
        self.pvs["li_slnd13"] = PV("LI-01:PS-Slnd-13:Current-SP")
        self.pvs["li_slnd14"] = PV("LI-Fam:PS-Slnd-14:Current-SP")
        self.pvs["li_slnd15"] = PV("LI-Fam:PS-Slnd-15:Current-SP")
        self.pvs["li_slnd16"] = PV("LI-Fam:PS-Slnd-16:Current-SP")
        self.pvs["li_slnd17"] = PV("LI-Fam:PS-Slnd-17:Current-SP")
        self.pvs["li_slnd18"] = PV("LI-Fam:PS-Slnd-18:Current-SP")
        self.pvs["li_slnd19"] = PV("LI-Fam:PS-Slnd-19:Current-SP")
        self.pvs["li_slnd20"] = PV("LI-Fam:PS-Slnd-20:Current-SP")
        self.pvs["li_slnd21"] = PV("LI-Fam:PS-Slnd-21:Current-SP")

        # LI quads
        props2init = ['Current-SP', 'Current-RB']
        self.devices['li_qf1'] = PowerSupply(
            'LI-Fam:PS-QF1', props2init=props2init
        )
        self.devices['li_qf2'] = PowerSupply(
            'LI-Fam:PS-QF2', props2init=props2init
        )
        self.devices['li_qf3'] = PowerSupply(
            'LI-01:PS-QF3', props2init=props2init
        )
        self.devices['li_qd1'] = PowerSupply(
            'LI-01:PS-QD1', props2init=props2init
        )
        self.devices['li_qd2'] = PowerSupply(
            'LI-01:PS-QD2', props2init=props2init
        )

        # TB quads
        props2init = ['Current-SP', 'Current-RB']
        self.devices['tb_qf1'] = PowerSupply(
            'TB-01:PS-QF1', props2init=props2init
        )
        self.devices['tb_qd1'] = PowerSupply(
            'TB-01:PS-QD1', props2init=props2init
        )
        self.devices['tb_qf2a'] = PowerSupply(
            'TB-02:PS-QF2A', props2init=props2init
        )
        self.devices['tb_qd2a'] = PowerSupply(
            'TB-02:PS-QD2A', props2init=props2init
        )
        self.devices['tb_qf2b'] = PowerSupply(
            'TB-02:PS-QF2B', props2init=props2init
        )
        self.devices['tb_qd2b'] = PowerSupply(
            'TB-02:PS-QD2B', props2init=props2init
        )
        self.devices['tb_qf3'] = PowerSupply(
            'TB-03:PS-QF3', props2init=props2init
        )
        self.devices['tb_qd3'] = PowerSupply(
            'TB-03:PS-QD3', props2init=props2init
        )
        self.devices['tb_qf4'] = PowerSupply(
            'TB-04:PS-QF4', props2init=props2init
        )
        self.devices['tb_qd4'] = PowerSupply(
            'TB-04:PS-QD4', props2init=props2init
        )

        # TB PosAng & Injkicker
        self.devices['pos_ang'] = PosAng(PosAng.DEVICES.TB)
        self.devices['injkckr'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.BO_INJ_KCKR
        )

        # TB PosAng knobs (alternative to PosAng)
        self.devices["tb_ch1"] = PowerSupply("TB-04:PS-CH-1")
        self.devices["tb_injsept"] = PowerSupplyPU("TB-04:PU-InjSept")
        self.devices["tb_cv1"] = PowerSupply("TB-04:PS-CV-1")
        self.devices["tb_cv2"] = PowerSupply("TB-04:PS-CV-2")

        # LI LLRF
        self.devices['li_llrf'] = LILLRF()

        # BO LLRF
        self.devices['bo_llrf'] = ASLLRF(ASLLRF.DEVICES.BO)

        # other relevant devices
        self.devices['ejekckr'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.BO_EJE_KCKR
        )
        self.devices['currinfo'] = CurrInfoBO()
        self.devices['dcct'] = DCCT(DCCT.DEVICES.BO)
        self.devices['evg'] = EVG()
        self.devices['ejekckr_trig'] = Trigger("BO-48D:TI-EjeKckr")
        self.devices['egun_trigps'] = EGTriggerPS()
        self.devices['injctrl'] = InjCtrl(props2init=None)

    def _curr150mev_update(self, pvname, value, **kwargs):
        self._thread_update = _Thread(target=self._update_flag, daemon=True)
        self._thread_update.start()

    def _update_flag(self):
        _time.sleep(0.05)
        if not self.devices['currinfo'].current3gev:
            self.news.set()
            # only signals as valid injection pulses
            # those for which there is no transmission to the
            # storage ring (non-top-up pulses)

    def _initialization(self):
        """."""
        if not super()._initialization():
            return False
        self.data['timestamp'] = _time.time()
        self.data['positions'] = []
        self.data['currents'] = []
        if self.params.trigger_injection:
            self.prepare_evg()
        self._curr150mev_pv.add_callback(self._curr150mev_update)
        self.pos0 = self.get_current_position()
        return True

    def _finalization(self):
        self._curr150mev_pv.clear_callbacks()
        self.news.clear()
        _log.info('Reseting machine initial position')
        self.set_position_to_machine(self.pos0)
        return super()._finalization()

    def wait_set_pos(self, pos, timeout=20):
        """Wait positions RB reach the desired SP vals.

        Wait current RB values reach `pos` within a `tol` precision up to
        `timeout` seconds.

        Args:
            pos (array): reference positions that have been set to SP PVs
            tol (float): relative tolerance for comparing values.
                i. e. |current_pos - pos| <= tol * |pos|. Defaults to 0.05
            timeout (float): timeout in seconds. Defaults to 10 s.
        """
        sleep_time = 0.1
        it = int(timeout/sleep_time)
        for _ in range(it):
            pos_ = self.get_current_position()
            if _np.all(_np.isclose(pos_, pos, atol=1e-4)):
                _log.info('Positions have been set.')
                break
            _time.sleep(sleep_time)
        else:
            _log.warning('Timed out waiting positions be set.')
            _log.warning(f'Diff: {pos - pos_}')
