"""."""
import time as _time
import logging as _log

import numpy as _np

from siriuspy.epics import PV
from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoBO, EVG, \
    EGTriggerPS, LILLRF, InjCtrl, PosAng, DCCT, Trigger, ASLLRF

from ..optimization.rcds import RCDS as _RCDS, RCDSParams as _RCDSParams


class OptimizeInjBOParams(_RCDSParams):
    """."""

    KNOBS = [
        'li_lens1',
        'li_lens2',
        'li_lens3',
        'li_lens4',

        'li_slnd1',
        'li_slnd2',
        'li_slnd3',
        'li_slnd4',
        'li_slnd5',
        'li_slnd6',
        'li_slnd7',
        'li_slnd8',
        'li_slnd9',
        'li_slnd10',
        'li_slnd11',
        'li_slnd12',
        'li_slnd13',
        'li_slnd14',
        'li_slnd15',
        'li_slnd16',
        'li_slnd17',
        'li_slnd18',
        'li_slnd19',
        'li_slnd20',
        'li_slnd21',

        'li_qf1',
        'li_qf2',
        'li_qf3',
        'li_qd1',
        'li_qd2',

        'tb_qf2a',
        'tb_qf2b',
        'tb_qd2a',
        'tb_qd2b',

        'posx',
        'angx',
        'posy',
        'angy',
        'kckr',

        'shb_amp',
        'kly1_amp',
        'kly2_amp',
        'shb_phs',
        'kly1_phs',
        'kly2_phs',

        'borf_amp',
        'borf_phs',
    ]
    LIMS_UPPER = [
        +5.0,    # 'li_lens1',
        +5.0,    # 'li_lens2',
        +5.0,    # 'li_lens3',
        +5.0,    # 'li_lens4',

        +35.0,   # 'li_slnd1',
        +35.0,   # 'li_slnd2',
        +35.0,   # 'li_slnd3',
        +35.0,   # 'li_slnd4',
        +35.0,   # 'li_slnd5',
        +35.0,   # 'li_slnd6',
        +35.0,   # 'li_slnd7',
        +35.0,   # 'li_slnd8',
        +35.0,   # 'li_slnd9',
        +35.0,   # 'li_slnd10',
        +35.0,   # 'li_slnd11',
        +35.0,   # 'li_slnd12',
        +35.0,   # 'li_slnd13',
        +35.0,   # 'li_slnd14',
        +35.0,   # 'li_slnd15',
        +35.0,   # 'li_slnd16',
        +35.0,   # 'li_slnd17',
        +35.0,   # 'li_slnd18',
        +35.0,   # 'li_slnd19',
        +35.0,   # 'li_slnd20',
        +35.0,   # 'li_slnd21',

        +5.0,    # 'li_qf1',
        +5.0,    # 'li_qf2',
        +5.0,    # 'li_qf3',
        +5.0,    # 'li_qd1',
        +5.0,    # 'li_qd2',

        9.5,     # 'tb_qf2a',
        6.0,     # 'tb_qf2b',
        8.5,     # 'tb_qd2a',
        5.5,     # 'tb_qd2b',

        +2.0,    # 'posx',
        +1.0,    # 'angx',
        +2.0,    # 'posy',
        +1.0,    # 'angy',
        -19.0,   # 'kckr',

        40,      # 'shb_amp',
        91,      # 'kly1_amp',
        76,      # 'kly2_amp',
        180,     # 'shb_phs',
        -150,     # 'kly1_phs',
        0,     # 'kly2_phs',

        80,      # 'borf_amp',
        160,     # 'borf_phs',
    ]
    LIMS_LOWER = [
        -5.0,  # 'li_lens1',
        -5.0,  # 'li_lens2',
        -5.0,  # 'li_lens3',
        -5.0,  # 'li_lens4',

        0.0,   # 'li_slnd1',
        0.0,   # 'li_slnd2',
        0.0,   # 'li_slnd3',
        0.0,   # 'li_slnd4',
        0.0,   # 'li_slnd5',
        0.0,   # 'li_slnd6',
        0.0,   # 'li_slnd7',
        0.0,   # 'li_slnd8',
        0.0,   # 'li_slnd9',
        0.0,   # 'li_slnd10',
        0.0,   # 'li_slnd11',
        0.0,   # 'li_slnd12',
        0.0,   # 'li_slnd13',
        0.0,   # 'li_slnd14',
        0.0,   # 'li_slnd15',
        0.0,   # 'li_slnd16',
        0.0,   # 'li_slnd17',
        0.0,   # 'li_slnd18',
        0.0,   # 'li_slnd19',
        0.0,   # 'li_slnd20',
        0.0,   # 'li_slnd21',

        -5.0,  # 'li_qf1',
        -5.0,  # 'li_qf2',
        -5.0,  # 'li_qf3',
        -5.0,  # 'li_qd1',
        -5.0,  # 'li_qd2',

        5.0,   # 'tb_qf2a',
        2.0,   # 'tb_qf2b',
        4.0,   # 'tb_qd2a',
        1.5,   # 'tb_qd2b',

        -2.0,   # 'posx',
        -1.0,   # 'angx',
        -2.0,   # 'posy',
        -1.0,   # 'angy',
        -25.0,  # 'kckr',

        20,    # 'shb_amp',
        85,    # 'kly1_amp',
        70,    # 'kly2_amp',
        160,  # 'shb_phs',
        -180,  # 'kly1_phs',
        -20,  # 'kly2_phs',

        30,    # 'borf_amp',
        90,    # 'borf_phs',
    ]

    def __init__(self):
        """."""
        super().__init__()
        self._knobs = self.KNOBS
        self.curr_wfm_index = 100
        self.limit_lower = self.LIMS_LOWER
        self.limit_upper = self.LIMS_UPPER
        self.initial_position = list(map(
            lambda x: sum(x)/2, zip(self.LIMS_UPPER, self.LIMS_LOWER)
        ))
        self.initial_search_directions = _np.eye(
            len(self.limit_upper), dtype=float
        )
        self.nrpulses = 5
        self.use_median = False
        self.wait_between_injections = 3  # [s]

    def __str__(self):
        """."""
        stg = '-----  RCDS Parameters  -----\n\n'
        stg += super().__str__()
        stg += '\n\n-----  OptimizeInjBO Parameters  -----\n\n'
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
        kns = []
        limu = []
        liml = []
        for i, kn in enumerate(self.KNOBS):
            if kn not in knobs:
                continue
            kns.append(kn)
            limu.append(self.LIMS_UPPER[i])
            liml.append(self.LIMS_LOWER[i])
        self._knobs = kns
        self.limit_lower = _np.array(liml)
        self.limit_upper = _np.array(limu)
        self.initial_search_directions = _np.eye(len(liml), dtype=float)


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

    def prepare_evg(self):
        """Prepare EVG for optimization."""
        evg = self.devices['evg']
        # configure to inject on first bucket just once
        evg.bucketlist = [1]
        evg.nrpulses = 1
        evg.cmd_update_events()
        _time.sleep(1)

    def objective_function(self, pos=None, apply=False):
        """."""
        pos0 = self.get_current_position()

        if pos is not None:
            self.set_position_to_machine(pos)
            self.data['positions'].append(pos)
            _time.sleep(2)
        else:
            self.data['positions'].append(pos0)

        injcurrs = list()
        for i in range(self.params.nrpulses):
            injcurrs.append(self.inject_beam_and_get_current())
            _time.sleep(self.params.wait_between_injections)
        injcurrs = _np.array(injcurrs)
        self.data['currents'].append(injcurrs)

        if pos is not None and not apply:
            self.set_position_to_machine(pos0)

        func = _np.median if self.params.use_median else _np.mean
        return -func(injcurrs)

    def inject_beam_and_get_current(self, get_injeff=True):
        """Inject beam and get injected current, if desired."""
        idx = self.params.curr_wfm_index
        inj0 = self.devices['dcct'].current_fast[idx]
        self.devices['evg'].cmd_turn_on_injection(wait_rb=True)
        self.devices['evg'].wait_injection_finish()
        if not get_injeff:
            return

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
        if pos is None:
            pos = self.params.initial_position
        obj = []
        for i in range(nr_evals):
            obj.append(self.objective_function(pos))
            _log.info(f'{i+1:02d}/{nr_evals:02d}  --> obj. = {obj[-1]:.3f}')
        noise_level = _np.std(obj)
        self.params.noise_level = noise_level
        self.data['measured_objfuncs_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level
        return noise_level, obj

    def get_current_position(self):
        """Return the values of the knobs.

        Returns:
            numpy.ndarray (N, 1): vector of knobs values.

        """
        pos = []
        for knob in self.params.knobs:
            fun = knob.lower().startswith
            if fun('li_lens1'):
                pos.append(self.pvs["li_lens1"].value)
            elif fun('li_lens2'):
                pos.append(self.pvs["li_lens2"].value)
            elif fun('li_lens3'):
                pos.append(self.pvs["li_lens3"].value)
            elif fun('li_lens4'):
                pos.append(self.pvs["li_lens4"].value)

            elif fun('li_slnd1'):
                pos.append(self.pvs["li_slnd1"].value)
            elif fun('li_slnd2'):
                pos.append(self.pvs["li_slnd2"].value)
            elif fun('li_slnd3'):
                pos.append(self.pvs["li_slnd3"].value)
            elif fun('li_slnd4'):
                pos.append(self.pvs["li_slnd4"].value)
            elif fun('li_slnd5'):
                pos.append(self.pvs["li_slnd5"].value)
            elif fun('li_slnd6'):
                pos.append(self.pvs["li_slnd6"].value)
            elif fun('li_slnd7'):
                pos.append(self.pvs["li_slnd7"].value)
            elif fun('li_slnd8'):
                pos.append(self.pvs["li_slnd8"].value)
            elif fun('li_slnd9'):
                pos.append(self.pvs["li_slnd9"].value)
            elif fun('li_slnd10'):
                pos.append(self.pvs["li_slnd10"].value)
            elif fun('li_slnd11'):
                pos.append(self.pvs["li_slnd11"].value)
            elif fun('li_slnd12'):
                pos.append(self.pvs["li_slnd12"].value)
            elif fun('li_slnd13'):
                pos.append(self.pvs["li_slnd13"].value)
            elif fun('li_slnd14'):
                pos.append(self.pvs["li_slnd14"].value)
            elif fun('li_slnd15'):
                pos.append(self.pvs["li_slnd15"].value)
            elif fun('li_slnd16'):
                pos.append(self.pvs["li_slnd16"].value)
            elif fun('li_slnd17'):
                pos.append(self.pvs["li_slnd7"].value)
            elif fun('li_slnd18'):
                pos.append(self.pvs["li_slnd18"].value)
            elif fun('li_slnd19'):
                pos.append(self.pvs["li_slnd19"].value)
            elif fun('li_slnd20'):
                pos.append(self.pvs["li_slnd20"].value)
            elif fun('li_slnd21'):
                pos.append(self.pvs["li_slnd21"].value)

            elif fun('li_qf1'):
                pos.append(self.devices['li_qf1'].current)
            elif fun('li_qf2'):
                pos.append(self.devices['li_qf2'].current)
            elif fun('li_qf3'):
                pos.append(self.devices['li_qf3'].current)
            elif fun('li_qd1'):
                pos.append(self.devices['li_qd1'].current)
            elif fun('li_qd2'):
                pos.append(self.devices['li_qd2'].current)

            elif fun('tb_qf2a'):
                pos.append(self.devices['tb_qf2a'].current)
            elif fun('tb_qf2b'):
                pos.append(self.devices['tb_qf2b'].current)
            elif fun('tb_qd2a'):
                pos.append(self.devices['tb_qd2a'].current)
            elif fun('tb_qd2b'):
                pos.append(self.devices['tb_qd2b'].current)

            elif fun('posx'):
                pos.append(self.devices['pos_ang'].delta_posx)
            elif fun('angx'):
                pos.append(self.devices['pos_ang'].delta_angx)
            elif fun('posy'):
                pos.append(self.devices['pos_ang'].delta_posy)
            elif fun('angy'):
                pos.append(self.devices['pos_ang'].delta_angy)
            elif fun('kckr'):
                pos.append(self.devices['injkckr'].strength)

            elif fun('shb_amp'):
                pos.append(self.devices['li_llrf'].dev_shb.amplitude)
            elif fun('kly1_amp'):
                pos.append(self.devices['li_llrf'].dev_klystron1.amplitude)
            elif fun('kly2_amp'):
                pos.append(self.devices['li_llrf'].dev_klystron2.amplitude)
            elif fun('shb_phs'):
                pos.append(self.devices['li_llrf'].dev_shb.phase)
            elif fun('kly1_phs'):
                pos.append(self.devices['li_llrf'].dev_klystron1.phase)
            elif fun('kly2_phs'):
                pos.append(self.devices['li_llrf'].dev_klystron2.phase)

            elif fun('borf_amp'):
                pos.append(self.devices['bo_llrf'].voltage_bottom)
            elif fun('borf_phs'):
                pos.append(self.devices['bo_llrf'].phase_bottom)
            else:
                raise ValueError('Wrong specification of knob.')
        return _np.array(pos)

    def set_position_to_machine(self, pos):
        """."""
        if len(pos) != len(self.params.knobs):
            raise ValueError(
                'Length of pos must match number of knobs selected.')

        for p, knob in zip(pos, self.params.knobs):
            fun = knob.lower().startswith
            if fun('li_lens1'):
                self.pvs["li_lens1"].value = p
            elif fun('li_lens2'):
                self.pvs["li_lens2"].value = p
            elif fun('li_lens3'):
                self.pvs["li_lens3"].value = p
            elif fun('li_lens4'):
                self.pvs["li_lens4"].value = p

            elif fun('li_slnd1'):
                self.pvs["li_slnd1"].value = p
            elif fun('li_slnd2'):
                self.pvs["li_slnd2"].value = p
            elif fun('li_slnd3'):
                self.pvs["li_slnd3"].value = p
            elif fun('li_slnd4'):
                self.pvs["li_slnd4"].value = p
            elif fun('li_slnd5'):
                self.pvs["li_slnd5"].value = p
            elif fun('li_slnd6'):
                self.pvs["li_slnd6"].value = p
            elif fun('li_slnd7'):
                self.pvs["li_slnd7"].value = p
            elif fun('li_slnd8'):
                self.pvs["li_slnd8"].value = p
            elif fun('li_slnd9'):
                self.pvs["li_slnd9"].value = p
            elif fun('li_slnd10'):
                self.pvs["li_slnd10"].value = p
            elif fun('li_slnd11'):
                self.pvs["li_slnd11"].value = p
            elif fun('li_slnd12'):
                self.pvs["li_slnd12"].value = p
            elif fun('li_slnd13'):
                self.pvs["li_slnd13"].value = p
            elif fun('li_slnd14'):
                self.pvs["li_slnd14"].value = p
            elif fun('li_slnd15'):
                self.pvs["li_slnd15"].value = p
            elif fun('li_slnd16'):
                self.pvs["li_slnd16"].value = p
            elif fun('li_slnd17'):
                self.pvs["li_slnd7"].value = p
            elif fun('li_slnd18'):
                self.pvs["li_slnd18"].value = p
            elif fun('li_slnd19'):
                self.pvs["li_slnd19"].value = p
            elif fun('li_slnd20'):
                self.pvs["li_slnd20"].value = p
            elif fun('li_slnd21'):
                self.pvs["li_slnd21"].value = p

            elif fun('li_qf1'):
                self.devices['li_qf1'].current = p
            elif fun('li_qf2'):
                self.devices['li_qf2'].current = p
            elif fun('li_qf3'):
                self.devices['li_qf3'].current = p
            elif fun('li_qd1'):
                self.devices['li_qd1'].current = p
            elif fun('li_qd2'):
                self.devices['li_qd2'].current = p

            elif fun('tb_qf2a'):
                self.devices['tb_qf2a'].current = p
            elif fun('tb_qf2b'):
                self.devices['tb_qf2b'].current = p
            elif fun('tb_qd2a'):
                self.devices['tb_qd2a'].current = p
            elif fun('tb_qd2b'):
                self.devices['tb_qd2b'].current = p

            elif fun('posx'):
                self.devices['pos_ang'].delta_posx = p
            elif fun('angx'):
                self.devices['pos_ang'].delta_angx = p
            elif fun('posy'):
                self.devices['pos_ang'].delta_posy = p
            elif fun('angy'):
                self.devices['pos_ang'].delta_angy = p
            elif fun('kckr'):
                self.devices['injkckr'].strength = p

            elif fun('shb_amp'):
                self.devices['li_llrf'].dev_shb.amplitude = p
            elif fun('kly1_amp'):
                self.devices['li_llrf'].dev_klystron1.amplitude = p
            elif fun('kly2_amp'):
                self.devices['li_llrf'].dev_klystron2.amplitude = p
            elif fun('shb_phs'):
                self.devices['li_llrf'].dev_shb.phase = p
            elif fun('kly1_phs'):
                self.devices['li_llrf'].dev_klystron1.phase = p
            elif fun('kly2_phs'):
                self.devices['li_llrf'].dev_klystron2.phase = p

            elif fun('borf_amp'):
                self.devices['bo_llrf'].voltage_bottom = p
            elif fun('borf_phs'):
                self.devices['bo_llrf'].phase_bottom = p
            else:
                raise ValueError('Wrong specification of knob.')

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
        self.devices['li_qf1'] = PowerSupply('LI-Fam:PS-QF1')
        self.devices['li_qf2'] = PowerSupply('LI-Fam:PS-QF2')
        self.devices['li_qf3'] = PowerSupply('LI-01:PS-QF3')
        self.devices['li_qd1'] = PowerSupply('LI-01:PS-QD1')
        self.devices['li_qd2'] = PowerSupply('LI-01:PS-QD2')
        # TB quads
        self.devices['tb_qf2a'] = PowerSupply('TB-02:PS-QF2A')
        self.devices['tb_qf2b'] = PowerSupply('TB-02:PS-QF2B')
        self.devices['tb_qd2a'] = PowerSupply('TB-02:PS-QD2A')
        self.devices['tb_qd2b'] = PowerSupply('TB-02:PS-QD2B')
        # TB PosAng & Injkicker
        self.devices['pos_ang'] = PosAng(PosAng.DEVICES.TB)
        self.devices['injkckr'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.BO_INJ_KCKR)
        # LI LLRF
        self.devices['li_llrf'] = LILLRF()
        # BO LLRF
        self.devices['bo_llrf'] = ASLLRF(ASLLRF.DEVICES.BO)
        # other rlevant devices
        self.devices['ejekckr'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.BO_EJE_KCKR)
        self.devices['currinfo'] = CurrInfoBO()
        self.devices['dcct'] = DCCT(DCCT.DEVICES.BO)
        self.devices['evg'] = EVG()
        self.devices['ejekckr_trig'] = Trigger("BO-48D:TI-EjeKckr")
        self.devices['egun_trigps'] = EGTriggerPS()
        self.devices['injctrl'] = InjCtrl()

    def _initialization(self):
        """."""
        if not super()._initialization():
            return False
        self.data['timestamp'] = _time.time()
        self.data['positions'] = []
        self.data['currents'] = []
        self.prepare_evg()
        return True
