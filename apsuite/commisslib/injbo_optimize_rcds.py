"""."""
import time as _time
import logging as _log

import numpy as _np

from pymodels import si as _si
from siriuspy.devices import PowerSupply, PowerSupplyPU, CurrInfoBO, EVG, \
    EGTriggerPS, LILLRF, InjCtrl, PosAng, DCCT

from ..optimization.rcds import RCDS as _RCDS, RCDSParams as _RCDSParams
from ..optics_analysis import ChromCorr


class OptimizeInjBOParams(_RCDSParams):
    """."""

    KNOBS = [
        'li_qf3', 'tb_qf2a', 'tb_qf2b', 'tb_qd2a', 'tb_qd2b', 'posx', 'angx',
        'posy', 'angy', 'kckr']
    LIMS_UPPER = [+3.0, 9.5, 6.0, 8.5, 5.5, +2.0, +1.0, +2.0, +1.0, -25.0]
    LIMS_LOWER = [-3.0, 5.0, 2.0, 4.0, 1.5, -2.0, -1.0, -2.0, -1.0, -19.0]

    def __init__(self):
        """."""
        super().__init__()
        self._knobs = self.KNOBS
        self.curr_wfm_index = 100
        self.limit_lower = self.LIMS_LOWER
        self.limit_upper = self.LIMS_UPPER

    def __str__(self):
        """."""
        stg = '-----  RCDS Parameters  -----\n\n'
        stg += super().__str__()
        stg += '\n\n-----  OptimizeInjBO Parameters  -----\n\n'
        stg += self._TMPD.format('curr_wfm_index', self.curr_wfm_index, '')
        stg += self._TMPS.format('knobs', ', '.join(self._knobs), '')
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
        self.knobs = kns
        self.limit_lower = _np.array(liml)
        self.limit_upper = _np.array(limu)


class OptimizeInjBO(_RCDS):
    """."""

    def __init__(self, isonline=True, use_thread=True):
        """."""
        _RCDS.__init__(self, isonline=isonline, use_thread=use_thread)
        self.params = OptimizeInjBOParams()
        self.data['strengths'] = []
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

    def objective_function(self, pos):
        """."""
        self.set_position_to_machine(pos)
        self.data['positions'].append(pos)
        _time.sleep(2)

        injcurr = self.inject_beam_and_get_current()
        self.data['currents'].append(injcurr)
        return -injcurr

    def inject_beam_and_get_current(self, get_injeff=True):
        """Inject beam and get injected current, if desired."""
        idx = self.params.curr_wfm_index
        # inj0 = self.devices['currinfo'].current1gev
        inj0 = self.devices['dcct'].current_fast[idx]
        self.devices['evg'].cmd_turn_on_injection(wait_rb=True)
        self.devices['evg'].wait_injection_finish()
        if not get_injeff:
            return

        for _ in range(50):
            # inj = self.devices['currinfo'].current1gev
            inj = self.devices['dcct'].current_fast[idx]
            if inj0 != inj:
                break
            _time.sleep(0.1)
        else:
            _log.warning('Timed out waiting current1gev to update.')
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
            if knob.lower.startswith('li_qf3'):
                pos.append(self.devices['li_qf3'].current)
            elif knob.lower.startswith('tb_qf2a'):
                pos.append(self.devices['tb_qf2a'].current)
            elif knob.lower.startswith('tb_qf2b'):
                pos.append(self.devices['tb_qf2b'].current)
            elif knob.lower.startswith('tb_qd2a'):
                pos.append(self.devices['tb_qd2a'].current)
            elif knob.lower.startswith('tb_qd2b'):
                pos.append(self.devices['tb_qd2b'].current)
            elif knob.lower.startswith('posx'):
                pos.append(self.devices['pos_ang'].delta_posx)
            elif knob.lower.startswith('angx'):
                pos.append(self.devices['pos_ang'].delta_angx)
            elif knob.lower.startswith('posy'):
                pos.append(self.devices['pos_ang'].delta_posy)
            elif knob.lower.startswith('angy'):
                pos.append(self.devices['pos_ang'].delta_angy)
            elif knob.lower.startswith('kckr'):
                pos.append(self.devices['injkckr'].delta_angy)
            else:
                raise ValueError('Wrong specification of knob.')
        return _np.array(pos)

    def set_position_to_machine(self, pos):
        """."""
        if len(pos) != len(self.params.knobs):
            raise ValueError(
                'Length of pos must match number of knobs selected.')

        for p, knob in zip(pos, self.params.knobs):
            if knob.lower.startswith('li_qf3'):
                self.devices['li_qf3'].current = p
            elif knob.lower.startswith('tb_qf2a'):
                self.devices['tb_qf2a'].current = p
            elif knob.lower.startswith('tb_qf2b'):
                self.devices['tb_qf2b'].current = p
            elif knob.lower.startswith('tb_qd2a'):
                self.devices['tb_qd2a'].current = p
            elif knob.lower.startswith('tb_qd2b'):
                self.devices['tb_qd2b'].current = p
            elif knob.lower.startswith('posx'):
                self.devices['pos_ang'].delta_posx = p
            elif knob.lower.startswith('angx'):
                self.devices['pos_ang'].delta_angx = p
            elif knob.lower.startswith('posy'):
                self.devices['pos_ang'].delta_posy = p
            elif knob.lower.startswith('angy'):
                self.devices['pos_ang'].delta_angy = p
            elif knob.lower.startswith('kckr'):
                self.devices['injkckr'].delta_angy = p
            else:
                raise ValueError('Wrong specification of knob.')

    def _create_devices(self):
        self.devices['li_qf3'] = PowerSupply('LI-01:PS-QF3')
        self.devices['tb_qf2a'] = PowerSupply('TB-02:PS-QF2A')
        self.devices['tb_qf2b'] = PowerSupply('TB-02:PS-QF2B')
        self.devices['tb_qd2a'] = PowerSupply('TB-02:PS-QD2A')
        self.devices['tb_qd2b'] = PowerSupply('TB-02:PS-QD2B')
        self.devices['pos_ang'] = PosAng(PosAng.DEVICES.TB)
        self.devices['injkckr'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.BO_INJ_KCKR)
        self.devices['ejekckr'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.BO_EJE_KCKR)
        self.devices['currinfo'] = CurrInfoBO()
        self.devices['dcct'] = DCCT(DCCT.DEVICES.BO)
        self.devices['evg'] = EVG()
        self.devices['egun_trigps'] = EGTriggerPS()
        self.devices['llrf'] = LILLRF()
        self.devices['injctrl'] = InjCtrl()

    def _initialization(self):
        """."""
        if not super()._initialization():
            return False
        self.data['timestamp'] = _time.time()
        self.data['postitions'] = []
        self.data['currents'] = []
        self.prepare_evg()
        return True
