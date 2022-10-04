"""."""
# import matplotlib.pyplot as _plt
import time as _time
import numpy as _np
from ..utils import ThreadedMeasBaseClass as _BaseClass
from siriuspy.devices import PosAng, CurrInfoSI, EVG, PowerSupplyPU, \
    InjSysPUModeHandler
from ..optimization.rcds import RCDS as _RCDS, RCDSParams as _RCDSParams


class OptimizePosAng(_RCDS, _BaseClass):
    """."""

    def __init__(self, isonline=True, use_thread=True):
        """."""
        _BaseClass.__init__(
            self, params=_RCDSParams(), isonline=isonline)
        if self.isonline:
            self._create_devices()
        _RCDS.__init__(self, use_thread=use_thread)


    def objective_function(self, pos, apply=True):
        """."""
        if apply:
            self.apply_changes(pos)
        injeff = self._inject()
        return -injeff

    def measure_objective_function_noise(self, nr_evals, pos=None):
        """."""
        if pos is None:
            pos = self.params.initial_position
        obj = []
        for _ in range(nr_evals):
            obj.append(self.objective_function(pos, apply=False))
        noise_level = _np.std(obj)
        self.params.noise_level = noise_level
        self.data['measured_objfuns_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level
        return noise_level, obj

    def initialization(self):
        """."""
        posang = self.devices['posang']
        pos0 = self.get_positions_machine()
        posx0, angx0, posy0, angy0, kick_nlk0 = pos0
        posang.cmd_update_reference()

        self._prepare_evg()

        self.devices['injsys'].cmd_switch_to_optim()

        self.data['timestamp'] = _time.time()
        self.data['posx0'] = posx0
        self.data['angx0'] = angx0
        self.data['posy0'] = posy0
        self.data['angy0'] = angy0
        self.data['kick_nlk0'] = kick_nlk0

    def get_positions_machine(self):
        """."""
        posang = self.devices['posang']
        posx = posang.delta_posx
        angx = posang.delta_angx
        posy = posang.delta_posy
        angy = posang.delta_angy
        kick_nlk = self.devices['nlk'].strength
        pos = _np.array([posx, angx, posy, angy, kick_nlk])
        return pos

    def apply_changes(self, pos):
        """."""
        print('applying variations to machine')
        posang, nlk = self.devices['posang'], self.devices['nlk']
        posx, angx, posy, angy, kick_nlk = pos
        print('     setting new posang')
        posang.delta_posx = posx
        posang.delta_angx = angx
        posang.delta_posy = posy
        posang.delta_angy = angy
        print('     setting new nlk strength')
        nlk.strength = kick_nlk

    def save_optimization_data(self, fname, apply_machine=False):
        """."""
        self.data['best_positions'] = self.best_positions
        self.data['best_objfuncs'] = self.best_objfuncs
        self.data['final_search_directions'] = self.final_search_directions

        if apply_machine:
            self.apply_changes(self.best_positions[-1])
        self.save_data(fname)

    def _create_devices(self):
        self.devices['posang'] = PosAng(PosAng.DEVICES.TS)
        self.devices['nlk'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_NLKCKR)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['evg'] = EVG()
        self.devices['injsys'] = InjSysPUModeHandler()

    def _inject(self):
        evg, currinfo = self.devices['evg'], self.devices['currinfo']
        print('     injecting')
        evg.cmd_turn_on_injection()
        evg.wait_injection_finish()
        _time.sleep(3)
        injeff = currinfo.injeff
        print(f'    injection efficiency: {injeff:.2f} %')
        return injeff

    def _prepare_evg(self):
        evg = self.devices['evg']
        # configure to inject on first bucket just once
        evg.bucketlist = [1]
        evg.nrpulses = 1
        evg.cmd_update_events()
        _time.sleep(1)
