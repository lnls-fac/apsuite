"""."""
# import matplotlib.pyplot as _plt
import time as _time
import numpy as _np
from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass
from siriuspy.devices import PosAng, CurrInfoSI, EVG, PowerSupplyPU, \
    InjSysPUModeHandler
from ..optimization.rcds import RCDS as _RCDS, RCDSParams as _RCDSParams


class OptimizePosAngParams(_ParamsBaseClass, _RCDSParams):
    """."""

    def __init__(self):
        """."""
        _ParamsBaseClass().__init__()
        _RCDSParams().__init__()

    def __str__(self):
        """."""
        stg = _RCDSParams().__str__()
        return stg


class OptimizePosAngPosAng(_RCDS, _BaseClass):
    """."""

    def __init__(self, isonline=True, use_thread=True):
        """."""
        _BaseClass.__init__(
            self, params=OptimizePosAngParams(), isonline=isonline)
        _RCDS.__init__(self, use_thread=use_thread)
        if isonline:
            self._create_devices()

    def objective_function(self, pos):
        """."""
        # apply variations to machine
        self.apply_changes(pos)
        # inject and measure injeff
        injeff = self._inject()
        return -injeff

    def measure_noise_level(self, nr_evals):
        """."""
        effs = []
        print('measuring noise level:')
        for _ in range(nr_evals):
            effs.append(self._inject())
        noise_level = _np.std(effs)
        self.params.noise_level = noise_level
        return noise_level

    def initialization(self):
        """."""
        if self.isonline:
            posang = self.devices['posang']
            posx0 = posang.delta_posx
            angx0 = posang.delta_angx
            posy0 = posang.delta_posy
            angy0 = posang.delta_angy
            kick_nlk0 = self.devices['nlk'].strength
            posang.cmd_update_reference()

            evg = self.devices['evg']
            # inject on first bucket just once
            evg.bucket_list = [1]
            evg.nrpulses = 1
            evg.cmd_update_events()
            _time.sleep(1)

            self.devices['injsys'].cmd_switch_to_optim()

            self.data['timestamp'] = _time.time()
            self.data['posx0'] = posx0
            self.data['angx0'] = angx0
            self.data['posy0'] = posy0
            self.data['angy0'] = angy0
            self.data['kick_nlk0'] = kick_nlk0

    def apply_changes(self, pos):
        """."""
        print('applying variations to machine')
        if self.isonline:
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
        _time.sleep(1)
        injeff = currinfo.injeff
        print(f'    injection efficiency: {injeff:.2} %')
        return injeff
