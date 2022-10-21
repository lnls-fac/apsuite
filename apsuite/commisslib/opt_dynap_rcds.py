"""."""
import time as _time
import numpy as _np
from ..utils import ThreadedMeasBaseClass as _BaseClass
from siriuspy.devices import PowerSupplyPU, Tune, CurrInfoSI, EVG, RFGen, \
     BunchbyBunch
from ..optimization.rcds import RCDS as _RCDS, RCDSParams as _RCDSParams


class OptimizeDA(_RCDS, _BaseClass):
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
        loss = self._calc_loss()
        return -loss

    def measure_objective_function_noise(self, nr_evals, pos=None):
        """."""
        if pos is None:
            pos = self.params.initial_position
        obj = []
        for _ in range(nr_evals):
            obj.append(self.objective_function(pos, apply=False))
        noise_level = _np.std(obj)
        self.params.noise_level = noise_level
        self.data['measured_objfuncs_for_noise'] = obj
        self.data['measured_noise_level'] = noise_level
        return noise_level, obj

    def initialization(self):
        """."""
        pingh = self.devices['pingh']
        pos0 = self.get_positions_machine()

    def get_positions_machine(self):
        """."""
        # how to get SL's?

    def apply_changes(self, pos):
        """."""
        print('Applying changes to machine')

    def _create_devices(self):
        # which sextupole families to use?
        self.devices['sextupole'] = PowerSupplyPU(PowerSupplyPU.DEVICES.)
        self.devices['pinhg'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['curr'] = CurrInfoSI()
        self.devices['rfgen'] = RFGen()
        self.devices['evg'] = EVG()
        self.devices['bbl'] = BunchbyBunch(BunchbyBunch.DEVICES.L)

    def _calc_loss(self):
        evg, currinfo = self.devices['evg'], self.devices['currinfo']
        dcct_offset = 0.04
        curr0 = currinfo.current - dcct_offset
        _time.sleep(1)
        evg.cmd_turn_on_injection()
        toca.cmd_reset() #  couldn't find what does this do
        toca.wait_buffer()
        time.sleep(3)

        currf = currinfo.current - dcct_offset
        loss = (currf-curr0)/curr0 * 100
        return loss

    def _prepare_evg(self):
        # injection scheme?
        evg = self.devices['evg']
        evg.bucketlist = [1]
        evg.nrpulses = 1
        evg.cmd_update_events()
        _time.sleep(1)
