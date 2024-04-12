"""."""
import numpy as _np
from scipy.optimize import curve_fit as _curve_fit
import matplotlib.pyplot as _mplt
import datetime as _datetime
from siriuspy.devices import PowerSupplyPU

from .meas_bpms_signals import AcqBPMsSignals as _AcqBPMsSignals, \
    AcqBPMsSignalsParams as _AcqBPMsSignalsParams

class MeasureTbTData(_AcqBPMsSignals):
    """."""
    def __init__(self, filename='', isonline=False):
        super.__init__(params=TbTDataParams(), isonline=isonline,
                        ispost_mortem=False)
        self._fname = filename


    def create_devices(self):
        super().create_devices()
        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['pinghv'] = PowerSupplyPU(PowerSupplyPU.DEVICES.SI_PING_V)


    @property
    def fname(self):
        """."""
        return self._fname

    @fname.setter
    def fname(self, val):
        self._fname = val

    def prepare_pingers(self):
        """."""
        pingh, pingv = self.devices['pingh'], self.devices['pingv']
        params = self.params
        hkick, vkick = params['hkick'], params['vkick']
        pingh.strength = hkick / 1e3   # [urad]
        pingv.strength = vkick / 1e3   # [urad]
        # MISSING: set to listen to the same event as BPMs

    def do_measurement(self):
        currinfo = self.devices['currinfo']
        self.prepare_pingers()
        current_before = currinfo.current()
        self.acquire_data()
        self.data['current_before'] = current_before
        self.data['current_after'] = self.data.pop('stored_current')
        self.data['trajx'] = self.data.pop('orbx')
        self.data['trajy'] = self.data.pop('orby')
