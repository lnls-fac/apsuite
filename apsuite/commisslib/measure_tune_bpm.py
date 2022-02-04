"Methods for measure tunes by bpm data"
import numpy as _np
import time as _time 
import pyaccel as _pa
from numpy.fft import rfft, rfftfreq
import matplotlib.pyplot as _plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from siriuspy.sofb.csdev import SOFBFactory

from siriuspy.devices import BPM, Tune, CurrInfoBO, PowerSupply, \
    Trigger, Event, EVG, RFGen, SOFB, PowerSupplyPU, FamBPMs

from apsuite.utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class BPMeasureParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.n_bpms = 1  # Numbers of BPMs
        self.nr_points_bpm_acq = 10000
        self.bpms_timeout = 30  # [s]


class BPMeasure(_ThreadBaseClass):
    """."""
    def __init__(self, params=None, isonline=True):
        """."""
        params = BPMeasureParams() if params is None else params
        # Do I need to set a target in the below line?
        super().__init__(params=params, isonline=isonline)
        self.sofb_data = SOFBFactory.create('BO')
        if self.isonline:
            self._create_devices()

    def _create_devices(self):
        self.devices['currinfo'] = CurrInfoBO()
        self.devices['bobpms'] = FamBPMs(FamBPMs.DEVICES.BO)
        # self.devices['event'] = Event('Study')
        self.devices['event'] = Event('DigBO')
        self.devices['evg'] = EVG()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.BO)
        self.devices['trigbpm'] = Trigger('BO-Fam:TI-BPM')
        self.devices['evg'] = EVG()
        self.devices['rfgen'] = RFGen()
        self.devices['ejekckr'] = PowerSupplyPU(PowerSupplyPU.
                                                DEVICES.BO_EJE_KCKR)

    def get_orbit_data():
        """Get orbit data from BPMS in TbT acquisition rate

        BPMs must be configured to listen DigBO event and the DigBO
        event must be in External mode."""
