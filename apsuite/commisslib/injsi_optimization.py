"""."""
import time as _time
import numpy as _np
from epics import PV
from apsuite.optimization import SimulAnneal
from siriuspy.devices import Tune, TuneCorr, CurrInfoSI
from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class InjSIParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.nr_iter = 10
        self.nr_pulses = 5
        self.max_delta_tunex = 1e-2
        self.max_delta_tuney = 1e-2
        self.wait_tunecorr = 1  # [s]
        self.pulse_freq = 2  # [Hz]

    def __str__(self):
        """."""
        ftmp = '{0:15s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:15s} = {1:9d}  {2:s}\n'.format
        stg = dtmp('nr_iter', self.nr_iter, '')
        stg += dtmp('nr_pulses', self.nr_pulses, '')
        stg += ftmp('max_delta_tunex', self.max_delta_tunex, '')
        stg += ftmp('max_delta_tuney', self.max_delta_tuney, '')
        stg += ftmp('wait_tunecorr', self.wait_tunecorr, '[s]')
        stg += ftmp('pulse_freq', self.pulse_freq, '[Hz]')
        return stg


class TuneScanInjSI(SimulAnneal, _BaseClass):
    """."""

    PV_INJECTION = 'AS-RaMO:TI-EVG:InjectionEvt-Sel'

    def __init__(self, save=False):
        """."""
        SimulAnneal.__init__(self, save=save)
        _BaseClass.__init__(self)
        self.devices = dict()
        self.params = InjSIParams()
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['tunecorr'] = TuneCorr(TuneCorr.DEVICES.SI)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['injection'] = PV(TuneScanInjSI.PV_INJECTION)
        self.devices['tunecorr'].cmd_update_reference()
        self.data['measure'] = dict()
        self.data['measure']['tunex'] = []
        self.data['measure']['tuney'] = []
        self.data['measure']['injeff'] = []

    def _inject(self):
        self.devices['injection'].value = 1

    def _apply_variation(self):
        tunecorr = self.devices['tunecorr']
        dnux, dnuy = self.position[0], self.position[1]
        tunecorr.delta_tunex = dnux
        tunecorr.delta_tuney = dnuy
        tunecorr.cmd_apply_delta()
        _time.sleep(self.params.wait_tunecorr)

    def calc_obj_fun(self):
        """."""
        tune = self.devices['tune']
        self.data['measure']['tunex'].append(tune.tunex)
        self.data['measure']['tuney'].append(tune.tuney)
        self._apply_variation()
        injeff = []
        for _ in range(self.params.nr_pulses):
            self._inject()
            injeff.append(self.devices['currinfo'].injeff)
            _time.sleep(1/self.params.pulse_freq)
        self.data['measure']['injeff'].append(injeff)
        return - _np.mean(injeff)

    def initialization(self):
        """."""
        self.niter = self.params.nr_iter
        self.position = _np.array([0, 0])
        self.limits_upper = _np.array(
            [self.params.max_delta_tunex, self.params.max_delta_tuney])
        self.limits_lower = - self.limits_upper
        self.deltas = self.limits_upper.copy()
