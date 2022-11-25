"""."""
import time as _time

import numpy as np

from siriuspy.devices import HLTiming, EGun, InjCtrl, CurrInfoAS

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class TopupParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.min_bias = -49  # [V]
        self.max_bias = -43  # [V]
        self.num_points = 30
        self.wait_each_point = 10  # [s]

    def __str__(self):
        """."""
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        ftmp = '{0:26s} = {1:9.2f}  {2:s}\n'.format
        stg = ftmp('min_bias', self.min_bias, '[V]')
        stg += ftmp('max_bias', self.max_bias, '[V]')
        stg += dtmp('num_points', self.num_points, '')
        stg += ftmp('wait_each_point', self.wait_each_point, '[s]')
        return stg


class MeasBiasVsInjCurr(_BaseClass):
    """."""

    _DEF_TIMEOUT = 60 * 60  # [s]

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=TopupParams(), target=self._run, isonline=isonline)
        if isonline:
            self.devices['egun'] = EGun()
            self.devices['injctrl'] = InjCtrl()
            self.devices['timing'] = HLTiming()
            self.devices['currinfo'] = CurrInfoAS()

    def _get_info(self, _, value, **kwargs):
        _ = kwargs
        egun = self.devices['egun']
        currinfo = self.devices['currinfo']
        self.data['injcurr'].append(value)
        self.data['bo3gevcurr'].append(currinfo.bo.current3gev)
        self.data['bias_mon'].append(egun.bias.voltage_mon)
        self.data['bias_rb'].append(egun.bias.voltage)

    def _run(self):
        """."""
        print('Starting measurement:')
        if self._stopevt.is_set():
            return
        egun = self.devices['egun']
        currinfo = self.devices['currinfo']
        timing = self.devices['timing']

        self.data['injcurr'] = []
        self.data['bo3gevcurr'] = []
        self.data['bias_mon'] = []
        self.data['bias_rb'] = []

        ini_bias = egun.bias.voltage
        biases = np.linspace(
            self.params.min_bias, self.params.max_bias, self.params.num_points)

        timing.evg.cmd_turn_on_injection(timeout=30)
        pvo = currinfo.si.pv_object('InjCurr-Mon')
        cbv = pvo.add_callback(self._get_info)
        for i, bias in enumerate(biases):
            print(f'  {i:03d}/{self.params.num_points:03d} bias = {bias:.2f}V')
            if self._stopevt.is_set():
                print('Stopping...')
                break
            egun.bias.set_voltage(bias)
            _time.sleep(self.params.wait_each_point)
        pvo.remove_callback(cbv)

        timing.evg.cmd_turn_on_injection(timeout=30)
        egun.bias.voltage = ini_bias
        print('Measurement finished!')
