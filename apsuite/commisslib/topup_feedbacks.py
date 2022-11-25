"""."""
import time as _time

import numpy as np

from siriuspy.devices import HLTiming, EGun, InjCtrl, CurrInfoAS

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


def _create_devices(obj):
    obj.devices['egun'] = EGun()
    obj.devices['injctrl'] = InjCtrl()
    obj.devices['timing'] = HLTiming()
    obj.devices['currinfo'] = CurrInfoAS()


class MeasBiasVsInjCurrParams(_ParamsBaseClass):
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
            params=MeasBiasVsInjCurrParams(), target=self._run,
            isonline=isonline)
        if isonline:
            _create_devices(self)

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

        timing.evg.cmd_turn_off_injection(timeout=30)
        egun.bias.voltage = ini_bias
        print('Measurement finished!')


class BiasFeedbackParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.min_lifetime = 10 * 3600  # [s]
        self.max_lifetime = 25 * 3600  # [s]
        self.default_lifetime = 17 * 3600  # [s]
        self.min_target_current = 97  # [mA]
        self.max_target_current = 105  # [mA]
        self.default_target_current = 101  # [mA]
        self.coeffs_dcurr_vs_bias = [1, 0, 0]
        self.min_bias_voltage = -49  # [V]
        self.max_bias_voltage = -43  # [V]
        self.ahead_set_time = 10  # [s]
        self.use_gaussian_process = False
        self.gaussian_process =

    def __str__(self):
        """."""
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        ftmp = '{0:26s} = {1:9.2f}  {2:s}\n'.format
        stg = ftmp('min_bias', self.min_bias, '[V]')
        stg += ftmp('max_bias', self.max_bias, '[V]')
        stg += dtmp('num_points', self.num_points, '')
        stg += ftmp('wait_each_point', self.wait_each_point, '[s]')
        return stg


class BiasFeedback(_BaseClass):
    """."""

    _DEF_TIMEOUT = 60 * 60  # [s]

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=MeasBiasVsInjCurr(), target=self._run, isonline=isonline)
        self._already_set = False
        if isonline:
            _create_devices(self)

    def _run(self):
        """."""
        print('Starting measurement:')
        if self._stopevt.is_set():
            return
        egun = self.devices['egun']
        injctrl = self.devices['injctrl']
        currinfo = self.devices['currinfo']

        pvo = currinfo.si.pv_object('InjCurr-Mon')
        cbv = pvo.add_callback(self._update_model)

        self._already_set = False
        while not self._stopevt.is_set():
            if injctrl.topup_state == injctrl.TopUpSts.Off:
                print('Topup is Off. Exiting...')
                break
            _time.sleep(2)
            next_inj = injctrl.topup_nextinj_timestamp
            dtim = next_inj - _time.time()
            if self._already_set or dtim > self.params.ahead_set_time:
                continue
            dcurr = self.get_delta_current_per_pulse()
            bias = self.get_bias_voltage(dcurr)
            egun.bias.voltage = bias
            self._already_set = True
        pvo.remove_callback(cbv)

    def _update_model(self, value, **kwgs):
        _ = kwgs
        self._already_set = False

    def get_bias_voltage(self, dcurr):
        bias = np.polynomial.polynomial.polyval(
            self.params.coeffs_dcurr_vs_bias, dcurr)
        if self.params.use_gaussian_process:
            bias += self.gaussian_process.get(dcurr)
        bias = min(bias, self.params.max_bias_voltage)
        bias = max(bias, self.params.min_bias_voltage)
        return bias


    def get_delta_current_per_pulse(self):
        currinfo = self.devices['currinfo']
        injctrl = self.devices['injctrl']

        per = injctrl.topup_period
        nrpul = injctrl.topup_nrpulses
        curr_avg = injctrl.target_current
        curr_now = currinfo.current_mon
        ltime = currinfo.si.lifetime

        if ltime/3600 < self.params.min_lifetime or \
                ltime/3600 > self.params.max_lifetime:
            ltime = self.param.default_lifetime

        curr_tar = curr_avg / (1 - per/2/ltime)
        if curr_tar < self.params.min_target_current or \
                curr_tar > self.params.max_target_current:
            curr_tar = self.params.default_target_current
        return (curr_tar - curr_now) / nrpul
