"""."""
import time as _time

import numpy as np
import GPy as gpy

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
        self.min_bias = -50  # [V]
        self.max_bias = -37  # [V]
        self.num_points = 14
        self.wait_each_point = 3  # [s]

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

    def _get_info(self, value, **kwargs):
        _ = kwargs
        print('here')
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

        egun.bias.set_voltage(biases[0])
        _time.sleep(2)
        timing.evg.cmd_turn_on_injection(timeout=30)

        pvo = currinfo.bo.pv_object('Current3GeV-Mon')
        pvo.auto_monitor = True
        pvo = egun.bias.pv_object('voltoutsoft')
        pvo.auto_monitor = True
        pvo = egun.bias.pv_object('voltinsoft')
        pvo.auto_monitor = True
        pvo = currinfo.si.pv_object('InjCurr-Mon')
        pvo.auto_monitor = True
        cbv = pvo.add_callback(self._get_info)

        for i, bias in enumerate(biases):
            print(f'  {i:03d}/{self.params.num_points:03d} bias = {bias:.2f}V')
            if self._stopevt.is_set():
                print('Stopping...')
                break
            print(egun.bias.voltage, egun.bias.voltage_mon)
            egun.bias.set_voltage(bias)
            _time.sleep(self.params.wait_each_point)

        pvo.auto_monitor = False
        pvo.remove_callback(cbv)

        timing.evg.cmd_turn_off_injection(timeout=30)
        egun.bias.set_voltage(ini_bias)
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
        self.min_delta_current = 0.001  # [mA]
        self.max_delta_current = 1  # [mA]
        self.min_bias_voltage = -49  # [V]
        self.max_bias_voltage = -43  # [V]
        self.ahead_set_time = 10  # [s]
        self.use_gpmodel = False
        self.gpmodel_lengthscale = 0.1  # [mA]
        self.gpmodel_variance = 9  # [V^2]
        self.gpmodel_noise_var = 0.3  # [V^2]

    def __str__(self):
        """."""
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:26s} = {1:9s}  {2:s}\n'.format
        ftmp = '{0:26s} = {1:9.2f}  {2:s}\n'.format

        stg = ''
        stg += ftmp('min_lifetime', self.min_lifetime, '[s]')
        stg += ftmp('max_lifetime', self.max_lifetime, '[s]')
        stg += ftmp('default_lifetime', self.default_lifetime, '[s]')
        stg += ftmp('min_target_current', self.min_target_current, '[mA]')
        stg += ftmp('max_target_current', self.max_target_current, '[mA]')
        stg += ftmp(
            'default_target_current', self.default_target_current, '[mA]')
        stg += ftmp('min_delta_current', self.min_delta_current, '[mA]')
        stg += ftmp('max_delta_current', self.max_delta_current, '[mA]')
        stg += ftmp('min_bias_voltage', self.min_bias_voltage, '[V]')
        stg += ftmp('max_bias_voltage', self.max_bias_voltage, '[V]')
        stg += ftmp('ahead_set_time', self.ahead_set_time, '[s]')
        # stg += ftmp('coeffs_dcurr_vs_bias', self.coeffs_dcurr_vs_bias, ' 0]')
        stg += stmp('use_gpmodel', str(self.use_gpmodel), '')
        stg += ftmp('gpmodel_lengthscale', self.gpmodel_lengthscale, '[mA]')
        stg += ftmp('gpmodel_variance', self.gpmodel_variance, '[V^2]')
        stg += ftmp('gpmodel_noise_var', self.gpmodel_noise_var, '[V^2]')
        return stg


class BiasFeedback(_BaseClass):
    """."""

    _DEF_TIMEOUT = 60 * 60  # [s]

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=BiasFeedbackParams(), target=self._run, isonline=isonline)
        self.params = BiasFeedbackParams()
        self._already_set = False
        self.gpmodel_kernel = gpy.kern.RBF(
            input_dim=1, variance=self.params.gpmodel_variance,
            lengthscale=self.params.gpmodel_lengthscale)
        self.gpmodel = gpy.models.GPRegression(
            np.zeros((5, 1)), np.zeros((5, 1)), self.gpmodel_kernel,
            noise_var=self.params.gpmodel_noise_var)
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

        bias = self.devices['egun'].bias.voltage
        if value > self.params.max_delta_current or \
                value < self.params.min_delta_current or \
                bias > self.params.max_bias_voltage or \
                bias < self.params.min_bias_voltage:
            return

        self.data['dcurr'].append(value)
        self.data['bias'].append(bias)

        x = np.array(self.data['dcurr'])
        y = np.array(self.data['bias'])
        x.shape = (x.size, 1)
        y.shape = (y.size, 1)

        y -= np.polynomial.polynomial.polyval(
            x, self.params.coeffs_dcurr_vs_bias)
        self.gpmodel.set_XY(x, y)

    def get_bias_voltage(self, dcurr):
        """."""
        bias = np.polynomial.polynomial.polyval(
            dcurr, self.params.coeffs_dcurr_vs_bias)

        if self.params.use_gaussian_process:
            dcurr = np.array(dcurr, ndmin=2)
            avg, var = self.gpmodel.predict(dcurr)
            bias += avg[0, 0]

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
            ltime = self.params.default_lifetime

        curr_tar = curr_avg / (1 - per/2/ltime)
        if curr_tar < self.params.min_target_current or \
                curr_tar > self.params.max_target_current:
            curr_tar = self.params.default_target_current
        return (curr_tar - curr_now) / nrpul
