"""."""
import time as _time
from threading import Thread as _Thread

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

    def plot_and_fit_data(self):
        """."""
        pass


class BiasFeedbackParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.min_lifetime = 10 * 3600  # [s]
        self.max_lifetime = 25 * 3600  # [s]
        self.default_lifetime = 17 * 3600  # [s]
        self.min_target_current = 99  # [mA]
        self.max_target_current = 102  # [mA]
        self.default_target_current = 100.0  # [mA]
        self.coeffs_dcurr_vs_bias = [-50.0, 10.0]
        self.min_delta_current = 0.000  # [mA]
        self.max_delta_current = 1  # [mA]
        self.min_bias_voltage = -52  # [V]
        self.max_bias_voltage = -40  # [V]
        self.ahead_set_time = 10  # [s]
        self.use_gpmodel = False
        self.gpmodel_lengthscale = 4  # [V]
        self.gpmodel_variance = 0.16  # [mA^2]
        self.gpmodel_noise_var = 0.03  # [mA^2]
        self.gpmodel_max_num_points = 20
        self.gpmodel_opt_each_pts = 10
        self.gpmodel_bidimensional = False
        self.gpmodel_issparse = False

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
        stg += stmp('coeffs_dcurr_vs_bias', str(self.coeffs_dcurr_vs_bias), '')
        stg += stmp('use_gpmodel', str(self.use_gpmodel), '')
        stg += dtmp(
            'gpmodel_opt_each_pts', self.gpmodel_opt_each_pts,
            ' (0 for no optimization)')
        stg += ftmp('gpmodel_lengthscale', self.gpmodel_lengthscale, '[mA]')
        stg += ftmp('gpmodel_variance', self.gpmodel_variance, '[V^2]')
        stg += ftmp('gpmodel_noise_var', self.gpmodel_noise_var, '[V^2]')
        stg += dtmp('gpmodel_max_num_points', self.gpmodel_max_num_points, '')
        stg += stmp(
            'gpmodel_bidimensional', str(self.gpmodel_bidimensional), '')
        stg += stmp('gpmodel_issparse', str(self.gpmodel_issparse), '')
        return stg


class BiasFeedback(_BaseClass):
    """."""

    _DEF_TIMEOUT = 60 * 60  # [s]

    def __init__(self, isonline=True, params=None):
        """."""
        params = params or BiasFeedbackParams()
        super().__init__(params=params, target=self._run, isonline=isonline)
        self._already_set = False

        bias = np.linspace(
            self.params.min_bias_voltage,
            self.params.max_bias_voltage,
            self.params.gpmodel_max_num_points)

        b, a = self.params.coeffs_dcurr_vs_bias
        dcurr = np.polynomial.polynomial.polyval(bias, (-b/a, 1/a))
        y = dcurr[:, None]

        if self.params.gpmodel_bidimensional:
            self.gpmodel_kernel = gpy.kern.RBF(
                input_dim=2, variance=self.params.gpmodel_variance,
                lengthscale=self.params.gpmodel_lengthscale, ARD=True)
            tim = -np.ones(bias.size, dtype=float)
            x = np.vstack([bias, tim]).T
        else:
            self.gpmodel_kernel = gpy.kern.RBF(
                input_dim=1, variance=self.params.gpmodel_variance,
                lengthscale=self.params.gpmodel_lengthscale)
            x = bias[:, None]

        if self.params.gpmodel_issparse:
            step = bias.size//6
            inducing = bias[::step][:, None]
            self.gpmodel = gpy.models.SparseGPRegression(
                x, y, self.gpmodel_kernel, Z=inducing)
        else:
            self.gpmodel = gpy.models.GPRegression(
                x, y, self.gpmodel_kernel,
                noise_var=self.params.gpmodel_noise_var)

        self.gpmodel.likelihood.variance = self.params.gpmodel_noise_var
        self._gpmodel_pts = 0
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

        self.data['dcurr'] = []
        self.data['bias'] = []

        pvo = egun.bias.pv_object('voltoutsoft')
        pvo.auto_monitor = True
        pvo = egun.bias.pv_object('voltinsoft')
        pvo.auto_monitor = True
        pvo = currinfo.si.pv_object('InjCurr-Mon')
        pvo.auto_monitor = True
        pvo = currinfo.bo.pv_object('Current3GeV-Mon')
        pvo.auto_monitor = True
        cbv = pvo.add_callback(self._callback_to_thread)

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
            egun.bias.set_voltage(bias)
            print(f'dcurr = {dcurr:.3f}, bias = {bias:.2f}')
            self._already_set = True
        pvo.remove_callback(cbv)
        print('Finished!')

    def _callback_to_thread(self, **kwgs):
        _Thread(target=self._update_model, kwargs=kwgs, daemon=True).start()

    def _update_model(self, **kwgs):
        simul = kwgs.get('simul', False)
        if not simul:
            _time.sleep(1)

        bias = kwgs.get('bias')
        if bias is None:
            bias = self.devices['egun'].bias.voltage
        dcurr = kwgs.get('dcurr')
        if dcurr is None:
            dcurr = self.devices['currinfo'].si.injcurr

        self.data['dcurr'].append(dcurr)
        self.data['bias'].append(bias)

        x = self.gpmodel.X[:, 0]
        y = self.gpmodel.Y[:, 0]
        xun, cnts = np.unique(x, return_counts=True)
        if bias in xun:
            idx = (xun == bias).nonzero()[0][0]
            if cnts[idx] >= max(2, x.size // 5):
                print(f'Rejected point! bias={bias:.2f}, counts={cnts[idx]:d}')
                return

        x = np.r_[x[:-1], bias, self.params.min_bias_voltage]
        y = np.r_[y[:-1], dcurr, 0]
        x = x[-self.params.gpmodel_max_num_points:]
        y = y[-self.params.gpmodel_max_num_points:]

        if self.params.gpmodel_bidimensional:
            tim = self.gpmodel.X[:, 1]
            tim = np.r_[tim[:-1], tim[-1]+1, tim[-1]+1]
            tim = tim[-self.params.gpmodel_max_num_points:]
            x = np.vstack([x, tim]).T
        else:
            x.shape = (x.size, 1)
        y.shape = (y.size, 1)

        self.gpmodel.set_XY(x, y)
        opt = self.params.gpmodel_opt_each_pts
        self._gpmodel_pts += 1
        if self.params.use_gpmodel and opt and not (self._gpmodel_pts % opt):
            print('Optimizing Gaussian Process Model')
            self.gpmodel.optimize_restarts(num_restarts=5, verbose=False)
            self._gpmodel_pts = 0

        if not simul:
            _time.sleep(3)
        self._already_set = False

    def get_bias_voltage(self, dcurr):
        """."""
        bias = np.polynomial.polynomial.polyval(
            dcurr, self.params.coeffs_dcurr_vs_bias)
        bias = np.array([bias]).ravel()

        xgp = self.gpmodel.X
        opt = self.params.gpmodel_opt_each_pts
        if self.params.use_gpmodel and xgp.shape[0] >= opt:
            y = np.array(dcurr, ndmin=1)
            bias = self.gpmodel_infer_newx(y)

        bias = np.minimum(bias, self.params.max_bias_voltage)
        bias = np.maximum(bias, self.params.min_bias_voltage)
        return bias if bias.size > 1 else bias[0]

    def gpmodel_infer_newx(self, y):
        """Infer x given y for current GP model.

        The GP model object has its own infer_newX method, but it is slow and
        didn't give good results in my tests. So I decided to implement this
        simpler version, which works well.

        Args:
            y (numpy.ndarray, (N,)): y's for which we want to infer x.

        Returns:
            x: infered x's.

        """
        x = np.linspace(
            self.params.min_bias_voltage,
            self.params.max_bias_voltage, 100)

        ys, _ = self.gpmodel_predict(x)

        idx = np.argmin(np.abs(ys - y[None, :]), axis=0)
        return x[idx, 0]

    def gpmodel_predict(self, x):
        """Get the GP model prediction of the injected current.

        Args:
            x (numpy.ndarray): bias voltage.

        Returns:
            numpy.ndarray: predicted injected current.

        """
        x.shape = (x.size, 1)
        if self.params.gpmodel_bidimensional:
            tim = np.ones(x.size) * self.gpmodel.X[-1, 1]
            x = np.vstack([x.ravel(), tim]).T

        return self.gpmodel.predict(x)

    def get_delta_current_per_pulse(self, **kwargs):
        """."""
        currinfo = self.devices.get('currinfo')
        injctrl = self.devices.get('injctrl')

        per = kwargs.get('topup_period') or injctrl.topup_period
        nrpul = kwargs.get('topup_nrpulses') or injctrl.topup_nrpulses
        curr_avg = kwargs.get('target_current') or injctrl.target_current
        curr_now = kwargs.get('current_mon') or currinfo.si.current
        ltime = kwargs.get('lifetime') or currinfo.si.lifetime

        if ltime < self.params.min_lifetime or \
                ltime > self.params.max_lifetime:
            ltime = self.params.default_lifetime

        curr_tar = curr_avg / (1 - per*60/2/ltime)
        if curr_tar < self.params.min_target_current or \
                curr_tar > self.params.max_target_current:
            curr_tar = self.params.default_target_current

        dcurr = (curr_tar - curr_now) / nrpul
        dcurr = min(dcurr, self.params.max_delta_current)
        dcurr = max(dcurr, self.params.min_delta_current)
        return dcurr
