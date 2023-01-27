"""."""
import time as _time

from epics.ca import CAThread as _Thread
import numpy as np
import GPy as gpy

from mathphys.functions import get_namedtuple
from siriuspy.devices import HLTiming, EGun, InjCtrl, CurrInfoAS

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass

_DTMP = '{0:26s} = {1:9d}  {2:s}\n'.format
_STMP = '{0:26s} = {1:9s}  {2:s}\n'.format
_FTMP = '{0:26s} = {1:9.2f}  {2:s}\n'.format
np_poly = np.polynomial.polynomial


class BiasFeedbackParams(_ParamsBaseClass):
    """."""

    ModelTypes = get_namedtuple('ModelTypes', ['Linear', 'GaussianProcess'])

    def __init__(self):
        """."""
        super().__init__()
        self.min_bias_voltage = -52  # [V]
        self.max_bias_voltage = -40  # [V]
        self.ahead_set_time = 10  # [s]
        self.initial_angcoeff = 10  # [V/mA]
        self.initial_offcoeff = -52  # [V]
        self.model_type = self.ModelTypes.GaussianProcess
        self.model_max_num_points = 20
        self.model_auto_fit_rate = 10
        self.model_update_data = True
        self.gpmod_2d = False
        self.gpmod_sparse = False

    def __str__(self):
        """."""
        stg = ''
        stg += _FTMP('min_bias_voltage', self.min_bias_voltage, '[V]')
        stg += _FTMP('max_bias_voltage', self.max_bias_voltage, '[V]')
        stg += _FTMP('ahead_set_time', self.ahead_set_time, '[s]')
        stg += _FTMP('initial_offcoeff', self.initial_offcoeff, '[V]')
        stg += _FTMP('initial_angcoeff', self.initial_angcoeff, '[V/mA]')
        stg += _STMP('model_type', self.model_type_str, '')
        stg += _DTMP('model_max_num_points', self.model_max_num_points, '')
        stg += _DTMP(
            'model_auto_fit_rate', self.model_auto_fit_rate,
            ' (0 for no optimization)')
        stg += _STMP('model_update_data', str(self.model_update_data), '')
        stg += _STMP('gpmod_2d', str(self.gpmod_2d), '')
        stg += _STMP('gpmod_sparse', str(self.gpmod_sparse), '')
        return stg

    @property
    def model_type_str(self):
        """."""
        return self.ModelTypes._fields[self.model_type]

    @property
    def use_gaussproc_model(self):
        """."""
        return self.model_type == self.ModelTypes.GaussianProcess


class BiasFeedback(_BaseClass):
    """."""

    _DEF_TIMEOUT = 60 * 60  # [s]
    _MINIMUM_LIFETIME = 1800  # [s]

    def __str__(self):
        """."""
        stg = 'Parameters:\n\n'
        stg += str(self.params)
        stg += '\n\nGaussian Process Model:\n'
        stg += str(self.gpmodel)
        stg += '\n\nLinear Model:\n'
        off, ang = self.linmodel_coeffs
        stg += f'      Bias[V] = {ang:.1f} x dI[mA]   +   {off:.1f}\n'
        return stg

    def __init__(self, isonline=True, params=None):
        """."""
        params = params or BiasFeedbackParams()
        super().__init__(params=params, target=self._run, isonline=isonline)
        self._already_set = False

        self.linmodel_angcoeff = self.params.initial_angcoeff
        self._npts_after_fit = 0

        self.gpmodel = self.initialize_models()
        if isonline:
            self._create_devices()

    @property
    def linmodel_coeffs(self):
        """."""
        ang = self.linmodel_angcoeff
        off = self.params.initial_offcoeff
        return (off, ang)

    @property
    def linmodel_coeffs_inverse(self):
        """."""
        ang = self.linmodel_angcoeff
        off = self.params.initial_offcoeff
        return (-off/ang, 1/ang)

    def initialize_models(self):
        """."""
        self.data['bias'] = np.linspace(
            self.params.min_bias_voltage,
            self.params.max_bias_voltage,
            self.params.model_max_num_points)

        self.data['injcurr'] = np_poly.polyval(
            self.data['bias'], self.linmodel_coeffs_inverse)

        x = self.data['bias'][:, None].copy()
        y = self.data['injcurr'][:, None].copy()

        if self.params.gpmod_2d:
            kernel = gpy.kern.RBF(input_dim=2, ARD=True)
            tim = -np.ones(x.size, dtype=float)
            x = np.vstack([x, tim]).T
        else:
            kernel = gpy.kern.RBF(input_dim=1)
            x = x[:, None]

        if self.params.gpmod_sparse:
            step = x.size//6
            inducing = x[::step][:, None]
            gpmodel = gpy.models.SparseGPRegression(x, y, kernel, Z=inducing)
        else:
            gpmodel = gpy.models.GPRegression(x, y, kernel)
        return gpmodel

    def get_delta_current_per_pulse(self, **kwargs):
        """."""
        currinfo = self.devices.get('currinfo')
        injctrl = self.devices.get('injctrl')

        per = kwargs.get('topup_period') or injctrl.topup_period
        nrpul = kwargs.get('topup_nrpulses') or injctrl.topup_nrpulses
        curr_avg = kwargs.get('target_current') or injctrl.target_current
        curr_now = kwargs.get('current_mon') or currinfo.si.current
        ltime = kwargs.get('lifetime') or currinfo.si.lifetime

        ltime = max(self._MINIMUM_LIFETIME, ltime)
        curr_tar = curr_avg / (1 - per*60/2/ltime)
        dcurr = (curr_tar - curr_now) / nrpul
        return dcurr

    def get_bias_voltage(self, dcurr):
        """."""
        dcurr = max(0, dcurr)
        if self.params.use_gaussproc_model:
            return self._get_bias_voltage_gpmodel(dcurr)
        return self._get_bias_voltage_linear_model(dcurr)

    # ############ Auxiliary Methods ############
    def _run(self):
        """."""
        print('Starting measurement:')
        if self._stopevt.is_set():
            return
        egun = self.devices['egun']
        injctrl = self.devices['injctrl']
        currinfo = self.devices['currinfo']

        pvo = egun.bias.pv_object('voltoutsoft')
        pvo.auto_monitor = True
        pvo = egun.bias.pv_object('voltinsoft')
        pvo.auto_monitor = True
        pvo = currinfo.bo.pv_object('Current3GeV-Mon')
        pvo.auto_monitor = True
        pvo = currinfo.si.pv_object('InjCurr-Mon')
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

    def _create_devices(self):
        self.devices['egun'] = EGun()
        self.devices['injctrl'] = InjCtrl()
        self.devices['timing'] = HLTiming()
        self.devices['currinfo'] = CurrInfoAS()

    def _callback_to_thread(self, **kwgs):
        _Thread(target=self._update_data, kwargs=kwgs, daemon=True).start()

    def _update_data(self, **kwgs):
        bias = kwgs.get('bias')
        if bias is None:
            bias = self.devices['egun'].bias.voltage
        injcurr = kwgs.get('injcurr')
        if injcurr is None:
            injcurr = self.devices['currinfo'].si.injcurr

        # Do not overload data with repeated points:
        xun, cnts = np.unique(self.data['bias'], return_counts=True)
        if bias in xun:
            idx = (xun == bias).nonzero()[0][0]
            if cnts[idx] >= max(2, self.data['bias'].size // 5):
                print(f'Rejected point! bias={bias:.2f}, counts={cnts[idx]:d}')
                return
        self._npts_after_fit += 1

        self.data['injcurr'] = np.r_[self.data['injcurr'], injcurr]
        self.data['bias'] = np.r_[self.data['bias'], bias]
        self._update_models()

    def _update_models(self):
        x = np.r_[self.data['bias'], self.params.initial_offcoeff]
        y = np.r_[self.data['dcurr'], 0]
        x = x[-self.params.model_max_num_points:]
        y = y[-self.params.model_max_num_points:]

        fit_rate = self.params.model_auto_fit_rate
        do_opt = fit_rate and not (self._npts_after_fit % fit_rate)

        # Optimize Linear Model
        if do_opt and not self.params.use_gaussproc_model:
            self.linmodel_angcoeff = np_poly.polyfit(
                y, x-self.params.initial_offcoeff, deg=[1,])[1]
            self._npts_after_fit = 0

        # update Gaussian Process Model data
        if self.params.gpmod_2d:
            tim = self.gpmodel.X[:, 1]
            tim = np.r_[tim[:-1], tim[-1]+1, tim[-1]+1]
            tim = tim[-self.params.model_max_num_points:]
            x = np.vstack([x, tim]).T
        else:
            x.shape = (x.size, 1)
        y.shape = (y.size, 1)
        self.gpmodel.set_XY(x, y)

        # Optimize Gaussian Process Model
        if do_opt and self.params.use_gaussproc_model:
            self.gpmodel.optimize_restarts(num_restarts=2, verbose=False)
            self._npts_after_fit = 0

        self._already_set = False

    def _get_bias_voltage_gpmodel(self, injcurr):
        bias = self._gpmodel_infer_newx(np.array(injcurr, ndmin=1))
        bias = np.minimum(bias, self.params.max_bias_voltage)
        bias = np.maximum(bias, self.params.min_bias_voltage)
        return bias if bias.size > 1 else bias[0]

    def _get_bias_voltage_linear_model(self, injcurr):
        bias = np_poly.polyval(injcurr, self.linmodel_coeffs)
        bias = np.minimum(bias, self.params.max_bias_voltage)
        bias = np.maximum(bias, self.params.min_bias_voltage)
        bias = np.array([bias]).ravel()
        return bias if bias.size > 1 else bias[0]

    def _gpmodel_infer_newx(self, y):
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
            self.params.max_bias_voltage, 300)
        ys, _ = self._gpmodel_predict(x)
        idm = ys[:, 0].argmax()
        idx = np.argmin(np.abs(ys[:idm] - y[None, :]), axis=0)
        return x[idx, 0]

    def _gpmodel_predict(self, x):
        """Get the GP model prediction of the injected current.

        Args:
            x (numpy.ndarray): bias voltage.

        Returns:
            numpy.ndarray: predicted injected current.

        """
        x.shape = (x.size, 1)
        if self.params.gpmod_2d:
            tim = np.ones(x.size) * (self.gpmodel.X[-1, 1]+1)
            x = np.vstack([x.ravel(), tim]).T
        return self.gpmodel.predict(x)
