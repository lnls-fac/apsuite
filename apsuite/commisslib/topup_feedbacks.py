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
        self.model_type = self.ModelTypes.GaussianProcess
        self.model_max_num_points = 20
        self.model_opt_each_pts = 10
        self.gaussproc_bidimensional = False
        self.gaussproc_issparse = False

    def __str__(self):
        """."""
        stg = ''
        stg += _FTMP('min_bias_voltage', self.min_bias_voltage, '[V]')
        stg += _FTMP('max_bias_voltage', self.max_bias_voltage, '[V]')
        stg += _FTMP('ahead_set_time', self.ahead_set_time, '[s]')
        stg += _FTMP('initial_angcoeff', self.initial_angcoeff, '[V/mA]')
        stg += _STMP('model_type', self.model_type_str, '')
        stg += _DTMP('model_max_num_points', self.model_max_num_points, '')
        stg += _DTMP(
            'model_opt_each_pts', self.model_opt_each_pts,
            ' (0 for no optimization)')
        stg += _STMP(
            'gaussproc_bidimensional', str(self.gaussproc_bidimensional), '')
        stg += _STMP('gaussproc_issparse', str(self.gaussproc_issparse), '')
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
        self._num_points_after_opt = 0

        self.gpmodel = self.initialize_gpmodel()
        if isonline:
            self._create_devices()

    @property
    def linmodel_coeffs(self):
        """."""
        ang = self.linmodel_angcoeff
        off = self.params.min_bias_voltage
        return (off, ang)

    def initialize_gpmodel(self):
        """."""
        bias = np.linspace(
            self.params.min_bias_voltage,
            self.params.max_bias_voltage,
            self.params.model_max_num_points)

        off, ang = self.linmodel_coeffs
        dcurr = np_poly.polyval(bias, (-off/ang, 1/ang))
        y = dcurr[:, None]

        if self.params.gaussproc_bidimensional:
            kernel = gpy.kern.RBF(input_dim=2, ARD=True)
            tim = -np.ones(bias.size, dtype=float)
            x = np.vstack([bias, tim]).T
        else:
            kernel = gpy.kern.RBF(input_dim=1)
            x = bias[:, None]

        if self.params.gaussproc_issparse:
            step = bias.size//6
            inducing = bias[::step][:, None]
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

    def _create_devices(self):
        self.devices['egun'] = EGun()
        self.devices['injctrl'] = InjCtrl()
        self.devices['timing'] = HLTiming()
        self.devices['currinfo'] = CurrInfoAS()

    def _callback_to_thread(self, **kwgs):
        _Thread(target=self._update_models, kwargs=kwgs, daemon=True).start()

    def _update_models(self, **kwgs):
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

        x = np.array(self.data['bias'])
        y = np.array(self.data['dcurr'])
        x = x[-self.params.model_max_num_points:]
        y = y[-self.params.model_max_num_points:]

        # Do not overload data with repeated points:
        xun, cnts = np.unique(x, return_counts=True)
        if bias in xun:
            idx = (xun == bias).nonzero()[0][0]
            if cnts[idx] >= max(2, x.size // 5):
                print(f'Rejected point! bias={bias:.2f}, counts={cnts[idx]:d}')
                return
        self._num_points_after_opt += 1

        opt = self.params.model_opt_each_pts
        do_opt = opt and not (self._num_points_after_opt % opt)

        # Optimize Linear Model
        if do_opt and not self.params.use_gaussproc_model:
            self.linmodel_angcoeff = np_poly.polyfit(
                y, x-self.params.min_bias_voltage, deg=[1,])[1]
            self._num_points_after_opt = 0

        # update Gaussian Process Model data
        x = self.gpmodel.X[:, 0]
        y = self.gpmodel.Y[:, 0]
        x = np.r_[x[:-1], bias, self.params.min_bias_voltage]
        y = np.r_[y[:-1], dcurr, 0]
        x = x[-self.params.model_max_num_points:]
        y = y[-self.params.model_max_num_points:]
        if self.params.gaussproc_bidimensional:
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
            self._num_points_after_opt = 0

        if not simul:
            _time.sleep(3)
        self._already_set = False

    def _get_bias_voltage_gpmodel(self, dcurr):
        bias = self._gpmodel_infer_newx(np.array(dcurr, ndmin=1))
        bias = np.minimum(bias, self.params.max_bias_voltage)
        bias = np.maximum(bias, self.params.min_bias_voltage)
        return bias if bias.size > 1 else bias[0]

    def _get_bias_voltage_linear_model(self, dcurr):
        bias = np_poly.polyval(dcurr, self.linmodel_coeffs)
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
            self.params.max_bias_voltage, 100)
        ys, _ = self._gpmodel_predict(x)
        idx = np.argmin(np.abs(ys - y[None, :]), axis=0)
        return x[idx, 0]

    def _gpmodel_predict(self, x):
        """Get the GP model prediction of the injected current.

        Args:
            x (numpy.ndarray): bias voltage.

        Returns:
            numpy.ndarray: predicted injected current.

        """
        x.shape = (x.size, 1)
        if self.params.gaussproc_bidimensional:
            tim = np.ones(x.size) * self.gpmodel.X[-1, 1]
            x = np.vstack([x.ravel(), tim]).T

        return self.gpmodel.predict(x)
