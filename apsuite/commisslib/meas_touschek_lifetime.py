"""."""
import time as _time
from functools import partial as _partial
from threading import Event as _Event

import numpy as _np
import numpy.polynomial.polynomial as _np_pfit
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
import scipy.optimize as _scy_opt
import scipy.integrate as _scy_int

from siriuspy.devices import BPM, CurrInfoSI, EGun, RFCav, Tune, Trigger, \
    Event, EVG
from siriuspy.search import BPMSearch
from siriuspy.epics import PV

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class MeasTouschekParams(_ParamsBaseClass):
    """."""

    DEFAULT_BPMNAME = 'SI-01M2:DI-BPM'

    def __init__(self):
        """."""
        super().__init__()
        self.total_duration = 0  # [s] 0 means infinity
        self.save_partial = True
        self.bpm_name = self.DEFAULT_BPMNAME
        self.bpm_attenuation = 14  # [dB]
        self.acquisition_timeout = 1  # [s]
        self.acquisition_period = 4  # [s]
        self.mask_beg_bunch_a = 180
        self.mask_end_bunch_a = 0
        self.mask_beg_bunch_b = 0
        self.mask_end_bunch_b = 240
        self.bucket_bunch_a = 1
        self.bucket_bunch_b = 550
        self.acq_nrsamples_pre = 10000
        self.acq_nrsamples_post = 10000
        self.filename = ''

    def __str__(self):
        """."""
        dtmp = '{0:20s} = {1:9d}\n'.format
        ftmp = '{0:20s} = {1:9.4f}  {2:s}\n'.format
        stmp = '{0:20s} = {1:}\n'.format

        stg = ''
        stg += ftmp(
            'total_duration', self.total_duration, '[s] (0) means forever')
        stg += f'{"save_partial":20s} = {str(bool(self.save_partial)):s}\n'
        stg += stmp('bpm_name', self.bpm_name)
        stg += ftmp('bpm_attenuation', self.bpm_attenuation, '[dB]')
        stg += ftmp('acquisition_timeout', self.acquisition_timeout, '[s]')
        stg += ftmp('acquisition_period', self.acquisition_period, '[s]')
        stg += dtmp('mask_beg_bunch_a', self.mask_beg_bunch_a)
        stg += dtmp('mask_end_bunch_a', self.mask_end_bunch_a)
        stg += dtmp('mask_beg_bunch_b', self.mask_beg_bunch_b)
        stg += dtmp('mask_end_bunch_b', self.mask_end_bunch_b)
        stg += dtmp('bucket_bunch_a', self.bucket_bunch_a)
        stg += dtmp('bucket_bunch_b', self.bucket_bunch_b)
        stg += dtmp('acq_nrsamples_pre', self.acq_nrsamples_pre)
        stg += dtmp('acq_nrsamples_post', self.acq_nrsamples_post)
        stg += stmp('filename', self.filename)
        return stg


class MeasTouschekLifetime(_BaseClass):
    """."""

    AVG_PRESSURE_PV = 'Calc:VA-CCG-SI-Avg:Pressure-Mon'
    RFFEAttMB = 0  # [dB]  Multibunch Attenuation
    RFFEAttSB = 30  # [dB] Singlebunch Attenuation
    FILTER_OUTLIER = 0.2  # Relative error data/fitting

    # calibration curves measured during machine shift in 2021/09/21:
    EXCCURVE_SUMA = [-1.836e-3, 1.9795e-4]
    EXCCURVE_SUMB = [-2.086e-3, 1.9875e-4]
    OFFSET_DCCT = 8.4e-3  # [mA]

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(
            self, params=MeasTouschekParams(), target=self._do_measure,
            isonline=isonline)

        self._recursion = 0
        self._updated_evt = _Event()

        if isonline:
            self._bpms = dict()
            bpmnames = BPMSearch.get_names({'sec': 'SI', 'dev': 'BPM'})
            bpm = BPM(bpmnames[0])
            self._bpms[bpmnames[0]] = bpm
            propties = bpm.auto_monitor_status
            for name in bpmnames[1:]:
                bpm = BPM(name)
                for ppt in propties:
                    bpm.set_auto_monitor(ppt, False)
                self._bpms[name] = bpm

            self.devices.update(self._bpms)
            self.devices['trigger'] = Trigger('SI-Fam:TI-BPM')
            self.devices['event'] = Event('Study')
            self.devices['evg'] = EVG()
            self.devices['currinfo'] = CurrInfoSI()
            self.devices['egun'] = EGun()
            self.devices['rfcav'] = RFCav(RFCav.DEVICES.SI)
            self.devices['tune'] = Tune(Tune.DEVICES.SI)
            self.pvs['avg_pressure'] = PV(MeasTouschekLifetime.AVG_PRESSURE_PV)

    def set_bpms_attenuation(self, value_att=RFFEAttSB):
        """."""
        if not self.isonline:
            raise ConnectionError('Cannot do that in offline mode.')

        for bpm in self._bpms.values():
            bpm.rffe_att = value_att
        _time.sleep(1.0)

        mstr = ''
        for name, bpm in self._bpms.items():
            if bpm.rffe_att != value_att:
                mstr += (
                    f'\n{name:<20s}: ' +
                    f'rb {bpm.rffe_att:.0f} != sp {value_att:.0f}')
        if not mstr:
            print('RFFE attenuation set confirmed in all BPMs.')
        else:
            print(
                'RFFE attenuation set confirmed in all BPMs, except:' + mstr)

    def cmd_switch_to_single_bunch(self):
        """."""
        return self.devices['egun'].cmd_switch_to_single_bunch()

    def cmd_switch_to_multi_bunch(self):
        """."""
        return self.devices['egun'].cmd_switch_to_multi_bunch()

    def process_data(
            self, proc_type='fit_model', nr_bunches=1, nr_intervals=1,
            window=1000, include_bunlen=False, outlier_std=6,
            outlier_max_recursion=3):
        """."""
        if 'analysis' in self.data:
            self.analysis = self.data.pop('analysis')
        if 'measure' in self.data:
            self.data = self.data.pop('measure')

        # Pre-processing of data:
        self._handle_data_lens()
        self._remove_nans()
        self._calc_current_per_bunch(nr_bunches=nr_bunches)
        self._remove_outliers(
            num_std=outlier_std, max_recursion=outlier_max_recursion)

        if proc_type.lower().startswith('fit_model'):
            self._process_model_totalrate(nr_intervals=nr_intervals)
        else:
            self._process_diffbunches(window, include_bunlen)

    @classmethod
    def totalrate_model(cls, curr, *coeff):
        """."""
        total = cls.gasrate_model(curr, *coeff[:-2])
        total += cls.touschekrate_model(curr, *coeff[-2:])
        return total

    @classmethod
    def gasrate_model(cls, curr, *gases):
        """."""
        nr_gas = len(gases)
        totsiz = curr.size//2
        quo = totsiz // nr_gas
        rest = totsiz % nr_gas
        lst = []
        for i, gas in enumerate(gases):
            siz = quo + (i < rest)
            lst.extend([gas, ]*siz)
        return _np.r_[lst, lst]

    @classmethod
    def touschekrate_model(cls, curr, *coeff):
        """."""
        tous = coeff[-2]
        blen = coeff[-1]
        return tous*curr/(1 + blen*curr)

    @classmethod
    def curr_model(cls, curr, *coeff, tim=None):
        """."""
        size = curr.size // 2
        curra = curr[:size]
        currb = curr[size:]

        drate_mod = -cls.totalrate_model(curr, *coeff)
        dratea_mod = drate_mod[:size]
        drateb_mod = drate_mod[size:]

        curra_mod = _scy_int.cumtrapz(dratea_mod * curra, x=tim, initial=0.0)
        currb_mod = _scy_int.cumtrapz(drateb_mod * currb, x=tim, initial=0.0)
        curra_mod += curra.mean() - curra_mod.mean()
        currb_mod += currb.mean() - currb_mod.mean()
        return _np.r_[curra_mod, currb_mod]

    def plot_touschek_lifetime(
            self, fname=None, title=None, fitting=False, rate=True):
        """."""
        anly = self.analysis
        curr_a, curr_b = anly['current_a'], anly['current_b']
        tsck_a, tsck_b = anly['touschek_a'], anly['touschek_b']
        window = anly.get('window', 1)

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        pwr = -1 if rate else 1
        ax1.plot(curr_a, tsck_a**pwr, '.', color='C0', label='Bunch A')
        ax1.plot(curr_b, tsck_b**pwr, '.', color='C1', label='Bunch B')

        if fitting:
            currs = _np.r_[curr_a, curr_b]
            tscks = _np.r_[tsck_a, tsck_b]
            poly = _np_pfit.polyfit(currs, 1/tscks, deg=1)
            currs_fit = _np.linspace(currs.min(), currs.max(), 2*currs.size)
            rate_fit = _np_pfit.polyval(currs_fit, poly)
            tsck_fit = 1/rate_fit
            label = r"Fitting, $\tau \times I_b$={:.4f} C".format(3.6*poly[1])
            ax1.plot(
                currs_fit, tsck_fit**pwr, '--', color='k', lw=3, label=label)

        ax1.set_xlabel('current single bunch [mA]')
        ylabel = 'rate [1/h]' if rate else 'lifetime [h]'
        ax1.set_ylabel('Touschek ' + ylabel)
        window_time = anly['tim_a'][window]/60
        stg0 = f'Fitting with window = {window:d} '
        stg0 += f'points ({window_time:.1f} min)'
        stg = title or stg0
        ax1.set_title(stg)
        ax1.legend()
        ax1.grid(ls='--', alpha=0.5)
        _mplt.tight_layout(True)
        if fname:
            fig.savefig(fname, dpi=300, format='png')
        return fig, ax1

    def plot_gas_lifetime(self, fname=None, title=None, rate=True):
        """."""
        anly = self.analysis
        curr_a, curr_b = anly['current_a'], anly['current_b']
        gaslt = anly['gas_lifetime']
        window = anly.get('window', 1)

        total_curr = curr_a + curr_b

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])

        pwr = -1 if rate else 1
        ax1.plot(total_curr, gaslt**pwr, '.', color='C0')
        ax1.set_xlabel('Total current [mA]')

        ylabel = 'rate [1/h]' if rate else 'lifetime [h]'
        ax1.set_ylabel('Gas ' + ylabel)

        window_time = anly['tim_a'][window]/60
        stg0 = f'Fitting with window = {window:d} '
        stg0 += f'points ({window_time:.1f} min)'
        stg = title or stg0
        ax1.set_title(stg)

        ax1.grid(ls='--', alpha=0.5)
        _mplt.tight_layout(True)
        if fname:
            fig.savefig(fname, dpi=300, format='png')
        return fig, ax1

    def plot_total_lifetime(
            self, fname=None, title=None, fitting=False,
            rate=True, errors=True):
        """."""
        anly = self.analysis
        curr_a, curr_b = anly['current_a'], anly['current_b']
        total_a, total_b = anly['total_lifetime_a'], anly['total_lifetime_b']
        err_a, err_b = anly['fiterror_a'], anly['fiterror_b']
        window = anly.get('window', 1)

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        pwr = -1 if rate else 1

        if errors:
            errbar_a = err_a/total_a**2 if rate else err_a
            ax1.errorbar(
                curr_a, total_a**pwr, yerr=errbar_a,
                marker='.', ls='', color='C0',
                label=f'Bunch A - Max. Error: {_np.max(errbar_a):.2e}')
            errbar_b = err_b/total_b**2 if rate else err_b
            ax1.errorbar(
                curr_b, total_b**pwr, yerr=errbar_b,
                marker='.', ls='', color='C1',
                label=f'Bunch B - Max. Error: {_np.max(errbar_b):.2e}')
        else:
            ax1.plot(curr_a, total_a**pwr, '-', color='C0', label='Bunch A')
            ax1.plot(curr_b, total_b**pwr, '-', color='C1', label='Bunch B')

        if fitting:
            currs = _np.hstack((curr_a, curr_b))
            totls = _np.hstack((total_a, total_b))
            poly = _np_pfit.polyfit(currs, 1/totls, deg=1)
            currs_fit = _np.linspace(currs.min(), currs.max(), 2*currs.size)
            rate_fit = _np_pfit.polyval(currs_fit, poly)
            totls = 1/rate_fit
            label = 'Fitting'
            ax1.plot(
                currs_fit, totls**pwr, ls='--', color='k', lw=3, label=label)

        ax1.set_xlabel('current single bunch [mA]')
        ylabel = 'rate [1/h]' if rate else 'lifetime [h]'
        ax1.set_ylabel('Total ' + ylabel)
        window_time = anly['tim_a'][window]/60
        stg0 = f'Fitting with window = {window:d} '
        stg0 += f'points ({window_time:.1f} min)'
        stg = title or stg0
        ax1.set_title(stg)
        ax1.legend()
        ax1.grid(ls='--', alpha=0.5)
        _mplt.tight_layout(True)
        if fname:
            fig.savefig(fname, dpi=300, format='png')
        return fig, ax1

    def plot_fitting_error(self, fname=None, title=None):
        """."""
        anly = self.analysis
        curr_a, curr_b = anly['current_a'], anly['current_b']
        fiterror_a, fiterror_b = anly['fiterror_a'], anly['fiterror_b']
        window = anly.get('window', 1)

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        ax1.plot(curr_a, fiterror_a, '.', color='C0', label='Bunch A')
        ax1.plot(curr_b, fiterror_b, '.', color='C1', label='Bunch B')

        ax1.set_xlabel('current single bunch [mA]')
        ax1.set_ylabel('Fitting Error')
        window_time = anly['tim_a'][window]/60
        stg0 = f'Fitting with window = {window:d} '
        stg0 += f'points ({window_time:.1f} min)'
        stg = title or stg0
        ax1.set_title(stg)
        ax1.legend()
        ax1.grid(ls='--', alpha=0.5)
        _mplt.tight_layout(True)
        if fname:
            fig.savefig(fname, dpi=300, format='png')
        return fig, ax1

    def plot_current_decay(self, fname=None, title=None):
        """."""
        anly = self.analysis
        curr_a, curr_b = anly['current_a'], anly['current_b']
        dt_a = anly['tim_a']/3600
        dt_b = anly['tim_b']/3600

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        ax1.plot(dt_a, curr_a, '.', color='C0', label='Bunch A')
        ax1.plot(dt_b, curr_b, '.', color='C1', label='Bunch B')
        ax1.set_xlabel('time [h]')
        ax1.set_ylabel('bunch current [mA]')
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(ls='--', alpha=0.5)
        _mplt.tight_layout(True)
        if fname:
            fig.savefig(fname, dpi=300, format='png')
        return fig, ax1

    def _handle_data_lens(self):
        meas = self.data
        len_min = min(len(meas['sum_a']), len(meas['sum_b']))

        sum_a = _np.array(meas['sum_a'])[:len_min]
        sum_b = _np.array(meas['sum_b'])[:len_min]
        tim_a = _np.array(meas['tim_a'])[:len_min]
        tim_b = _np.array(meas['tim_b'])[:len_min]
        currt = _np.array(meas['current'])[:len_min]

        anly = dict()
        anly['sum_a'], anly['sum_b'] = sum_a, sum_b
        anly['tim_a'], anly['tim_b'] = tim_a - tim_a[0], tim_b - tim_b[0]
        anly['current'] = currt
        self.analysis = anly

    def _remove_nans(self):
        anly = self.analysis
        sum_a, sum_b = anly['sum_a'], anly['sum_b']
        tim_a, tim_b = anly['tim_a'], anly['tim_b']
        currt = anly['current']

        nanidx = _np.isnan(sum_a).ravel()
        nanidx |= _np.isnan(sum_b).ravel()
        sum_a, sum_b = sum_a[~nanidx], sum_b[~nanidx]
        tim_a, tim_b = tim_a[~nanidx], tim_b[~nanidx]
        currt = currt[~nanidx]

        anly = dict()
        anly['sum_a'], anly['sum_b'] = sum_a, sum_b
        anly['tim_a'], anly['tim_b'] = tim_a, tim_b
        anly['current'] = currt
        self.analysis = anly

    def _calc_current_per_bunch(self, nr_bunches=1):
        """."""
        anly = self.analysis
        anly['current'] -= self.OFFSET_DCCT
        anly['current_a'] = _np_pfit.polyval(
            anly['sum_a']/nr_bunches, self.EXCCURVE_SUMA)
        anly['current_b'] = _np_pfit.polyval(
            anly['sum_b']/nr_bunches, self.EXCCURVE_SUMB)

    def _remove_outliers(self, num_std=6, max_recursion=3):
        anly = self.analysis

        dt_a = anly['tim_a']/3600
        dt_b = anly['tim_b']/3600
        curr_a = anly['current_a']
        curr_b = anly['current_b']

        loga = _np.log(curr_a)
        logb = _np.log(curr_b)
        pol_a = _np_pfit.polyfit(dt_a, loga, deg=1)
        pol_b = _np_pfit.polyfit(dt_b, logb, deg=1)
        fit_a = _np.exp(_np_pfit.polyval(dt_a, pol_a))
        fit_b = _np.exp(_np_pfit.polyval(dt_b, pol_b))

        diff_a = _np.abs(curr_a - fit_a)
        diff_b = _np.abs(curr_b - fit_b)

        out = num_std
        idx_keep = (diff_a < out*diff_a.std()) & (diff_b < out*diff_b.std())

        for key in anly.keys():
            anly[key] = anly[key][idx_keep]

        print('Filtering outliers: recursion = {0:d}, num = {1:d}'.format(
            self._recursion, curr_a.size - _np.sum(idx_keep)))
        if _np.sum(idx_keep) < curr_a.size and self._recursion < max_recursion:
            self._recursion += 1
            self._remove_outliers(num_std, max_recursion=max_recursion)
        else:
            self._recursion = 0

    def _process_model_totalrate(self, nr_intervals=5):
        anly = self.analysis
        curra = anly['current_a']
        currb = anly['current_b']
        tim = anly['tim_a']
        size = curra.size
        currt = _np.r_[curra, currb]

        # First do one round without bounds to use LM algorithm and find the
        # true miminum:
        coeff0 = [1/40/3600, ] * nr_intervals + [1/10/3600, 0.2]
        coeff, pconv = _scy_opt.curve_fit(
            _partial(self.curr_model, tim=tim), currt, currt, p0=coeff0)

        # Then fix the negative arguments to make the final round with bounds:
        coeff = _np.array(coeff)
        idcs = coeff < 0
        if idcs.any():
            coeff[idcs] = 0
            lower = [0, ] * (nr_intervals + 2)
            upper = [_np.inf, ] * (nr_intervals + 2)
            coeff, pconv = _scy_opt.curve_fit(
                _partial(self.curr_model, tim=tim), currt, currt, p0=coeff,
                bounds=(lower, upper))
        errs = _np.sqrt(_np.diag(pconv))

        tousrate = self.touschekrate_model(currt, *coeff[-2:])
        gasrate = self.gasrate_model(currt, *coeff[:-2])
        gasrate *= 3600
        tousrate *= 3600
        totrate = tousrate + gasrate

        currt_fit = self.curr_model(currt, *coeff, tim=tim)

        anly['coeffs'] = coeff
        anly['coeffs_pconv'] = pconv
        anly['current_a_fit'] = currt_fit[:size]
        anly['current_b_fit'] = currt_fit[size:]
        anly['total_lifetime_a'] = 1 / totrate[:size]
        anly['total_lifetime_b'] = 1 / totrate[size:]
        fiterr = _np.sqrt(_np.sum(errs*errs/_np.array(coeff)))
        anly['fiterror_a'] = 1/totrate[:size] * fiterr
        anly['fiterror_b'] = 1/totrate[size:] * fiterr

        # Calc Touschek and Gas Lifetime
        anly['touschek_a'] = 1/tousrate[:size]
        anly['touschek_b'] = 1/tousrate[size:]
        anly['gas_lifetime_a'] = 1/gasrate[:size]
        anly['gas_lifetime_b'] = 1/gasrate[size:]
        anly['gas_lifetime'] = 1/gasrate[:size]

    def _process_diffbunches(self, window, include_bunlen=False):
        anly = self.analysis

        tim_a, tim_b = anly['tim_a'], anly['tim_b']
        curr_a, curr_b = anly['current_a'], anly['current_b']
        currt = anly['current']

        # Fit total lifetime
        window = (int(window) // 2)*2 + 1
        ltime_a, fiterr_a = self._fit_lifetime(tim_a, curr_a, window=window)
        ltime_b, fiterr_b = self._fit_lifetime(tim_b, curr_b, window=window)
        anly['window'] = window
        anly['total_lifetime_a'] = ltime_a
        anly['total_lifetime_b'] = ltime_b
        anly['fiterror_a'] = fiterr_a
        anly['fiterror_b'] = fiterr_b

        # Resize all vectors to match lifetime size
        leng = window // 2
        slc = slice(leng+1, -leng)
        # slc = slice(0, -window)
        curr_a = curr_a[slc]
        curr_b = curr_b[slc]
        tim_a = tim_a[slc]
        tim_b = tim_b[slc]
        currt = currt[slc]
        anly['current_a'] = curr_a
        anly['current_b'] = curr_b
        anly['tim_a'] = tim_a - tim_a[0]
        anly['tim_b'] = tim_b - tim_a[0]
        anly['current'] = currt

        # Calc Touschek Lifetime from Total Lifetime of both bunches
        if include_bunlen:
            self._calc_touschek_lifetime()
            tsck_a = anly['touschek_a']
            tsck_b = anly['touschek_b']
        else:
            num = 1 - curr_b/curr_a
            den = 1/ltime_a - 1/ltime_b
            tsck_a = num/den
            tsck_b = tsck_a * curr_a / curr_b
            anly['touschek_a'] = tsck_a
            anly['touschek_b'] = tsck_b

        # Recover Gas Lifetime
        gas_rate_a = 1/ltime_a - 1/tsck_a
        gas_rate_b = 1/ltime_b - 1/tsck_b
        anly['gas_lifetime_a'] = 1/gas_rate_a
        anly['gas_lifetime_b'] = 1/gas_rate_b
        anly['gas_lifetime'] = 2/(gas_rate_a + gas_rate_b)

    def _calc_touschek_lifetime(self):
        """."""
        def model(curr, tous, blen):
            curr_a, curr_b = curr
            alpha_a = tous*curr_a/(1+blen*curr_a)
            alpha_b = tous*curr_b/(1+blen*curr_b)
            return alpha_a - alpha_b

        anly = self.analysis
        curr_a, curr_b = anly['current_a'], anly['current_b']
        ltime_a, ltime_b = anly['total_lifetime_a'], anly['total_lifetime_b']
        curr_a = curr_a[:ltime_a.size]
        curr_b = curr_b[:ltime_b.size]

        p0_ = (1, 1)
        alpha_total = 1/ltime_a - 1/ltime_b
        coeffs, pconv = _scy_opt.curve_fit(
            model, (curr_a, curr_b), alpha_total, p0=p0_)
        tous, blen = coeffs

        alpha_a = tous*curr_a/(1+blen*curr_a)
        alpha_b = tous*curr_b/(1+blen*curr_b)
        anly['touschek_a'] = 1/alpha_a
        anly['touschek_b'] = 1/alpha_b
        anly['alpha_total'] = alpha_total
        anly['tous_coeffs'] = coeffs
        anly['tous_coeffs_pconv'] = pconv

    def _do_measure(self):
        meas = dict(
            sum_a=[], sum_b=[], nan_a=[], nan_b=[], tim_a=[], tim_b=[],
            current=[], rf_voltage=[], avg_pressure=[], tunex=[], tuney=[])
        parms = self.params

        curr = self.devices['currinfo']
        rfcav = self.devices['rfcav']
        tune = self.devices['tune']
        press = self.pvs['avg_pressure']
        bpm = self.devices[parms.bpm_name]

        excx0 = tune.enablex
        excy0 = tune.enabley

        pvsum = bpm.pv_object('GEN_SUMArrayData')
        pvsum.auto_monitor = True
        pvsum.add_callback(self._pv_updated)

        self.devices['trigger'].source = 'Study'
        self.devices['event'].mode = 'Continuous'
        self.devices['evg'].cmd_update_events()

        swtch0 = bpm.switching
        bpm.cmd_turn_off_switching()
        bpm.cmd_acq_abort()
        bpm.acq_nrsamples_pre = parms.acq_nrsamples_pre
        bpm.acq_nrsamples_post = parms.acq_nrsamples_post
        bpm.acq_repeat = 'normal'
        bpm.acq_trigger = 'external'
        bpm.rffe_att = parms.bpm_attenuation
        bpm.tbt_mask_enbl = 1

        maxidx = parms.total_duration / parms.acquisition_period
        maxidx = float('inf') if maxidx < 1 else maxidx
        idx = 0

        while idx < maxidx and not self._stopevt.is_set():
            # Get data for bunch with higher current
            bpm.tbt_mask_beg = parms.mask_beg_bunch_a
            bpm.tbt_mask_end = parms.mask_end_bunch_a
            _time.sleep(parms.acquisition_period/4)
            self._updated_evt.clear()
            bpm.cmd_acq_start()
            bpm.wait_acq_finish(timeout=parms.acquisition_timeout)
            self._updated_evt.wait(timeout=parms.acquisition_timeout)
            suma = bpm.mt_possum
            # Use median to remove influence of bad points:
            meas['sum_a'].append(_np.nanmedian(suma))
            meas['nan_a'].append(_np.sum(_np.isnan(suma)))
            meas['tim_a'].append(_time.time())

            _time.sleep(parms.acquisition_period/4)
            
            # Get data for bunch with lower current
            bpm.tbt_mask_beg = parms.mask_beg_bunch_b
            bpm.tbt_mask_end = parms.mask_end_bunch_b
            _time.sleep(parms.acquisition_period/4)
            self._updated_evt.clear()
            bpm.cmd_acq_start()
            bpm.wait_acq_finish(timeout=parms.acquisition_timeout)
            self._updated_evt.wait(timeout=parms.acquisition_timeout)
            sumb = bpm.mt_possum
            # Use median to remove influence of bad points:
            meas['sum_b'].append(_np.nanmedian(sumb))
            meas['nan_b'].append(_np.sum(_np.isnan(sumb)))
            meas['tim_b'].append(_time.time())

            # Get other relevant parameters
            meas['current'].append(curr.current)
            meas['rf_voltage'].append(rfcav.dev_cavmon.gap_voltage)
            meas['avg_pressure'].append(press.value)

            if not idx % 100 and parms.save_partial:
                # Only get the tune at every 100 iterations not to disturb the
                # beam too much with the tune shaker
                tunex, tuney = self._get_tunes()
                meas['tunex'].append(tunex)
                meas['tuney'].append(tuney)

                self.data = meas
                self.save_data(fname=parms.filename, overwrite=True)
                print(f'{idx:04d}: data saved to file.')
            idx += 1
            _time.sleep(parms.acquisition_period/4)

        self.data = meas
        self.save_data(fname=parms.filename, overwrite=True)
        print(f'{idx:04d}: data saved to file.')

        if excx0:
            tune.cmd_enablex()
        if excy0:
            tune.cmd_enabley()
        if swtch0:
            bpm.cmd_turn_on_switching()

        self.devices['trigger'].source = 'DigSI'
        self.devices['event'].mode = 'External'
        self.devices['evg'].cmd_update_events()
        pvsum.clear_callbacks()
        pvsum.auto_monitor = False

        print('Done!')

    def _pv_updated(self, *args, **kwargs):
        _ = args, kwargs
        self._updated_evt.set()

    def _get_tunes(self):
        tune = self.devices['tune']
        tune.cmd_enablex()
        tune.cmd_enabley()
        _time.sleep(1)
        tunex = tune.tunex
        tuney = tune.tuney
        tune.cmd_disablex()
        tune.cmd_disabley()
        return tunex, tuney

    @staticmethod
    def _linear_fun(tim, *coeff):
        amp, tau = coeff
        return amp*(1 - tim/tau)

    @staticmethod
    def _fit_lifetime(dtime, current, window):
        """."""
        lifetimes, fiterrors = [], []
        for idx in range(len(dtime)-window):
            beg = idx
            end = idx + window
            dtm = _np.array(dtime[beg:end]) - dtime[beg]
            dtm /= 3600
            dcurr = current[beg:end]/current[beg]
            coeff, pconv = _scy_opt.curve_fit(
                MeasTouschekLifetime._linear_fun, dtm, dcurr, p0=(1, 1))
            errs = _np.sqrt(_np.diag(pconv))
            lifetimes.append(coeff[-1])
            fiterrors.append(errs[-1])
        return _np.array(lifetimes), _np.array(fiterrors)
