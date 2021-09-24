"""."""
import time as _time
from functools import partial as _partial

import numpy as _np
import numpy.polynomial.polynomial as pfit
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
import scipy.optimize as _scy_opt
import scipy.integrate as _scy_int

from siriuspy.devices import BPM, CurrInfoSI, EGun, RFCav
from siriuspy.search.bpms_search import BPMSearch
from siriuspy.epics import PV

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class MeasTouschekParams(_ParamsBaseClass):
    """."""

    DEFAULT_BPMNAME = 'SI-01M2:DI-BPM'

    def __init__(self):
        """."""
        _ParamsBaseClass().__init__()
        self.total_duration = 0  # [s] 0 means infinity
        self.save_partial = True
        self.bpm_name = self.DEFAULT_BPMNAME
        self.bpm_attenuation = 14  # [dB]
        self.wait_mask = 2  # [s]
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
        stg += f'save_partial = {str(bool(self.save_partial)):s}'
        stg += stmp('bpm_name', self.bpm_name)
        stg += ftmp('bpm_attenuation', self.bpm_attenuation, '[dB]')
        stg += ftmp('wait_mask', self.wait_mask, '[s]')
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

        if isonline:
            bpmsnames = BPMSearch.get_names({'sec': 'SI', 'dev': 'BPM'})
            self._bpms = {name: BPM(name) for name in bpmsnames}
            self.devices.update(self._bpms)
            self.devices['currinfo'] = CurrInfoSI()
            self.devices['egun'] = EGun()
            self.devices['rfcav'] = RFCav(RFCav.DEVICES.SI)
            self.pvs['avg_pressure'] = PV(MeasTouschekLifetime.AVG_PRESSURE_PV)

    def set_si_bpms_attenuation(self, value_att=RFFEAttSB):
        """."""
        if not self.isonline:
            raise ConnectionError('Cannot do that in offline mode.')

        val_old = {name: bpm.rffe_att for name, bpm in self._bpms.items()}
        for bpm in self._bpms.values():
            bpm.rffe_att = value_att
        _time.sleep(1.0)

        for name, bpm in self._bpms.items():
            print(f'{name:<20s}: {val_old[name]:.0f} -> {bpm.rffe_att:.0f}')

    def cmd_switch_to_single_bunch(self):
        """."""
        return self.devices['egun'].cmd_switch_to_single_bunch()

    def cmd_switch_to_multi_bunch(self):
        """."""
        return self.devices['egun'].cmd_switch_to_multi_bunch()

    def process_data(
            self, proc_type='fit_model', nr_bunches=1, nr_intervals=1,
            window=1000, include_bunlen=False):
        """."""
        if 'analysis' in self.data:
            self.analysis = self.data.pop('analysis')
        if 'measure' in self.data:
            self.data = self.data.pop('measure')

        # Pre-processing of data:
        self._handle_data_lens()
        self._remove_nans()
        self._calc_current_per_bunch(nr_bunches=nr_bunches)
        self._remove_outliers()

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
    def curr_model(cls, curr, *coeff, dtim=1):
        """."""
        size = curr.size // 2
        curra = curr[:size]
        currb = curr[size:]

        drate_mod = -cls.totalrate_model(curr, *coeff)
        dratea_mod = drate_mod[:size]
        drateb_mod = drate_mod[size:]

        curra_mod = _scy_int.cumtrapz(dratea_mod * curra, dx=dtim, initial=0.0)
        currb_mod = _scy_int.cumtrapz(drateb_mod * currb, dx=dtim, initial=0.0)
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
            poly = pfit.polyfit(currs, 1/tscks, deg=1)
            currs_fit = _np.linspace(currs.min(), currs.max(), 2*currs.size)
            rate_fit = pfit.polyval(currs_fit, poly)
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
            poly = pfit.polyfit(currs, 1/totls, deg=1)
            currs_fit = _np.linspace(currs.min(), currs.max(), 2*currs.size)
            rate_fit = pfit.polyval(currs_fit, poly)
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
        anly['current_a'] = pfit.polyval(
            anly['sum_a']/nr_bunches, self.EXCCURVE_SUMA)
        anly['current_b'] = pfit.polyval(
            anly['sum_b']/nr_bunches, self.EXCCURVE_SUMB)

    def _remove_outliers(self, filter_outlier=None):
        anly = self.analysis

        dt_a = anly['tim_a']/3600
        dt_b = anly['tim_b']/3600
        curr_a = anly['current_a']
        curr_b = anly['current_b']
        func = MeasTouschekLifetime._exp_fun
        p0_ = (1, 1, 1)
        coeff_a, *_ = _scy_opt.curve_fit(func, dt_a, curr_a, p0=p0_)
        coeff_b, *_ = _scy_opt.curve_fit(func, dt_b, curr_b, p0=p0_)
        fit_a = func(dt_a, *coeff_a)
        fit_b = func(dt_b, *coeff_b)
        diff_a = (curr_a - fit_a)/curr_a
        diff_b = (curr_b - fit_b)/curr_b
        out = filter_outlier or MeasTouschekLifetime.FILTER_OUTLIER
        idx_keep = (_np.abs(diff_a) < out) & (_np.abs(diff_b) < out)
        for key in anly.keys():
            anly[key] = _np.array(anly[key])[idx_keep]
        self.analysis = anly

    def _process_model_totalrate(self, nr_intervals=5):
        anly = self.analysis
        curra = anly['current_a']
        currb = anly['current_b']
        dtim = anly['tim_a'][1]
        size = curra.size
        currt = _np.r_[curra, currb]

        # First do one round without bounds to use LM algorithm and find the
        # true miminum:
        coeff0 = [1/40/3600, ] * nr_intervals + [1/10/3600, 0.2]
        coeff, pconv = _scy_opt.curve_fit(
            _partial(self.curr_model, dtim=dtim), currt, currt, p0=coeff0)

        # Then fix the negative arguments to make the final round with bounds:
        coeff = _np.array(coeff)
        idcs = coeff < 0
        if idcs.any():
            coeff[idcs] = 0
            lower = [0, ] * (nr_intervals + 2)
            upper = [_np.inf, ] * (nr_intervals + 2)
            coeff, pconv = _scy_opt.curve_fit(
                _partial(self.curr_model, dtim=dtim), currt, currt, p0=coeff,
                bounds=(lower, upper))
        errs = _np.sqrt(_np.diag(pconv))

        tousrate = self.touschekrate_model(currt, *coeff[-2:])
        gasrate = self.gasrate_model(currt, *coeff[:-2])
        gasrate *= 3600
        tousrate *= 3600
        totrate = tousrate + gasrate

        currt_fit = self.curr_model(currt, *coeff, dtim=dtim)

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
            sum_a=[], sum_b=[], tim_a=[], tim_b=[], current=[],
            rf_voltage=[], avg_pressure=[])
        parms = self.params

        curr = self.devices['currinfo']
        rfcav = self.devices['rfcav']
        press = self.pvs['avg_pressure']
        bpm = self.devices[parms.bpm_name]

        bpm.acq_nrsamples_pre = parms.acq_nrsamples_pre
        bpm.acq_nrsamples_post = parms.acq_nrsamples_post
        bpm.rffe_att = parms.bpm_attenuation
        bpm.tbt_mask_enbl = 1
        _time.sleep(parms.wait_mask)

        maxidx = parms.total_duration / (2*parms.wait_mask)
        maxidx = float('inf') if maxidx < 1 else maxidx
        idx = 0
        while idx < maxidx and not self._stopevt.is_set():
            bpm.tbt_mask_beg = parms.mask_beg_bunch_a
            bpm.tbt_mask_end = parms.mask_end_bunch_a
            _time.sleep(parms.wait_mask)
            meas['sum_a'].append(bpm.mt_possum.mean())
            meas['tim_a'].append(_time.time())

            meas['current'].append(curr.current)
            meas['rf_voltage'].append(rfcav.dev_cavmon.gap_voltage)
            meas['avg_pressure'].append(press.value)

            bpm.tbt_mask_beg = parms.mask_beg_bunch_b
            bpm.tbt_mask_end = parms.mask_end_bunch_b
            _time.sleep(parms.wait_mask)
            meas['sum_b'].append(bpm.mt_possum.mean())
            meas['tim_b'].append(_time.time())
            if not idx % 100 and parms.save_partial:
                self.data = meas
                self.save_data(fname=parms.filename, overwrite=True)
                print(f'{idx:04d}: data saved to file.')
            idx += 1

        self.data = meas
        self.save_data(fname=parms.filename, overwrite=True)
        print(f'{idx:04d}: data saved to file.')
        print('Done!')

    @staticmethod
    def _linear_fun(tim, *coeff):
        amp, tau = coeff
        return amp*(1 - tim/tau)

    @staticmethod
    def _exp_fun(tim, *coeff):
        amp, off, tau = coeff
        return amp*_np.exp(-tim/tau) + off

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
