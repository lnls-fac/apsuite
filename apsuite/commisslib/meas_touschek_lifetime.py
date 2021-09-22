"""."""
import time as _time
import numpy as _np
import numpy.polynomial.polynomial as pfit
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
import scipy.optimize as _opt

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
            self._bpms = self._create_si_bpms()
            self.devices.update(self._bpms)
            self.devices['currinfo'] = CurrInfoSI()
            self.devices['egun'] = EGun()
            self.devices['rfcav'] = RFCav(RFCav.DEVICES.SI)
            self.devices['avg_pressure'] = PV(
                MeasTouschekLifetime.AVG_PRESSURE_PV)

    def _create_si_bpms(self):
        bpmsnames = BPMSearch.get_names({'sec': 'SI', 'dev': 'BPM'})
        return {name: BPM(name) for name in bpmsnames}

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

    def _do_measure(self):
        meas = dict(
            sum_a=[], sum_b=[], tim_a=[], tim_b=[], current=[],
            rf_voltage=[], avg_pressure=[])
        parms = self.params

        curr = self.devices['currinfo']
        rfcav = self.devices['rfcav']
        press = self.devices['avg_pressure']
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
    def _exp_fun(tim, *coeff):
        amp, off, tau = coeff
        return amp*_np.exp(-tim/tau) + off

    @staticmethod
    def _linear_fun(tim, *coeff):
        amp, tau = coeff
        return amp*(1 - tim/tau)

    @staticmethod
    def _exp_model(timcurrs, *coeff):
        # curra0, currb0, gas_rate, gamma, delta = coeff
        curra0, currb0, gamma, delta = coeff
        tim, curra, currb = timcurrs
        tsck_rate_a = gamma*curra/(1+delta*curra)
        tsck_rate_b = gamma*currb/(1+delta*currb)
        # arga = -(gas_rate + tsck_rate_a)*tim
        # argb = -(gas_rate + tsck_rate_b)*tim
        arga = -tsck_rate_a*tim
        argb = -tsck_rate_b*tim
        # return _np.r_[curra0*_np.exp(arga), currb0*_np.exp(argb)]
        # return curra0*_np.exp(arga) + currb0*_np.exp(argb)
        return currb0/curra0*_np.exp(argb-arga)

    @staticmethod
    def fit_lifetime(dtime, current, window):
        """."""
        lifetimes = []
        fiterrors = []

        for idx in range(len(dtime)-window):
            beg = idx
            end = idx + window
            dtm = _np.array(dtime[beg:end]) - dtime[beg]
            dtm /= 3600
            dcurr = current[beg:end]/current[beg]
            coeff, pconv = _opt.curve_fit(
                MeasTouschekLifetime._linear_fun, dtm, dcurr, p0=(1, 1))
            errs = _np.sqrt(_np.diag(pconv))
            lifetimes.append(coeff[-1])
            fiterrors.append(errs[-1])
        return _np.array(lifetimes), _np.array(fiterrors)

    @staticmethod
    def fit_lifetime_alt(dtime, current_a, current_b):
        """."""
        p0_ = (current_a[0], current_b[0], 1, 0.5)
        # goal = current_a + current_b
        # goal = _np.r_[current_a, current_b]
        goal = current_b/current_a
        coeffs, pconv = _opt.curve_fit(
            MeasTouschekLifetime._exp_model,
            (dtime, current_a, current_b),
            goal, p0=p0_)
        errs = _np.sqrt(_np.diag(pconv))
        return coeffs, errs

    def _handle_data_lens(self):
        meas = self.data.get('measure', self.data)
        len_min = min(len(meas['sum_a']), len(meas['sum_b']))

        sum_a, sum_b = meas['sum_a'][:len_min], meas['sum_b'][:len_min]
        tim_a, tim_b = meas['tim_a'][:len_min], meas['tim_b'][:len_min]
        currt = meas['current'][:len_min]

        anly = dict()
        anly['sum_a'], anly['sum_b'] = sum_a, sum_b
        anly['tim_a'], anly['tim_b'] = tim_a, tim_b
        anly['current'] = currt
        self.analysis = anly

    def _remove_nans(self):
        anly = self.analysis
        sum_a, sum_b = anly['sum_a'], anly['sum_b']
        tim_a, tim_b = anly['tim_a'], anly['tim_b']
        currt = anly['current']

        nanidx = _np.logical_not(_np.isnan(sum_a)).ravel()
        nanidx |= _np.logical_not(_np.isnan(sum_b)).ravel()
        sum_a, sum_b = _np.array(sum_a)[nanidx], _np.array(sum_b)[nanidx]
        tim_a, tim_b = _np.array(tim_a)[nanidx], _np.array(tim_b)[nanidx]
        currt = _np.array(currt)[nanidx]

        anly = dict()
        anly['sum_a'], anly['sum_b'] = sum_a, sum_b
        anly['tim_a'], anly['tim_b'] = tim_a, tim_b
        anly['current'] = currt
        self.analysis = anly

    def _remove_outliers(self, filter_outlier=None):
        anly = self.analysis

        dt_a = (anly['tim_a'] - anly['tim_a'][0])/3600
        dt_b = (anly['tim_b'] - anly['tim_b'][0])/3600
        curr_a = anly['current_a']
        curr_b = anly['current_b']
        func = MeasTouschekLifetime._exp_fun
        p0_ = (1, 1, 1)
        coeff_a, *_ = _opt.curve_fit(func, dt_a, curr_a, p0=p0_)
        coeff_b, *_ = _opt.curve_fit(func, dt_b, curr_b, p0=p0_)
        fit_a = func(dt_a, *coeff_a)
        fit_b = func(dt_b, *coeff_b)
        diff_a = (curr_a - fit_a)/curr_a
        diff_b = (curr_b - fit_b)/curr_b
        out = filter_outlier or MeasTouschekLifetime.FILTER_OUTLIER
        idx_keep = (_np.abs(diff_a) < out) & (_np.abs(diff_b) < out)
        for key in anly.keys():
            anly[key] = _np.array(anly[key])[idx_keep]
        self.analysis = anly

    def _calc_current_per_bunch(self, nr_bunches):
        """."""
        anly = self.analysis
        anly['current'] -= self.OFFSET_DCCT
        curr_a = pfit.polyval(anly['sum_a']/nr_bunches, self.EXCCURVE_SUMA)
        curr_b = pfit.polyval(anly['sum_b']/nr_bunches, self.EXCCURVE_SUMB)
        anly['current_a'] = curr_a
        anly['current_b'] = curr_b
        self.analysis = anly

    def calc_touschek_lifetime(self):
        """."""
        def model(curr, gamma, delta):
            curr_a, curr_b = curr
            # alpha_a = gamma*curr_a/(1+delta*curr_a+beta*curr_a**2)
            # alpha_b = gamma*curr_b/(1+delta*curr_b+beta*curr_b**2)
            alpha_a = gamma*curr_a/(1+delta*curr_a)
            alpha_b = gamma*curr_b/(1+delta*curr_b)
            return alpha_a - alpha_b

        anly = self.analysis
        curr_a, curr_b = anly['current_a'], anly['current_b']
        ltme_a, ltme_b = anly['total_lifetime_a'], anly['total_lifetime_b']
        window_a, window_b = anly['window_a'], anly['window_b']
        curr_a = curr_a[:-window_a]
        curr_b = curr_b[:-window_b]
        # num = 1-curr_b/curr_a
        # den = 1/ltme_a - 1/ltme_b
        # tsck_a = num/den
        # tsck_b = tsck_a * curr_a / curr_b

        p0_ = (1, 1)
        alpha_total = 1/ltme_a - 1/ltme_b
        coeffs, pcov = _opt.curve_fit(
            model, (curr_a, curr_b), alpha_total, p0=p0_)
        errs = _np.sqrt(_np.diag(pcov))
        # gamma, delta, beta = coeffs
        gamma, delta = coeffs
        # alpha_a = gamma*curr_a/(1+delta*curr_a+beta*curr_a**2)
        # alpha_b = gamma*curr_b/(1+delta*curr_b+beta*curr_b**2)
        alpha_a = gamma*curr_a/(1+delta*curr_a)
        alpha_b = gamma*curr_b/(1+delta*curr_b)
        anly['touschek_a'] = 1/alpha_a
        anly['touschek_b'] = 1/alpha_b
        anly['gamma'] = gamma
        anly['delta'] = delta
        # anly['beta'] = beta
        anly['gamma_error'] = errs[0]
        anly['delta_error'] = errs[1]
        # anly['beta_error'] = errs[2]
        anly['alpha_total'] = alpha_total
        self.analysis = anly

    def calc_gas_lifetime(self):
        """."""
        anly = self.analysis
        gas_rate_a = 1/anly['total_lifetime_a'] - 1/anly['touschek_a']
        gas_rate_b = 1/anly['total_lifetime_b'] - 1/anly['touschek_b']
        anly['gas_lifetime_a'] = 1/gas_rate_a
        anly['gas_lifetime_b'] = 1/gas_rate_b
        anly['gas_lifetime'] = (1/gas_rate_a + 1/gas_rate_b)/2
        self.data['analysis'] = anly

    def process_data(self, window_a, window_b, nr_bunches):
        """."""
        if 'analysis' in self.data:
            self.analysis = self.data.pop('analysis')

        self._handle_data_lens()
        self._remove_nans()
        self._calc_current_per_bunch(nr_bunches=nr_bunches)
        self._remove_outliers()
        anly = self.analysis

        tim_a, tim_b = anly['tim_a'], anly['tim_b']
        curr_a, curr_b = anly['current_a'], anly['current_b']
        anly['window_a'], anly['window_b'] = window_a, window_b
        lifetime_a, fiterror_a = self.fit_lifetime(
            tim_a, curr_a, window=window_a)
        lifetime_b, fiterror_b = self.fit_lifetime(
            tim_b, curr_b, window=window_b)
        anly['total_lifetime_a'] = lifetime_a
        anly['total_lifetime_b'] = lifetime_b
        anly['fiterror_a'] = fiterror_a
        anly['fiterror_b'] = fiterror_b

        # dtime = (tim_a - tim_a[0])/3600
        # coeffs, errs = self.fit_lifetime_alt(dtime, curr_a, curr_b)
        # anly['initial_curr_a'] = coeffs[0]
        # anly['initial_curr_b'] = coeffs[1]
        # # anly['gas_rate'] = coeffs[2]
        # anly['gamma'] = coeffs[2]
        # anly['delta'] = coeffs[3]
        self.analysis = anly
        self.calc_touschek_lifetime()
        self.calc_gas_lifetime()

    def plot_touschek_lifetime(
            self, fname=None, title=None, fitting=False, rate=True):
        """."""
        anly = self.analysis
        curr_a, curr_b = anly['current_a'], anly['current_b']
        window_a, window_b = anly['window_a'], anly['window_b']
        tsck_a, tsck_b = anly['touschek_a'], anly['touschek_b']
        curr_a, curr_b = curr_a[:-window_a], curr_b[:-window_b]

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        pwr = -1 if rate else 1
        ax1.plot(curr_a, tsck_a**pwr, '.', color='C0', label='Bunch A')
        ax1.plot(curr_b, tsck_b**pwr, '.', color='C1', label='Bunch B')

        if fitting:
            currs = _np.hstack((curr_a, curr_b))
            tscks = _np.hstack((tsck_a, tsck_b))
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
        window_time = (anly['tim_a'][window_a]-anly['tim_a'][0])/60
        stg0 = f'Fitting with window = {window_a:d} '
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
        window_a, window_b = anly['window_a'], anly['window_b']
        curr_a, curr_b = curr_a[:-window_a], curr_b[:-window_b]
        total_curr = curr_a + curr_b

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        pwr = -1 if rate else 1
        ax1.plot(total_curr, anly['gas_lifetime']**pwr, '.', color='C0')
        ax1.set_xlabel('Total current [mA]')
        ylabel = 'rate [1/h]' if rate else 'lifetime [h]'
        ax1.set_ylabel('Gas ' + ylabel)
        window_time = (anly['tim_a'][window_a]-anly['tim_a'][0])/60
        stg0 = f'Fitting with window = {window_a:d} '
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
        window_a, window_b = anly['window_a'], anly['window_b']
        total_a, total_b = anly['total_lifetime_a'], anly['total_lifetime_b']
        err_a, err_b = anly['fiterror_a'], anly['fiterror_b']
        curr_a, curr_b = curr_a[:-window_a], curr_b[:-window_b]

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
        window_time = (anly['tim_a'][window_a]-anly['tim_a'][0])/60
        stg0 = f'Fitting with window = {window_a:d} '
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
        window_a, window_b = anly['window_a'], anly['window_b']
        fiterror_a, fiterror_b = anly['fiterror_a'], anly['fiterror_b']
        curr_a, curr_b = curr_a[:-window_a], curr_b[:-window_b]

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        ax1.plot(curr_a, fiterror_a, '.', color='C0', label='Bunch A')
        ax1.plot(curr_b, fiterror_b, '.', color='C1', label='Bunch B')

        ax1.set_xlabel('current single bunch [mA]')
        ax1.set_ylabel('Fitting Error')
        window_time = (anly['tim_a'][window_a]-anly['tim_a'][0])/60
        stg0 = f'Fitting with window = {window_a:d} '
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
        dt_a = (anly['tim_a'] - anly['tim_a'][0])/3600
        dt_b = (anly['tim_b'] - anly['tim_b'][0])/3600

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
