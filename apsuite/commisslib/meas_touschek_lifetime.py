"""."""
import time as _time
import numpy as _np
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
import scipy.optimize as _opt

from siriuspy.devices import BPM, CurrInfoSI, EGun, SOFB
from siriuspy.search.bpms_search import BPMSearch

from ..utils import ThreadedMeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class MeasTouschekParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        _ParamsBaseClass().__init__()
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
        stg = ftmp('wait_mask', self.wait_mask, '[s]')
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

    BPMNAME = 'SI-01M2:DI-BPM'
    RFFEAttMB = 0  # [dB]  Multibunch Attenuation
    RFFEAttSB = 30  # [dB] Singlebunch Attenuation
    FILTER_OUTLIER = 0.2  # Relative error data/fitting

    def __init__(self, isonline=True):
        """."""
        _BaseClass.__init__(self)
        self.data = dict(measure=dict(), analysis=dict())
        self.params = MeasTouschekParams()

        if isonline:
            self.devices['bpm'] = BPM(MeasTouschekLifetime.BPMNAME)
            self.devices['currinfo'] = CurrInfoSI()
            self.devices['egun'] = EGun()
            self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)

    def _create_si_bpms(self):
        si_bpm_filter = {'sec': 'SI', 'sub': '[0-9][0-9](M|C).*'}
        bpmsnames = BPMSearch.get_names(si_bpm_filter)
        bpms = {name: BPM(name) for name in bpmsnames}
        _ = [bpm.wait_for_connection() for bpm in bpms.values()]
        return bpms

    def set_si_bpms_attenuation(self, bpms=None, value_att=RFFEAttSB):
        """."""
        if bpms is None:
            bpms = self._create_si_bpms()

        val_old = dict()
        for name, bpm in bpms.items():
            val_old[name] = bpm.rffe_att

        for name, bpm in bpms.items():
            bpm.rffe_att = value_att
        _time.sleep(1.0)

        val_new = dict()
        for name, bpm in bpms.items():
            val_new[name] = bpm.rffe_att

        for name in bpms:
            print('{:<20s}: {} -> {}'.format(
                    name, val_old[name], val_new[name]))

    def switch_to_single_bunch(self):
        """."""
        self.devices['egun'].cmd_switch_to_single_bunch()

    def switch_to_multi_bunch(self):
        """."""
        self.devices['egun'].cmd_switch_to_multi_bunch()

    def prepare_sofb(self):
        """."""
        sofb = self.devices['sofb']
        sofb.opmode = 'MultiTurn'
        sofb.nr_points = 1
        # Change SOFB Trigger Source to Study Event and Mode to Continuous

    def _run(self, save=True):
        meas = dict(sum_a=[], sum_b=[], tim_a=[], tim_b=[], current=[])
        bpm = self.devices['bpm']
        curr = self.devices['currinfo']
        parms = self.params

        bpm.acq_nrsamples_pre = parms.acq_nrsamples_pre
        bpm.acq_nrsamples_post = parms.acq_nrsamples_post
        bpm.tbt_mask_enbl = 1
        _time.sleep(parms.wait_mask)

        idx = 0
        while True:
            if not idx % 2 and idx:
                bpm.tbt_mask_beg = parms.mask_beg_bunch_a
                bpm.tbt_mask_end = parms.mask_end_bunch_a
                _time.sleep(parms.wait_mask)
                meas['sum_a'].append(bpm.mt_possum.mean())
                meas['tim_a'].append(_time.time())
                meas['current'].append(curr.current)
            else:
                bpm.tbt_mask_beg = parms.mask_beg_bunch_b
                bpm.tbt_mask_end = parms.mask_end_bunch_b
                _time.sleep(parms.wait_mask)
                meas['sum_b'].append(bpm.mt_possum.mean())
                meas['tim_b'].append(_time.time())
            if not idx % 100:
                if save:
                    self.data['measure'] = meas
                    self.save_data(
                        fname=parms.filename, overwrite=True)
                    print(f'{idx:04d}: data saved to file.')
            idx += 1

    @staticmethod
    def _exp_fun(tim, *coeff):
        amp, off, tau = coeff
        return amp*_np.exp(-tim/tau) + off

    @staticmethod
    def _linear_fun(tim, *coeff):
        off, tau = coeff
        return off - tim/tau

    @staticmethod
    def fit_lifetime(dtime, current, window):
        """."""
        lifetimes = []

        for idx in range(len(dtime)-window):
            beg = idx
            end = idx + window
            dtm = _np.array(dtime[beg:end]) - dtime[beg]
            dtm /= 3600
            dcurr = current[beg:end]/current[beg]
            coeff, *_ = _opt.curve_fit(
                MeasTouschekLifetime._linear_fun, dtm, dcurr, p0=(1, 1))
            lifetimes.append(coeff[-1])
        return _np.array(lifetimes)

    def _remove_nans(self):
        meas = self.data['measure']
        sum_a, sum_b = meas['sum_a'], meas['sum_b']
        tim_a, tim_b = meas['tim_a'], meas['tim_b']
        currt = meas['current']

        vec = sum_a
        for _ in range(1):
            nanidx = _np.logical_not(_np.isnan(vec)).ravel()
            sum_a, sum_b = _np.array(sum_a)[nanidx], _np.array(sum_b)[nanidx]
            tim_a, tim_b = _np.array(tim_a)[nanidx], _np.array(tim_b)[nanidx]
            currt = _np.array(currt)[nanidx]
            vec = sum_b

        anly = dict()
        anly['sum_a'] = sum_a
        anly['sum_b'] = sum_b
        anly['tim_a'] = tim_a
        anly['tim_b'] = tim_b
        anly['current'] = currt
        self.data['analysis'] = anly

    def _remove_outliers(self, filter_outlier=None):
        self.calc_current_bunch()
        anly = self.data['analysis']
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
        idx_keep = (abs(diff_a) < out) & (abs(diff_b) < out)
        for key in anly.keys():
            anly[key] = _np.array(anly[key])[idx_keep]
        self.data['analysis'] = anly

    def calc_current_bunch(self):
        """."""
        anly = self.data['analysis']
        total_sum = anly['sum_a'] + anly['sum_b']
        curr_a = anly['current'] * anly['sum_a']/total_sum
        curr_b = anly['current'] * anly['sum_b']/total_sum
        anly['current_a'] = curr_a
        anly['current_b'] = curr_b
        self.data['analysis'] = anly

    def calc_touschek(self):
        """."""
        anly = self.data['analysis']
        curr_a, curr_b = anly['current_a'], anly['current_b']
        ltme_a, ltme_b = anly['total_lifetime_a'], anly['total_lifetime_b']
        window_a, window_b = anly['window_a'], anly['window_b']
        curr_a = curr_a[:-window_a]
        curr_b = curr_b[:-window_b]
        num = 1-curr_b/curr_a
        den = 1/ltme_a - 1/ltme_b
        tsck_a = num/den
        tsck_b = tsck_a * curr_a / curr_b
        anly['touschek_a'] = tsck_a
        anly['touschek_b'] = tsck_b
        self.data['analysis'] = anly

    def calc_gas_lifetime(self):
        """."""
        anly = self.data['analysis']
        glifetime = 1/anly['total_lifetime_a'] - 1/anly['touschek_a']
        anly['gas_lifetime'] = 1/glifetime
        self.data['analysis'] = anly

    def process_data(self, window_a, window_b):
        """."""
        self._remove_nans()
        self._remove_outliers()
        anly = self.data['analysis']
        tim_a, tim_b = anly['tim_a'], anly['tim_b']
        curr_a, curr_b = anly['current_a'], anly['current_b']
        anly['window_a'], anly['window_b'] = window_a, window_b
        lifetime_a = self.fit_lifetime(tim_a, curr_a, window=window_a)
        lifetime_b = self.fit_lifetime(tim_b, curr_b, window=window_b)
        anly['total_lifetime_a'] = lifetime_a
        anly['total_lifetime_b'] = lifetime_b
        self.data['analysis'] = anly
        self.calc_touschek()
        self.calc_gas_lifetime()

    def plot_touschek_lifetime(self, fname=None, title=None):
        """."""
        anly = self.data['analysis']
        curr_a, curr_b = anly['current_a'], anly['current_b']
        window_a, window_b = anly['window_a'], anly['window_b']
        tsck_a, tsck_b = anly['touschek_a'], anly['touschek_b']
        curr_a, curr_b = curr_a[:-window_a], curr_b[:-window_b]

        fig = _mplt.figure(figsize=(12, 8))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        ax1.plot(curr_a, tsck_a, '.', color='C0', label='Bunch A')
        ax1.plot(curr_b, tsck_b, '.', color='C1', label='Bunch B')
        ax1.set_xlabel('current single bunch [mA]')
        ax1.set_ylabel('Touschek lifetime [h]')
        window_time = (anly['tim_a'][window_a]-anly['tim_a'][0])/60
        stg0 = f'Fitting with window = {window_a:d} '
        stg0 += f'points ({window_time:.1f} min)'
        stg = title or stg0
        ax1.set_title(stg)
        ax1.legend()
        ax1.grid(ls='--', alpha=0.5)
        _mplt.tight_layout(True)
        if fname:
            fig.savefig(
                fname, dpi=300, format='png')

    def plot_gas_lifetime(self, fname=None, title=None):
        """."""
        anly = self.data['analysis']
        curr_a, curr_b = anly['current_a'], anly['current_b']
        window_a, window_b = anly['window_a'], anly['window_b']
        curr_a, curr_b = curr_a[:-window_a], curr_b[:-window_b]
        total_curr = curr_a + curr_b

        fig = _mplt.figure(figsize=(12, 8))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        ax1.plot(total_curr, anly['gas_lifetime'], '.', color='C0')
        ax1.set_xlabel('Total current [mA]')
        ax1.set_ylabel('Gas lifetime [h]')
        window_time = (anly['tim_a'][window_a]-anly['tim_a'][0])/60
        stg0 = f'Fitting with window = {window_a:d} '
        stg0 += f'points ({window_time:.1f} min)'
        stg = title or stg0
        ax1.set_title(stg)
        ax1.legend()
        ax1.grid(ls='--', alpha=0.5)
        _mplt.tight_layout(True)
        if fname:
            fig.savefig(
                fname, dpi=300, format='png')

    def plot_total_lifetime(self, fname=None, title=None):
        """."""
        anly = self.data['analysis']
        curr_a, curr_b = anly['current_a'], anly['current_b']
        window_a, window_b = anly['window_a'], anly['window_b']
        total_a, total_b = anly['total_lifetime_a'], anly['total_lifetime_a']
        curr_a, curr_b = curr_a[:-window_a], curr_b[:-window_b]

        fig = _mplt.figure(figsize=(12, 8))
        gs = _mgs.GridSpec(1, 1)
        ax1 = _mplt.subplot(gs[0, 0])
        ax1.plot(curr_a, total_a, '.', color='C0', label='Bunch A')
        ax1.plot(curr_b, total_b, '.', color='C1', label='Bunch B')
        ax1.set_xlabel('current single bunch [mA]')
        ax1.set_ylabel('Total lifetime [h]')
        window_time = (anly['tim_a'][window_a]-anly['tim_a'][0])/60
        stg0 = f'Fitting with window = {window_a:d} '
        stg0 += f'points ({window_time:.1f} min)'
        stg = title or stg0
        ax1.set_title(stg)
        ax1.legend()
        ax1.grid(ls='--', alpha=0.5)
        _mplt.tight_layout(True)
        if fname:
            fig.savefig(
                fname, dpi=300, format='png')

    def plot_current_decay(self, fname=None, title=None):
        """."""
        anly = self.data['analysis']
        curr_a, curr_b = anly['current_a'], anly['current_b']
        dt_a = (anly['tim_a'] - anly['tim_a'][0])/3600
        dt_b = (anly['tim_b'] - anly['tim_b'][0])/3600

        fig = _mplt.figure(figsize=(12, 8))
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
            fig.savefig(
                fname, dpi=300, format='png')
