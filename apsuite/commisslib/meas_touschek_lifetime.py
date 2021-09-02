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
        stg += dtmp('acq_nrsamples_pre', self.acq_nrsamples_pre)
        stg += dtmp('acq_nrsamples_post', self.acq_nrsamples_post)
        stg += stmp('filename', self.filename)
        return stg


class MeasTouschekLifetime(_BaseClass):
    """."""

    BPMNAME = 'SI-01M2:DI-BPM'
    RFFEAttSB = 30  # [dB]

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

        bpm.acq_nrsamples_pre = 10000
        bpm.acq_nrsamples_post = 10000
        bpm.tbt_mask_enbl = 1
        _time.sleep(parms.wait_mask)

        idx = 0
        while True:
            if not idx % 2:
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
