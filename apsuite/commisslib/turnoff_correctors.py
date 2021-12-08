"""."""
import time as _time
import numpy as _np
from siriuspy.devices import PowerSupply, CurrInfoSI, Tune, TuneCorr, \
    SOFB, FamBPMs, Trigger, Event, EVG, RFGen
from siriuspy.sofb.csdev import SOFBFactory
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient

from ..utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class TurnOffCorrParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.max_delta_orbit = 500  # [um]
        self.min_orbres = 5  # [um]
        self.max_tunex_var = 0.005
        self.max_tuney_var = 0.005
        self.chs_idx = []
        self.nr_orbcorrs = 5
        self.wait_tunecorr = 3  # [s]
        self.wait_iter = 3  # [s]
        self.nr_points_bpm_acq = 10000
        self.bpms_timeout = 30  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += ftmp('max_delta_orbit', self.max_delta_orbit, '[um]')
        stg += ftmp('min_orbres', self.min_orbres, '[um]')
        stg += ftmp('max_tunex_var', self.max_tunex_var, '')
        stg += ftmp('max_tuney_var', self.max_tuney_var, '')
        stg += stmp('chs_idx', str(self.chs_idx), '')
        stg += dtmp('nr_orbcorrs', self.nr_orbcorrs, '')
        stg += ftmp('wait_tunecorr', self.wait_tunecorr, '[s]')
        stg += ftmp('wait_iter', self.wait_iter, '[s]')
        stg += dtmp('nr_points_bpm_acq', self.nr_points_bpm_acq, '')
        stg += ftmp('bpms_timeout', self.bpms_timeout, '[s]')
        return stg


class TurnOffCorr(_ThreadBaseClass):
    """."""

    MIN_CURRENT = 0.1

    def __init__(self, params=None, isonline=True):
        """."""
        params = TurnOffCorrParams() if params is None else params
        super().__init__(
            params=params, target=self._do_measure, isonline=isonline)
        self.sofb_data = SOFBFactory.create('SI')
        # self.chs_subset = self.sofb_data.ch_names[self.params.chs_idx]
        # client = _ConfigDBClient(config_type='si_orbcorr_respm')
        # respmat = _np.array(client.get_config_value(name='ref_respmat'))
        # respmat = _np.reshape(respmat, (320, 281))
        # self.respmat = respmat
        self.tunex0, self.tuney0 = None, None
        self.initial_kicks = None
        if self.isonline:
            self._create_devices()
            self.respmat = self.devices['sofb'].respmat

    def _create_devices(self):
        self.devices.update(
            {nme: PowerSupply(nme) for nme in self.sofb_data.ch_names})
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['tunecorr'] = TuneCorr(TuneCorr.DEVICES.SI)
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.devices['sibpms'] = FamBPMs(FamBPMs.DEVICES.SI)
        self.devices['event'] = Event('Study')
        self.devices['trigger'] = Trigger('SI-Fam:TI-BPM')
        self.devices['evg'] = EVG()
        self.devices['rfgen'] = RFGen()

    def get_orbit_data(self):
        """."""
        prms = self.params
        tune = self.devices['tune']
        sibpms = self.devices['sibpms']
        sibpms.mturn_config_acquisition(
            nr_points_after=prms.nr_points_bpm_acq, nr_points_before=0,
            acq_rate='Monit1', repeat=False, external=True)

        sibpms.mturn_reset_flags()
        # self.devices['trigger'].source = 'Study'
        # self.devices['event'].mode = 'External'
        # self.devices['evg'].cmd_update_events()
        self.devices['event'].cmd_external_trigger()
        ret = sibpms.mturn_wait_update_flags(timeout=prms.bpms_timeout)
        if ret != 0:
            print(f'Problem waiting BPMs update. Error code: {ret:d}')
            return dict()

        orbx, orby = sibpms.get_mturn_orbit()
        chs_names = [self.sofb_data.ch_names[idx] for idx in prms.chs_idx]
        data = dict()
        data['timestamp'] = _time.time()
        data['chs_off'] = chs_names
        data['stored_current'] = self.devices['currinfo'].current
        data['orbx'], data['orby'] = orbx, orby
        data['tunex'], data['tuney'] = tune.tunex, tune.tuney
        data['mt_acq_rate'] = 'Monit1'
        data['rf_frequency'] = self.devices['rfgen'].frequency
        self.data = data

    def check_tunes(self):
        """."""
        tunecorr, tune = self.devices['tunecorr'], self.devices['tune']
        prms = self.params

        dnux0, dnuy0 = tunecorr.delta_tunex, tunecorr.delta_tuney
        dnux = self.tunex0 - tune.tunex
        dnuy = self.tuney0 - tune.tuney
        cond = abs(dnux) > prms.max_tunex_var
        cond |= abs(dnuy) > prms.max_tuney_var
        if cond:
            print('  Tune Correction...')
            print(
                f'    Initial Tunes x: {tune.tunex:.4f}, y: {tune.tuney:.4f}')
            print(
                f'    DeltaTunex: {dnux:.4f}, DeltaTuneY: {dnuy:.4f}')
            tunecorr.delta_tunex = dnux0 + dnux
            tunecorr.delta_tuney = dnuy0 + dnuy
            tunecorr.cmd_apply_delta()
            _time.sleep(prms.wait_tunecorr)
            print(
                f'    Final Tunes x: {tune.tunex:.4f}, y: {tune.tuney:.4f}')

    @staticmethod
    def apply_corr_ramp(self, corr, ramp):
        """."""
        sofb = self.devices['sofb']
        prms = self.params
        for kick in ramp:
            corr.kick = kick
            sofb.correct_orbit_manually(
                nr_iters=prms.nr_orbcorrs, residue=prms.min_orbres)
            self.check_tunes()

    def do_single_corr(self, corr_name):
        """."""
        sofb, prms = self.devices['sofb'], self.params
        corr_dev = self.devices[corr_name]
        names = self.sofb_data.ch_names
        enbllist0 = sofb.chenbl.copy()

        corr_idx = names.index(corr_name)
        corr_enbl = enbllist0.copy()
        corr_enbl[corr_idx] = 0
        sofb.chenbl = corr_enbl

        kick0 = corr_dev.kick
        kickstep = prms.max_kick_step if kick0 < 0 else -prms.max_kick_step
        rampdown = _np.r_[_np.arange(kick0, 0, kickstep)[1:], 0]
        self.apply_corr_ramp(corr_dev, rampdown)
        corr_dev.cmd_turn_off()
        corr_dev.cmd_turn_on()
        rampup = _np.r_[rampdown[::-1][1:], kick0]
        self.apply_corr_ramp(corr_dev, rampup)
        sofb.chenbl = enbllist0
        # return data

    def _calc_delta_orbit(self):
        chnames = self.sofb_data.ch_names
        nr_chs = len(chnames)
        app_kicks = self.devices['sofb'].kickch
        chs_idx = self.params.chs_idx
        kicks = _np.zeros(nr_chs)
        kicks[chs_idx] = app_kicks[chs_idx]
        delta_orbit = self.respmat[:, :nr_chs] @ (-kicks)
        return delta_orbit, kicks

    def _calc_reduction_factor(self, delta_orbit):
        factor = self.params.max_delta_orbit/_np.max(_np.abs(delta_orbit))
        return min(1, factor)

    def _select_chs(self):
        sofb, prms = self.devices['sofb'], self.params
        enbllist0 = sofb.chenbl
        corr_enbl = enbllist0.copy()
        corr_enbl[prms.chs_idx] = 0
        sofb.chenbl = corr_enbl
        return enbllist0, corr_enbl

    def _calc_corrs_compensation(self):
        delta_orbit, kicks = self._calc_delta_orbit()
        factor = self._calc_reduction_factor(delta_orbit)
        dkicks = -factor * kicks
        nr_chs = len(self.sofb_data.ch_names)

        irespmat = self.devices['sofb'].invrespmat
        dorbit = self.respmat[:, :nr_chs] @ dkicks
        dkicks_all = -(irespmat @ dorbit)[:nr_chs]
        dkicks_all[self.params.chs_idx] = dkicks[self.params.chs_idx]
        return dkicks_all, factor

    def _turn_off_corrs(self):
        names = self.sofb_data.ch_names
        print('Turning off CHs...')
        for idx in self.params.chs_idx:
            name = names[idx]
            print(f'  {name:s}')
            self.devices[name].cmd_turn_off()

    def _turn_on_corrs(self):
        names = self.sofb_data.ch_names
        print('Turning on CHs...')
        for idx in self.params.chs_idx:
            name = names[idx]
            print(f'  {name:s}')
            self.devices[name].cmd_turn_on()

    def _do_measure(self):
        tune, tunecorr = self.devices['tune'], self.devices['tunecorr']
        sofb, prms = self.devices['sofb'], self.params
        excx0, excy0 = tune.enablex, tune.enabley
        if not excx0:
            tune.cmd_enablex()
            _time.sleep(prms.wait_tunecorr)
        if not excy0:
            tune.cmd_enabley()
            _time.sleep(prms.wait_tunecorr)

        self.initial_kicks = sofb.kickch
        tunecorr.cmd_update_reference()
        self.tunex0, self.tuney0 = tune.tunex, tune.tuney
        _ = self._select_chs()
        factor = 0
        iter_idx = 1
        beam_dump = False
        while factor < 1 and not self._stopevt.is_set():
            if self.devices['currinfo'].current < self.MIN_CURRENT:
                beam_dump = True
                print('Beam dump, exiting...')
                break
            delta_kicks, factor = self._calc_corrs_compensation()
            print(f'Iter: {iter_idx:02d}, factor: {factor:.3f}')
            sofb.deltakickch = delta_kicks
            sofb.cmd_applycorr_ch()
            sofb.wait_apply_delta_kick()
            self.check_tunes()
            print('  Correcting the orbit...')
            sofb.correct_orbit_manually(
                nr_iters=prms.nr_orbcorrs, residue=prms.min_orbres)
            _time.sleep(prms.wait_iter)
            iter_idx += 1

        if not beam_dump and not self._stopevt.is_set():
            self._turn_off_corrs()
            print('  Correcting the orbit...')
            sofb.correct_orbit_manually(
                nr_iters=prms.nr_orbcorrs, residue=prms.min_orbres)
            print('Done!')
        elif self._stopevt.is_set():
            print('Stop was set, exiting...')

        if not excx0:
            tune.cmd_disablex()
        if not excy0:
            tune.cmd_disabley()

    def restore_initial_state(self):
        """."""
        tunecorr, sofb = self.devices['tunecorr'], self.devices['sofb']
        tunecorr.cmd_update_reference()

        self._turn_on_corrs()

        enbllist0 = sofb.chenbl
        sofb.chenbl = [1]*len(enbllist0)
        diff = _np.max(_np.abs(sofb.kickch - self.initial_kicks))
        while diff > 10:
            print(f'Max. Diff in CHs: {diff:.2f} [urad]')
            sofb.correct_orbit_manually(nr_iters=1)
            self.check_tunes()
            diff = _np.max(_np.abs(sofb.kickch - self.initial_kicks))
        print('Done!')
