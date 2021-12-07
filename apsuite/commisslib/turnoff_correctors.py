"""."""
import time as _time
import numpy as _np
from siriuspy.devices import PowerSupply, CurrInfoSI, Tune, TuneCorr, SOFB
from siriuspy.sofb.csdev import SOFBFactory
from siriuspy.clientconfigdb import ConfigDBClient as _ConfigDBClient

from ..utils import ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class TurnOffCorrParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.max_kick_step = 30  # [urad]
        self.min_orbres = 5  # [um]
        self.max_tunex_var = 0.005
        self.max_tuney_var = 0.005
        self.chs_idx = []
        self.nr_orbcorrs = 5
        self.wait_tunecorr = 3  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += ftmp('max_kick_step', self.max_kick_step, '')
        stg += ftmp('min_orbres', self.min_orbres, '')
        stg += ftmp('max_tunex_var', self.max_tunex_var, '')
        stg += ftmp('max_tuney_var', self.max_tuney_var, '')
        stg += stmp('chs_idx', str(self.chs_idx), '')
        stg += dtmp('nr_orbcorrs', self.nr_orbcorrs, '')
        stg += ftmp('wait_tunecorr', self.wait_tunecorr, '[s]')
        return stg


class TurnOffCorr(_ThreadBaseClass):
    """."""

    def __init__(self, params=None, isonline=True):
        """."""
        params = TurnOffCorrParams() if params is None else params
        super().__init__(
            params=params, target=self._do_measure, isonline=isonline)
        self.sofb_data = SOFBFactory.create('SI')
        # self.chs_subset = self.sofb_data.ch_names[self.params.chs_idx]
        client = _ConfigDBClient(config_type='si_orbcorr_respm')
        respmat = _np.array(client.get_config_value(name='ref_respmat'))
        respmat = _np.reshape(respmat, (320, 281))
        self.respmat = respmat
        self.tunex0, self.tuney0 = None, None
        if self.isonline:
            self._create_devices()

    def _create_devices(self):
        self.devices.update({
            nme: PowerSupply(nme) for nme in self.sofb_data.ch_names})
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['tunecorr'] = TuneCorr(TuneCorr.DEVICES.SI)
        self.devices['tunecorr'].cmd_update_reference()
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)

    def get_orbit_data(self):
        """."""
        return NotImplementedError

    def check_tunes(self):
        """."""
        tunecorr, tune = self.devices['tunecorr'], self.devices['tune']
        sofb = self.devices['sofb']
        prms = self.params

        dnux0, dnuy0 = tunecorr.delta_tunex, tunecorr.delta_tuney
        dnux = self.tunex0 - tune.tunex
        dnuy = self.tuney0 - tune.tuney
        cond = abs(dnux) > prms.max_tunex_var
        cond |= abs(dnuy) > prms.max_tuney_var
        if cond:
            print('Tune Correction...')
            print(f'DeltaTunex: {dnux0:.4f}, DeltaTuneY: {dnuy0:.4f}')
            tunecorr.delta_tunex = dnux0 + dnux
            tunecorr.delta_tuney = dnuy0 + dnuy
            tunecorr.cmd_apply_delta()
            _time.sleep(prms.wait_tune_corr)
            sofb.correct_orbit_manually(
                        nr_iters=prms.nr_orbcorrs, res=prms.min_orbres)

    @staticmethod
    def apply_corr_ramp(self, corr, ramp):
        """."""
        sofb = self.devices['sofb']
        prms = self.params
        for kick in ramp:
            corr.kick = kick
            sofb.correct_orbit_manually(
                nr_iters=prms.nr_orbcorrs, res=prms.min_orbres)
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
        # To turn on and off the PSSOFB mode must be disabled
        corr_dev.cmd_turn_off()
        # Measure Orbit
        # data = self.get_orbit_data()
        corr_dev.cmd_turn_on()
        rampup = _np.r_[rampdown[::-1][1:], kick0]
        self.apply_corr_ramp(corr_dev, rampup)
        sofb.chenbl = enbllist0
        # return data

    def calc_delta_orbit(self):
        """."""
        chnames = self.sofb_data.ch_names
        nr_chs = len(chnames)
        kicks = _np.zeros(nr_chs)
        for idx in self.params.chs_idx:
            name = chnames[idx]
            # Do not get kicks from device, get and set from SOFB
            # PSSOFB will be on, timing event must be set
            kicks = self.devices[name].kick
        # delta_kicks = - kicks
        delta_orbit = self.respmat[:, :nr_chs] @ (-kicks)
        return delta_orbit, kicks

    def _calc_reduction_factor(self, delta_orbit):
        return self.params.max_delta_orbit/_np.max(abs(delta_orbit))

    def _calc_corrs_compensation(self):
        delta_orbit, kicks = self.calc_delta_orbit()
        factor = self._calc_reduction_factor(self, delta_orbit)
        if factor < 1:
            dkicks = (1 - factor) * kicks
        else:
            dkicks = -kicks
        nr_chs = len(self.sofb_data.ch_names)
        irespmat = self.sofb.invrespmat
        dorbit = self.respmat[:, :nr_chs] @ dkicks
        dkicks_comp = -(irespmat @ dorbit)[:nr_chs-len(self.params.chs_idx)]

        dkicks_all = _np.zeros(nr_chs)
        dkicks_all[self.params.chs_idx] = dkicks
        comp = list(set(range(nr_chs)) - set(self.params.ch_idx))
        dkicks_all[comp] = dkicks_comp
        return dkicks_all

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

        self.tunex0, self.tuney0 = tune.tunex, tune.tuney

        for chname in self.chs_subset:
            print(chname)
            # this method will return the data set
            self.do_single_corr(chname)

        if not excx0:
            tune.cmd_disablex()
        if not excy0:
            tune.cmd_disabley()

        print('Restoring Quadrupoles Configuration...')
        tunecorr.delta_tunex = 0
        tunecorr.delta_tuney = 0
        tunecorr.cmd_apply_delta()
        _time.sleep(prms.wait_tune_corr)
        sofb.correct_orbit_manually(
                    nr_iters=prms.nr_orbcorrs, res=prms.min_orbres)
