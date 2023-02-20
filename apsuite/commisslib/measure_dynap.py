"""."""
import numpy as _np
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
import time as _time
import os as _os
from apsuite.commisslib.inj_traj_fitting import SIFitInjTraj
from apsuite.optics_analysis import TuneCorr
from siriuspy.epics import PV as _PV
from siriuspy.devices import PowerSupplyPU, Tune, CurrInfoSI, EVG, RFGen, \
      FamBPMs, Trigger, EGTriggerPS
from apsuite.utils import ParamsBaseClass as _ParamsBaseClass, \
     ThreadedMeasBaseClass as _ThreadBaseClass


class DynapParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.max_kickx = 0  # [mrad]
        self.max_kicky = 0  # [mrad]
        self.alphas = _np.linspace(0.8, 1.2, 11)
        self.thetas = _np.linspace(_np.pi/2, _np.pi, 5)
        self.dcct_offset = 0  # [mA]
        self.min_stored_curr = 2  # [mA]
        self.min_sum = 0
        self.nr_fits = None
        self.acq_nrsamples_pre = 10
        self.acq_nrsamples_post = 2000
        self.orbit_timeout = 20  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += dtmp('max_kickx', self.max_kickx, '[mrad]')
        stg += dtmp('max_kicky', self.max_kicky, '[mrad]')
        # stg += alphas = _np.linspace(0.8, 1.2, 11)
        # stg += thetas = _np.linspace(_np.pi/2, _np.pi, 5)
        stg += ftmp('dcct_offset', self.dcct_offset, '[mA]')
        stg += ftmp('min_stored_curr', self.min_stored_curr, '[mA]')
        stg += ftmp('min_sum', self.min_sum, '')
        if self.nr_fits is None:
            stg += stmp('nr_fits', 'not set'.rjust(9), '')
        else:
            stg += dtmp('nr_fits', self.nr_fits, '')
        stg += dtmp('acq_nrsamples_pre', self.acq_nrsamples_pre, '')
        stg += dtmp('acq_nrsamples_post', self.acq_nrsamples_post, '')
        stg += dtmp('orbit_timeout', self.orbit_timeout, '[s]')
        return stg


class MeasDynap(_ThreadBaseClass):
    """."""

    def __init__(self, params=None, isonline=True):
        """."""
        params = DynapParams() if params is None else params
        super().__init__(params=params, target=self._start_measurement,
                         isonline=isonline)
        self.fit_traj = SIFitInjTraj()
        self.tunecorr = TuneCorr(self.fit_traj.model, 'SI',
                                 method='Proportional', grouping='TwoKnobs')
        if self.isonline:
            self._create_devices()

    def _create_devices(self):
        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['pingv'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_PING_V)
        self.devices['fambpms'] = FamBPMs(FamBPMs.DEVICES.SI)
        self.devices['trigbpm'] = Trigger('SI-Fam:TI-BPM')
        # self.devices['toca'] = self.fit_traj.devices['sofb']
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['rfgen'] = RFGen()
        self.devices['evg'] = EVG()
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        # self.devices['exc_tunex'] = _PV('SI-Glob:DI-Tune-H:Enbl-Sts')
        # self.devices['exc_tuney'] = _PV('SI-Glob:DI-Tune-V:Enbl-Sts')
        self.devices['exc_tunex_sel'] = _PV('SI-Glob:DI-Tune-H:Enbl-Sel')
        self.devices['exc_tuney_sel'] = _PV('SI-Glob:DI-Tune-V:Enbl-Sel')
        self.devices['egun_trigps'] = EGTriggerPS()


    def correct_tunes(self, tunex_goal, tuney_goal):
        """."""
        print('--- correcting SI tunes ---')
        tunecorr = self.tunecorr
        model = self.fit_traj.model
        print('     initial tunes   : ',  tunecorr.get_tunes(model))
        tunemat = tunecorr.calc_jacobian_matrix()
        goal_tunes = _np.array([tunex_goal, tuney_goal])
        tunecorr.correct_parameters(model=model, goal_parameters=goal_tunes,
                                    jacobian_matrix=tunemat, tol=1e-8)
        print('     final tunes : ', tunecorr.get_tunes(model))

    def save_data(self, kickx, kicky, overwrite=False):
        """."""
        fname = self._get_filename(kickx, kicky, prefix='dynap')
        return super().save_data(fname, overwrite)

    def _get_filename(pingh, pingv, prefix=''):
        fname = 'tbt_' + prefix
        fname += f'_pingh_m{int(abs(pingh*1000)):03d}urad'
        fname += f'_pingh_p{int(abs(pingv*1000)):03d}urad'
        # fname += f"_drf_{'m' if dfreq<0 else 'p':s}{abs(dfreq):04.0f}hz"
        return fname

    def _start_measurement(self, dir_name=None):
        """."""
        if not self.isonline:
            raise ValueError('You need to be online to do the measurement')

        if dir_name is not None:
            try:
                _os.mkdir(dir_name)
                _os.chdir(dir_name)
            except FileExistsError('Choose a different directory name'):
                return

        max_kickx = self.params.max_kickx
        max_kicky = self.params.max_kicky
        alphas = self.params.alphas
        thetas = self.params.thetas

        pingh = self.devices['pingh']
        pingv = self.devices['pingv']
        currinfo = self.devices['currinfo']

        for theta in thetas:
            for alpha in alphas:
                curr0 = currinfo.current - self.params.dcct_offset

                if self._stopevt.is_set():
                    print('Stopping...')
                    return

                kickx = alpha * max_kickx * _np.cos(theta)
                kicky = alpha * max_kicky * _np.sin(theta)
                pingh.strength = kickx
                pingv.strength = kicky

                self._check_current_and_inject()

                stg = f'Measuring: kick x = {kickx*1000:3d} urad, '
                stg += f'kick y = {kicky*1000:3d} urad'
                print(stg)

                self.acquire_data()
                self.save_data(kickx=kickx, kicky=kicky)

                currf = currinfo.current - self.params.dcct_offset
                loss = (1 - currf/curr0) * 100
                print(f'    Beam loss: {loss:.2f}%')
        print('Measurement complete!')
        
        return None

    def _check_current_and_inject(self, min_stored_current):
            evg = self.devices['evg']
            currinfo = self.devices['currinfo']
            
            self.devices['egun_trigps'].cmd_enable_trigger()
            _time.sleep(1.0)
            curr = currinfo.current
            while curr < min_stored_current:
                print('Injecting')
                evg.cmd_turn_on_injection(wait_rb=True)
                evg.wait_injection_finish()
                curr = currinfo.current
            self.devices['egun_trigps'].cmd_disable_trigger()
            _time.sleep(1.0)
            return

    def acquire_data(self):
        """."""
        fambpms = self.devices['fambpms']
        evt_study = self.devices['evt_study']  # how does this change when using Linac evt ?

        # turn off tune excitator ?
        self.devices['exc_tunex_sel'].value = 0
        self.devices['exc_tuney_sel'].value = 0
        _time.sleep(2)

        self._prepare_bpms_acquisition()
        self._prepare_pulsed_magnets()
        fambpms.mturn_reset_flags()
        evt_study.cmd_external_trigger()  # how does this change when using Linac evt ?
        # to fire the LINAC event should I just fire  evg.cmd_turn_on_injection(wait_rb=True)?
        # the egun trigger and pulsed mags other than pingers would be down 
        fambpms.mturn_wait_update_flags(timeout=self.params.orbit_timeout)
        trajx, trajy, trajsum = fambpms.get_mturn_orbit(return_sum=True)

        data = dict()
        data['timestamp'] = _time.time()
        data['rf_frequency'] = self.devices['rfgen'].frequency
        data['stored_current'] = self.devices['currinfo'].current
        data['trajx'], data['trajy'], data['trajsum'] = trajx, trajy, trajsum
        data['pingh_kick'] = self.devices['pingh'].strength
        data['pingv_kick'] = self.devices['pingv'].strength
        data['pingh_voltage'] = self.devices['pingh'].voltage
        data['pingv_voltage'] = self.devices['pingv'].voltage
        data['pingh_pulse_sts'] = self.devices['pingh'].pulse
        data['pingv_pulse_sts'] = self.devices['pingv'].pulse
        tune = self.devices['tune']
        data['tunex'], data['tuney'] = tune.tunex, tune.tuney
        bpm0 = self.devices['fambpms'].devices[0]
        csbpm = bpm0.csdata
        data['bpms_acq_rate'] = csbpm.AcqChan._fields[bpm0.acq_channel]
        data['bpms_nrsamples_pre'] = bpm0.acq_nrsamples_pre
        data['bpms_nrsamples_post'] = bpm0.acq_nrsamples_post
        data['bpms_trig_delay_raw'] = self.devices['trigbpm'].delay_raw
        data['bpms_switching_mode'] = csbpm.SwModes._fields[
            bpm0.switching_mode]
        data['tunex_enable'] = tune.enablex
        data['tuney_enable'] = tune.enabley

        # turn on tune excitator ?
        self.devices['exc_tunex_sel'].value = 1
        self.devices['exc_tuney_sel'].value = 1
        self.data = data

    def _prepare_pulsed_magnets(self):
        # turn off NLK, septa
        raise NotImplementedError

    def _prepare_timing(self):
        """."""
        trigbpm = self.devices['trigbpm']
        trigbpm.delay = 0.0
        trigbpm.nr_pulses = 1
        trigbpm.source = 'Linac'  # ?
        self.devices['evg'].cmd_update_events()   # needed?

    def _prepare_bpms_acquisition(self):
        """."""
        fambpms = self.devices['fambpms']
        prms = self.params
        fambpms.mturn_config_acquisition(
            nr_points_after=prms.acq_nrsamples_pre,
            nr_points_before=prms.acq_nrsamples_post,
            acq_rate='TbT',
            repeat=False)

    def plot_traj_fit(self, turn_idx=-1):
        """."""
        trajx = self.data['trajx'].reshape(-1, 160)[turn_idx]
        trajy = self.data['trajy'].reshape(-1, 160)[turn_idx]
        trajsum = self.data['trajsum'].reshape(-1, 160)[turn_idx]

        trajx *= 1e-6  # m -> um
        trajy *= 1e-6

        vecs = self.fit_traj.do_fitting(trajx, trajy, tol=1e-8, max_iter=20)
        rx, px, ry, py, de = vecs[-1]
        
        tmpl = '{:10s} ' + '{:^10.2f} '*5
        ttmpl = '{:10s} ' + '{:^10s} '*5
        print(ttmpl.format('', 'x [mm]', 'xl [mrad]', 'y [mm]',
                           'yl [mrad]', 'de [%]'))
        print(tmpl.format('Fit', rx*1e3, px*1e3, ry*1e3, py*1e3, de*1e2))

        fig = _mplt.figure(figsize=(9, 10))
        gs = _mgs.GridSpec(3, 1)
        gs.update(left=0.12, right=0.98, top=0.97, bottom=0.08, hspace=0.25)
        ax = _mplt.subplot(gs[0, 0])
        ay = _mplt.subplot(gs[1, 0])
        asum = _mplt.subplot(gs[2, 0])

        bpmpos = self.fit_traj.twiss.spos[self.fit_traj.bpm_idx]

        ax.plot(
            bpmpos[:trajx.size], 1e3*trajx, '-d', label='trajectory')
        ay.plot(
            bpmpos[:trajy.size], 1e3*trajy, '-d', label='trajectory')
        asum.plot(bpmpos[:trajsum.size], trajsum)

        trajx_fit, trajy_fit = self.fit_traj.calc_traj(
            *vecs[-1], size=trajx.size)
        ax.plot(bpmpos[:trajx_fit.size], 1e3*trajx_fit, '-o', label='fitting',
                linewidth=1)
        ay.plot(bpmpos[:trajy_fit.size], 1e3*trajy_fit, '-o', label='fitting',
                linewidth=1)

        title = r"$x$ = {:.3f}mm $x$' = {:.3f}mrad".format(rx*1e3, px*1e3)
        title += r"$\delta$ = {:.2f}%".format(de*1e2)
        ax.set_title(title)
        title = r"$y$ = {:.3f}mm $y$' = {:.3f}mrad".format(ry*1e3, py*1e3)
        ay.set_title(title)
        ay.legend()
        asum.set_xlabel('position [m]')
        ax.set_ylabel(r'$x$ [mm]')
        ay.set_ylabel(r'$y$ [mm]')
        asum.set_ylabel('sum signal [counts]')
        fig.tight_layout()
        fig.show()
        # fix units!
        return fig, ax, ay, asum

    def process_dynap_data(self, fnames=None, files_dir_path=None):
        """."""
        if files_dir_path is not None:
            _os.chdir(files_dir_path)
        if fnames is None:
            fnames = _os.listdir()
        fnames = sorted([val for val in fnames if '.pickle' in val])

        kicksx, kicksy = [], []
        xmin, ymax = [], []
        kxmin, kymax = [], []
        losses, residues = [], []

        for fname in fnames:
            print(f'Loading file {fname}...')
            data = self.load_data(fname)
            trajsum = data['trajsum'].reshape(-1, 160)
            trajsum_avg = _np.mean(trajsum, axis=1)
            # rehsapes really necessary?
            trajx = data['trajx'].reshape(-1, 160) * 1e-6  # [um]
            trajy = data['trajy'].reshape(-1, 160) * 1e-6
            loss = (1 - trajsum_avg.min()/trajsum_avg.max()) * 100
            loss = min(max(loss, 0), 100)
            kickx = data['pingh_kick'] * 1e3  # [mrad]
            kicky = data['pingv_kick'] * 1e3
            print(f'Loaded!\n')

            min_sum = self.params.min_sum
            nr_fits = self.params.nr_fits
            if nr_fits is None:
                nr_fits = (trajsum_avg < min_sum).nonzero()[0]
            if not nr_fits.size:
                nr_fits = trajsum_avg.size
            print(f'Number of fits = {nr_fits}.\n')

            vecs = []
            for i in range(nr_fits):
                print(f'    fitting turn {i:3d}/{nr_fits:3d}...')
                trajx_i, trajy_i = trajx[i], trajy[i]
                sum_i = trajsum[i]
                ini_sum_avg = _np.mean(sum_i[:3]) / 3
                max_idx = _np.sum(sum_i > ini_sum_avg)
                trajx_i = trajx_i[:max_idx]  # ?
                trajy_i = trajy_i[:max_idx]
                fits, res, chis = self.fit_traj.do_fitting(
                    trajx_i, trajy_i, tol=1e-8, max_iter=5, full=True,
                    update_jacobian=True)
                print(f'        fitted! Chi = {chis[-1]*1000:1.2f}\n')  # ?
                vec = fits[-1] * _np.nan if chis[-1] >= 2 else fits[-1]
                vecs.append(vec)
            vecs = _np.array(vecs)

            if not vecs.size:
                continue
            else:
                if not _np.isnan(vecs).any():  # ?
                    xmin.append(vecs[:, 0].min())
                    kxmin.append(vecs[:, 1].min())  # ?
                    ymax.append(vecs[:, 2].max())
                    kymax.append(vecs[:, 3].max())
                    kicksx.append(kickx)
                    kicksy.append(kicky)
                    losses.append(loss)
                    residues.append(res)

        kicksx = _np.array(kicksx, dtype=int)
        kicksy = _np.array(kicksy, dtype=int)
        xmin = _np.array(xmin)
        ymax = _np.array(ymax)
        losses = _np.array(losses)
        kxmin = _np.array(kxmin)
        kymax = _np.array(kymax)

        idx_nan = (xmin < -9.5e-3) | (ymax > 3.5e-3)  # ?
        xmin[idx_nan], ymax[idx_nan] = _np.nan, _np.nan
        kxmin[idx_nan], kymax[idx_nan] = _np.nan, _np.nan
        losses[idx_nan] = _np.nan

        self.data['dynap_data'] = {
            'kicksx': kicksx,
            'kicksy': kicksy,
            'xmin': xmin,
            'ymax': ymax,
            'losses': losses,
            'kxmin': kxmin,
            'kymax': kymax
            }

    def plot_dynap(vec1, vec2, losses, labelx='', labely='', loss_tol=5,
                   figname=None):
        """."""
        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)

        ax1 = _mplt.subplot(gs[0, 0])
        pcm = ax1.scatter(
            vec1, vec2, c=losses, cmap='plasma', vmin=0, vmax=100)
        idx_loss = losses > loss_tol

        vertx = _np.sort(vec1[idx_loss])
        verty = _np.sort(vec2[idx_loss])
        verty, idxuni = _np.unique(verty, return_index=True)  # ?
        vertx = vertx[idxuni]

        ax1.plot(vertx, verty, color='tab:gray', ls='-', lw=4,
                 label=f'Beam loss $>$ {loss_tol:01d}%')
        stg = 'Measured Dynamic Aperture @ 01SA'
        stg += '\n'
        stg += r'${x_{min}}$ = '
        stg += f'{_np.nanmin(vertx):.1f} mm, '
        stg += r'${y_{max}}$ = '
        stg += f'{_np.nanmax(verty):.1f} mm'
        ax1.set_xlabel(labelx)
        ax1.set_ylabel(labely)

        cbar = fig.colorbar(pcm, ax=ax1)
        cbar.set_label('Beam loss [%]')
        ax1.set_title(stg)
        ax1.legend()
        _mplt.grid(True, alpha=0.5, ls='--', color='tab:gray')
        _mplt.tight_layout(True)
        if figname is not None:
            _mplt.savefig(f'{figname}.png', dpi=600, format='png')
        else:
            print('No figname. Not saving figure.')
        fig.show()
        return fig, ax1
