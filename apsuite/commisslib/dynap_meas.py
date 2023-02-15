import numpy as _np
import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
import time as _time
import os as _os
from apsuite.commisslib.inj_traj_fitting import SIFitInjTraj
from apsuite.optics_analysis import TuneCorr
from siriuspy.epics import PV as _PV
from siriuspy.devices import PowerSupplyPU, Tune, CurrInfoSI, EVG, RFGen, \
      BunchbyBunch
from apsuite.utils import ParamsBaseClass as _ParamsBaseClass, \
     ThreadedMeasBaseClass as _ThreadBaseClass


class DynapParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        super().__init__()
        self.max_kickx = 770 * 1e-3
        self.max_kicky = 320 * 1e-3
        self.alphas = _np.linspace(0.8, 1.2, 11)
        self.thetas = _np.linspace(_np.pi/2, _np.pi, 5)
        self.dcct_offset = 0
        self.tunex = 0
        self.tuney = 0
        self.dfreq = 0
        self.min_stored_curr = 2 #  [mA]
        self.min_sum = None
        self.nr_fits = None

    def __str__(self) -> str:
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''

        return stg


class MeasDynap(_ThreadBaseClass):
    """."""

    def __init__(self, params=None, target=None, isonline=True):
        self.params = DynapParams()
        if self.isonline:
            self._create_devices()
        self.fit_traj = SIFitInjTraj()
        self.tunecorr = TuneCorr(self.fit_traj.model, 'SI',
                                 method='Proportional', grouping='TwoKnobs')

    def _create_devices(self):
        self.devices['pingh'] = PowerSupplyPU(
            PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
        self.devices['pingv'] = PowerSupplyPU(PowerSupplyPU.DEVICES.SI_PING_V)
        self.devices['toca'] = self.fit_traj.devices['sofb']
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['rfgen'] = RFGen()
        self.devices['evg'] = EVG()
        self.devices['bbbl'] = BunchbyBunch(BunchbyBunch.DEVICES.L)
        self.devices['tune'] = Tune(Tune.DEVICES.SI)

    def correct_tunes(self, tunex_goal, tuney_goal):
        """."""
        print('--- correcting si tunes ---')
        tunecorr = self.tunecorr
        model = self.fit_traj.model
        print('     initial tunes   : ',  tunecorr.get_tunes(model))
        tunemat = tunecorr.calc_jacobian_matrix()
        goal_tunes = _np.array([tunex_goal, tuney_goal])
        tunecorr.correct_parameters(model=model, goal_parameters=goal_tunes,
                                    jacobian_matrix=tunemat, tol=1e-8)
        print('     final tunes : ', tunecorr.get_tunes(model))

    def save_data(self, overwrite=False, tunex=None, tuney=None,
                  kickx=None, kicky=None):
        """."""
        self.data['timestamp'] = _time.time()
        self.data['pingh_kick'] = self.devices['pingh'].strength
        self.data['pingv_kick'] = self.devices['pingv'].strength
        self.data['pingh_voltage'] = self.devices['pingh'].voltage
        self.data['pingv_voltage'] = self.devices['pingv'].voltage
        self.data['pingh_pulse_sts'] = self.devices['pingh'].pulse
        self.data['pingv_pulse_sts'] = self.devices['pingv'].pulse
        if tunex is not None:
            self.data['tunex'] = tunex
        if tuney is not None:
            self.data['tuney'] = tuney
        if tunex is not None:
            self.data['pingh_kick_sp'] = kickx
        if tuney is not None:
            self.data['pingv_kick_sp'] = kicky
        bbl = self.devices['bbl']
        self.data['tunes'] = bbl.sram.spec_marker1_freq
        self.data['tunes'] /= bbl.info.rf_freq_nom * bbl.info.harmonic_number
        self.data['trajx'] = self.devices['toca'].mt_trajx.reshape(-1, 160)
        self.data['trajy'] = self.devices['toca'].mt_trajy.reshape(-1, 160)
        self.data['trajsum'] = self.devices['toca'].mt_sum.reshape(-1, 160)
        self.data['toca_buffer_count'] = self.devices['toca'].buffer_count
        self.data['toca_nr_samplespre'] = self.devices['toca'].trigsamplepre
        self.data['toca_nr_samplespost'] = self.devices['toca'].trigsamplepost
        self.data['rf_frequency'] = self.devices['rfgen'].frequency
        self.data['stored_current'] = self.devices['currinfo'].current

        fname = self._get_filename(kickx, kicky, prefix='ad')

        return super().save_data(fname, overwrite)

    def _get_filename(pingh, pingv, prefix=''):
        fname = 'tbt_' + prefix
        fname += f'_pingh_m{int(abs(pingh*1000)):03d}urad' 
        fname += f'_pingh_p{int(abs(pingv*1000)):03d}urad'
        # fname += f"_drf_{'m' if dfreq<0 else 'p':s}{abs(dfreq):04.0f}hz"
        return fname

    def do_measurement(self, dir_name=None):
        """."""
        if not (dir_name is None):
            try:
                _os.mkdir(dir_name)
                _os.chdir(dir_name)
            except  FileExistsError:
                print('Choose a different directory name')

        max_kickx = self.params.max_kickx
        max_kicky = self.params.max_kicky
        alphas = self.params.alphas
        thetas = self.params.thetas

        pingh = self.devices['pingh']
        pingv = self.devices['pingv']
        tunex = self.devices['tune'].tunex
        tuney = self.devices['tune'].tuney
        currinfo = self.devices['currinfo']

        self._check_current_and_inject()
        curr0 = currinfo.current - self.params.dcct_offset

        for theta in thetas:
            for alpha in alphas:
                kickx = alpha * max_kickx * _np.cos(theta)
                kicky = alpha * max_kicky * _np.sin(theta)

                pingh.strength = kickx
                pingv.strength = kicky

                stg = f'Measuring: kick x = {kickx*1000:3d} urad, '
                stg += f'kick y = {kicky*1000:3d} urad'
                print(stg)

                self.save_data(tunex=tunex, tuney=tuney, kickx=kickx,
                               kicky=kicky)

                currf = currinfo.current - self.params.dcct_offset
                loss = (currf-curr0)/curr0 * 100
                print(f'    Beam loss: {loss:.2f}%')

    def load_and_apply(self, fname):
        """."""
        return super().load_and_apply(fname)

    def plot_traj_fit(self):
        """."""
        raise NotImplementedError

    def _check_current_and_inject(self, min_stored_current):
        evg = self.devices['evg']
        currinfo = self.devices['currinfo']

        curr = currinfo.current
        while curr < min_stored_current:
            evg.cmd_turn_on_injection(wait_rb=True)
            evg.wait_injection_finish()
            curr = currinfo.current
        return

    def process_dynap_data(self, fnames=None, files_dir_path=None):
        """."""
        if not (files_dir_path is None):
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
            kickx = data['pingh_kick'] * 1e3 #  [mrad]
            kicky = data['pingv_kick'] * 1e3
            print(f' loaded!\n')
            
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
                print(f'fitted! Chi = {chis[-1]*1000:1.2f}\n')  # ?
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
