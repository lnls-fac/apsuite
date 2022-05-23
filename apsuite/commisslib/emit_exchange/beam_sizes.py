"""Classes to measure beam sizes at TS line screens"""

import numpy as _np
import matplotlib.pyplot as _plt
import matplotlib.patches as _patches
import time as _time

from scipy.optimize import curve_fit
# from siriuspy.ramp.ramp import BoosterRamp
# from siriuspy.ramp.conn import ConnPS
from siriuspy.devices import EVG, Screen, CurrInfoBO
from siriuspy.epics import PV

from ...utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class BeamSizesParams(_ParamsBaseClass):
    """."""
    def __init__(self):
        """."""
        self.measures_per_point = 3
        self.nr_points = 15
        self.ts_screen_number = 1
        self.ramp_config = 'bo_ramp_flop_emit_exchange_slower'
        self.init_delay = -3  # [ms]
        self.final_delay = 3  # [ms]
        self.roix = [500, 800]
        self.roiy = [400, 600]
        self.line_window = 4

    def __str__(self):
        """."""
        ftmp = '{0:26s} = {1:9.6f}  {2:s}\n'.format
        dtmp = '{0:26s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:15s} = {1:9}  {2:s}\n'.format
        stg = ''
        stg += dtmp('measures_per_point', self.measures_per_point, '')
        stg += dtmp('nr_points', self.nr_points, '')
        stg += dtmp('ts_screen_number', self.ts_screen_number, '')
        stg += stmp('ramp_config', self.ramp_config, '')
        stg += ftmp('init_delay', self.init_delay, '[ms]')
        stg += ftmp('final_delay', self.final_delay, '[ms]')
        stg += stmp('roix', str(self.roix), '[pixels]')
        stg += stmp('roiy', str(self.roiy), '[pixels]')
        stg += dtmp('line_window', self.line_window, '[pixels]')

        return stg


class BeamSizesAnalysis(_BaseClass):
    """."""
    def __init__(self, params=None, isonline=True):
        """."""
        params = BeamSizesParams() if params is None else params
        super().__init__(params=params, isonline=isonline)
        if self.isonline:
            self.create_pvs()
            self.create_devices()

    def create_pvs(self):
        """."""
        self.pvs['bo-qf-wfm'] = PV('BO-Fam:PS-QF:Wfm-RB')
        self.pvs['bo-qd-wfm'] = PV('BO-Fam:PS-QD:Wfm-RB')

    def create_devices(self):
        """."""
        self.devices['evg'] = EVG()
        self.devices['bo_ramp'] = BoosterRamp(self.params.ramp_config)
        self.devices['bo_ramp'].load()
        self.devices['curr_info'] = CurrInfoBO()
        self.devices['conn_ps'] = ConnPS()
        if not self.devices['conn_ps'].connected:
            raise AssertionError(
                "Can't stablish connection with power sources")

        scrns = []
        for i in range(1, 7):
            if i <= 3:
                scrns.append(f'TS-0{i}:DI-Scrn')
            else:
                scrns.append(f'TS-04:DI-Scrn-{i-3}')
        scrn_idx = self.params.ts_screen_number-1
        self.devices['ts_screen'] = Screen(scrns[scrn_idx])

    def reset_ramp_config(self):
        """."""
        old_boramp = BoosterRamp(self.params.ramp_config)
        old_boramp.load()
        self._send_ramp_wfm(boramp=old_boramp)

    def scan_emittance_exchange(self):
        """Method to measure the beam sizes while moving the emittance exchange
        ramp 'peak'. The results are useful to analyze the emittance exchange
        process."""
        boramp = self.devices['bo_ramp']
        evg = self.devices['evg']
        scrn = self.devices['ts_screen']
        currinfo = self.devices['curr_info']
        pvs = self.pvs
        prms = self.params
        data = self._create_dict_with_lists()

        delays = _np.linspace(
            prms.init_delay, prms.final_delay, prms.nr_points)

        config_times = _np.sort(boramp.ps_normalized_configs_times)
        emit_exc_points = config_times[-4:-1]
        init_old, mid_old, end_old = emit_exc_points

        for delay in delays:
            # Moves the emit exchange points
            initn, midn, endn = emit_exc_points + delay
            boramp.ps_normalized_configs_change_time(init_old, initn)
            boramp.ps_normalized_configs_change_time(mid_old, midn)
            boramp.ps_normalized_configs_change_time(end_old, endn)
            init_old, mid_old, end_old = initn, midn, endn

            # Send the ramp config to the machine
            self._send_ramp_wfm()
            _time.sleep(5)

            for i in range(prms.measures_per_point):
                print(f"Delay = {delay}, measure {i}.")
                evg.cmd_turn_on_injection()
                data['measure'].append(i)
                data['delays'].append(delay)
                data['timestamp'].append(_time.time())
                data['qf_wfm'].append(pvs['bo-qf-wfm'].get())
                data['qd_wfm'].append(pvs['bo-qd-wfm'].get())
                _time.sleep(1)
                image = scrn.image
                data['images'].append(image)
                data['curr3gev'].append(currinfo.current3gev)

        data['scl_factx'] = _np.abs(scrn.scale_factor_x)
        data['scl_facty'] = _np.abs(scrn.scale_factor_x)

        self.data = data
        self.reset_ramp_config()

    def process_data(self, plot_flag=False):
        """."""
        data = self.data
        prms = self.params
        sigmasx = []
        sigmasy = []

        sclx, scly = data['scl_factx'], data['scl_facty']  # pixel to mm

        for image in data['images']:
            sx, _, sy, _ = self.calc_beam_sizes(
                image, roix=prms.roix, roiy=prms.roiy, window=prms.line_window,
                plot_flag=plot_flag)
            sigmasx.append(sx*sclx)
            sigmasy.append(sy*scly)

        data['sigmasx'] = sigmasx
        data['sigmasy'] = sigmasy

    def plot_time_scan(self, figname=None, legend=True, axis=None):
        """."""
        prms = self.params
        data = self.data

        if (data['sigmasx'] == []) or (data['sigmasy'] == []):
            self.process_data()

        sx, sy = data['sigmasx'], data['sigmasy']
        rsx = _np.array(sx).reshape(prms.measures_per_point, -1)
        rsy = _np.array(sy).reshape(prms.measures_per_point, -1)
        mean_sx, stdx = rsx.mean(axis=1), rsx.std(axis=1)
        mean_sy, stdy = rsy.mean(axis=1), rsy.std(axis=1)

        delays = data['delays']
        u_delays = _np.unique(delays)

        if axis is None:
            _, ax = _plt.subplots()
        else:
            ax = axis

        ax.scatter(
            delays, sx, c='C0', s=1.5,
            label=r'measured $\sigma_x$', marker='v')
        ax.scatter(
            delays, sy, c='tab:orange', s=1.5,
            label=r'measured $\sigma_y$', marker='^')
        ax.errorbar(
            u_delays, mean_sx, yerr=stdx, ls='-', marker='', c='tab:blue',
            label=r'$<\sigma_x>$')
        ax.errorbar(
            u_delays, mean_sy, yerr=stdy, ls='-', marker='', c='tab:orange',
            label=r'$<\sigma_y>$')
        ax.set_xlabel(r'$\Delta t_{ext}$ [ms]')
        ax.set_ylabel(r'$\sigma_{x,y}$ [mm]')
        if legend:
            # ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
            ax.legend(loc='best', fontsize=8, frameon=True)
        if figname is not None:
            _plt.savefig(figname)
            _plt.show()

    @staticmethod
    def calc_beam_sizes(
            image, roix=[500, 800], roiy=[400, 600], window=4,
            plot_flag=False):
        """Compute the beam sizes from an image.

        Parameters
        ----------
        image : np.array
            Image of the beam.
        roix : list, optional
            Horizontal region of interest, by default [500, 800]
        roiy : list, optional
            Vertical region of interest, by default [400, 600]
        window : int, optional
            Width of the lines that will be taken to generate the gaussian
            distribution where the sizes are computed, by default 4
        plot_flag : bool, optional
            Plot fit statistics, by default False

        Returns
        -------
        sigmax: float
            Horizontal beam size.
        usigmax: float
            sigmax fit uncertainty.

        sigmay: float
            Vertical beam size.
        usigmay: float
            sigmay fit uncertainty.
        """
        gauss = BeamSizesAnalysis.gauss

        roix1, roix2 = roix
        roiy1, roiy2 = roiy
        roix = slice(roix1, roix2)
        roiy = slice(roiy1, roiy2)
        xx = _np.arange(image.shape[1])[roix]
        xy = _np.arange(image.shape[0])[roiy]

        image = image[roiy, roix]
        projx = _np.sum(image, axis=0)
        projx = _np.array(projx)/_np.sum(projx)
        projy = _np.sum(image, axis=1)
        projy = _np.array(projy)/_np.sum(projy)

        xx_mean = _np.sum(xx*projx)
        xy_mean = _np.sum(xy*projy)
        xx_std = _np.sqrt(_np.sum((xx-xx_mean)**2*projx))
        xy_std = _np.sqrt(_np.sum((xy-xy_mean)**2*projy))

        p0x = [_np.max(projx), xx_mean, xx_std, 0]
        p0y = [_np.max(projy), xy_mean, xy_std, 0]
        poptx_proj, _ = curve_fit(gauss, xx, projx, p0=p0x)
        popty_proj, _ = curve_fit(gauss, xy, projy, p0=p0y)

        x_mean_idx = _np.argmin(_np.abs(xx-poptx_proj[1]))
        y_mean_idx = _np.argmin(_np.abs(xy-popty_proj[1]))

        hsection = image[y_mean_idx-int(window/2):y_mean_idx+int(window/2), :]
        vsection = image[:, x_mean_idx-int(window/2):x_mean_idx+int(window/2)]
        hline = _np.sum(hsection, axis=0)
        hline = _np.array(hline)/_np.sum(hline)
        vline = _np.sum(vsection, axis=1)
        vline = _np.array(vline)/_np.sum(vline)

        poptx, pcovx = curve_fit(gauss, xx, hline, p0=poptx_proj)
        popty, pcovy = curve_fit(gauss, xy, vline, p0=popty_proj)
        sigmax, sigmay = abs(poptx[2]), abs(popty[2])
        usigmax, usigmay = _np.sqrt(pcovx[2, 2]), _np.sqrt(pcovy[2, 2])

        if plot_flag:
            fig = _plt.figure(figsize=(9, 12))
            gs = _plt.GridSpec(2, 2)
            aimg = _plt.subplot(gs[0, :])
            ax = _plt.subplot(gs[1, 0])
            ay = _plt.subplot(gs[1, 1])

            aimg.imshow(image)
            aimg.plot(xx_mean, xy_mean, 'o', ms=5, color='tab:red')
            w, h = _np.abs(roix2-roix1), _np.abs(roiy2-roiy1)
            rect = _patches.Rectangle(
                (roix1, roiy1), w, h, linewidth=1, edgecolor='k', fill='False',
                facecolor='none')
            aimg.add_patch(rect)

            ax.plot(xx, projx, '.', label='data')
            ax.plot(xx, gauss(xx, *poptx_proj), label='proj')
            ax.plot(xx, hline/_np.sum(hline), '.', label='line')
            ax.plot(xx, gauss(xx, *poptx)/_np.sum(hline), label='slice')

            ay.plot(xy, projy, '.', label='data')
            ay.plot(xy, gauss(xy, *popty_proj), label='proj')
            ay.plot(xy, vline/_np.sum(vline), '.', label='line')
            ay.plot(xy, gauss(xy, *popty)/_np.sum(vline), label='slice')

            ax.legend()
            ay.legend()
            fig.tight_layout()
            fig.show()

        return sigmax, usigmax, sigmay, usigmay

    @staticmethod
    def extract_quadrupoles_ramp(ramp):
        """."""
        conf_times = _np.sort(ramp.ps_normalized_configs_times)
        qf_ramp = _np.zeros(len(conf_times))
        qd_ramp = qf_ramp.copy()
        for i, c_time in enumerate(conf_times):
            qf_ramp[i] = ramp[c_time]['BO-Fam:PS-QF']
            qd_ramp[i] = ramp[c_time]['BO-Fam:PS-QD']
        return qf_ramp, qd_ramp

    @staticmethod
    def gauss(x, amp, x0, sigma, off):
        """."""
        return amp*_np.exp(-(x-x0)**2/(2*sigma**2)) + off

    def _send_ramp_wfm(self, boramp=None):
        """."""
        conn = self.devices['conn_ps']
        if boramp is None:
            boramp = self.devices['bo_ramp']
        conn.get_ramp_config(boramp)
        conn.cmd_wfm()

    def _create_dict_with_lists(self):
        """."""
        data = dict()
        data['measure'] = []
        data['delays'] = []
        data['timestamp'] = []
        data['qf_wfm'] = []
        data['qd_wfm'] = []
        data['images'] = []
        data['curr3gev'] = []

        return data
