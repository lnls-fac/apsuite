import numpy as _np
import matplotlib.pyplot as _plt
import time as _time

from scipy.optimize import curve_fit
from siriuspy.ramp.ramp import BoosterRamp
from siriuspy.ramp.conn import ConnPS
from siriuspy.devices import EVG, Screen
from siriuspy.epics import PV

from ...utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class BeamSizesParams(_ParamsBaseClass):
    def __init__(self):
        """."""
        self.measures_per_point = 3
        self.nr_points = 15
        self.ts_screen_number = 1
        self.ramp_config = 'bo_ramp_flop_emit_exchange_slower'
        self.init_delay = -3  # [ms]
        self.final_delay = 3  # [ms]


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
        self.devices['bo_ramp'] = BoosterRamp(self.params.ramp_base_config)
        self.devices['bo_ramp'].load()

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

    def reset_ramp(self):
        pass

    def scan_emittance_exchange(self):
        """."""
        boramp = self.devices['bo_ramp']
        evg = self.devices['evg']
        scrn = self.devices['ts_screen']
        pvs = self.pvs
        prms = self.params

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

            # Send the config to the machine
            self._send_ramp_wfm()
            _time.sleep(5)

            for i in range(prms.measures_per_point):
                data = self._create_dict_with_lists()
                print(f"Delay = {delay}, measure {i}.")
                evg.cmd_turn_on_injection()
                data['measure'].append(i)
                data['delay'].append(delay)
                data['timestamp'].append(_time.time())
                data['qf_wfm'].append(pvs['bo-qf-wfm'].get())
                data['qd_wfm'].append(pvs['bo-qd-wfm'].get())
                _time.sleep(10)
                image = scrn.image
                scl_factx = scrn.scale_factor_x
                scl_facty = scrn.scale_factor_x
                data['images'].append(image)
                data['scl_factx'] = _np.abs(scl_factx)
                data['scl_facty'] = _np.abs(scl_facty)

            self.data = data

    def process_data(self):
        """."""
        data = self.data
        prms = self.params
        n_samples = prms.measures_per_point * prms.nr_points

        scl_vec = _np.array([data['scl_factx'],  # pixel to mm
                             data['scl_facty']])
        sigma_arr = _np.zeros([2, n_samples])

        for i in range(n_samples):
            image = data['images'][i]
            mini_sigma_arr = self.calc_beam_size(image) * scl_vec
            sigma_arr[:, i] = mini_sigma_arr

        # Saving beam sizes in the data dict with the flat array convention:
        data['sigmasx'] = sigma_arr
        # Reshaping sigma_arr to output it in a more useful structure,
        # (n_measures_x + n_measures_y) x n_points
        sigmas = sigma_arr.reshape(2*prms.measures_per_point, prms.nr_points)
        # delay_arr is a flat array with non repeated delays
        delay_arr = -_np.array(data['delay'][::prms.measures_per_point])

        return sigmas, delay_arr

    def plot_time_scan(self, figname=None, legend=True, axis=None):
        """."""
        sigmas, delays = self.process_data()
        sx = sigmas[:3, :]
        sy = sigmas[3:, :]
        mean_sx, usx = sx.mean(axis=0), sx.std(axis=0)
        mean_sy, usy = sy.mean(axis=0), sy.std(axis=0)

        if axis is None:
            _, ax = _plt.subplots()
        else:
            ax = axis

        rep_delays = _np.tile(delays, 3)
        ax.scatter(
            rep_delays, sx.ravel(), c='C0', s=1.5,
            label=r'measured $\sigma_x$', marker='v')
        ax.scatter(
            rep_delays, sy.ravel(), c='tab:orange', s=1.5,
            label=r'measured $\sigma_y$', marker='^')
        ax.errorbar(
            delays, mean_sx, yerr=usx, ls='-', marker='', c='C0',
            label=r'$<\sigma_x>$')
        ax.errorbar(
            delays, mean_sy, yerr=usy, ls='-', marker='', c='tab:orange',
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
    def calc_beam_size(self, image):
        """."""
        projx = _np.sum(image, axis=0)
        projy = _np.sum(image, axis=1)
        x_mean_idx = _np.argmax(projx)
        y_mean_idx = _np.argmax(projy)
        hline = image[y_mean_idx, :]
        vline = image[:, x_mean_idx]
        xx = _np.arange(hline.size)
        xy = _np.arange(vline.size)
        sigx = _np.std(hline)
        sigy = _np.std(vline)

        poptx, _ = curve_fit(
            self._gauss, xx, hline, p0=[
                _np.max(hline), x_mean_idx, sigx, hline[0]
                ])
        popty, _ = curve_fit(
            self._gauss, xy, vline, p0=[
                _np.max(vline), y_mean_idx, sigy, vline[0]
                ])

        sigmas_arr = _np.array([poptx[2], popty[2]])
        return sigmas_arr

    @staticmethod
    def extract_quadrupoles_ramp(self, ramp):
        """."""
        ramp = self.devices['bo_ramp']
        conf_times = _np.sort(ramp.ps_normalized_configs_times)
        qf_ramp = _np.zeros(len(conf_times))
        qd_ramp = qf_ramp.copy()
        for i, c_time in enumerate(conf_times):
            qf_ramp[i] = ramp[c_time]['BO-Fam:PS-QF']
            qd_ramp[i] = ramp[c_time]['BO-Fam:PS-QD']
        return qf_ramp, qd_ramp

    def _gauss(x, amp, x0, sigma, off):
        return amp*_np.exp(-(x-x0)**2/(2*sigma**2)) + off

    def _send_ramp_wfm(self):
        """."""
        conn = self.devices['conn_ps']
        boramp = self.devices['bo_ramp']
        conn.get_ramp_config(boramp)
        conn.cmd_wfm()

    def _create_dict_with_lists():
        """."""
        data = dict()
        data['measure'] = []
        data['delay'] = []
        data['timestamp'] = []
        data['qf_wfm'] = []
        data['qd_wfm'] = []
        data['images'] = []
        data['sigmasx'] = []
        data['sigmasy'] = []
        return data
