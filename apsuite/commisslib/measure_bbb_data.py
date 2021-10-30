"""Main module."""
import time as _time

import numpy as _np
import scipy.signal as _scysig
import scipy.integrate as _scyint
import scipy.optimize as _scyopt

import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs
from matplotlib.collections import PolyCollection as _PolyCollection

from siriuspy.devices import BunchbyBunch, PowerSupplyPU, EGTriggerPS

from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass, \
    ThreadedMeasBaseClass as _ThreadBaseClass


class UtilClass:
    """."""

    @staticmethod
    def get_data(bbb, acqtype):
        """Get Raw data to file."""
        acq = bbb.sram if acqtype in 'SRAM' else bbb.bram
        rawdata = acq.data_raw.reshape(
            (-1, bbb.info.harmonic_number)).T.astype(float)
        dtime = acq.downsample / bbb.info.revolution_freq_nom
        return dict(
            rawdata=rawdata,
            time=_np.arange(rawdata.shape[1]) * dtime * 1000,
            dft_freq=_np.fft.fftfreq(rawdata.shape[1], d=dtime),
            stored_current=bbb.dcct.current,
            timestamp=_time.time(),
            cavity_data=UtilClass.get_cavity_data(bbb),
            acqtype=acqtype, downsample=acq.downsample,
            fb_set0=bbb.coeffs.set0, fb_set1=bbb.coeffs.set1,
            fb_set0_desc=bbb.coeffs.set0_desc,
            fb_set1_desc=bbb.coeffs.set1_desc,
            fb_downsample=bbb.feedback.downsample,
            fb_state=bbb.feedback.loop_state,
            fb_shift_gain=bbb.feedback.shift_gain,
            fb_setsel=bbb.feedback.coeff_set,
            fb_growdamp_state=bbb.feedback.grow_damp_state,
            fb_mask=bbb.feedback.mask,
            growth_time=acq.growthtime, acq_time=acq.acqtime,
            hold_time=acq.holdtime, post_time=acq.posttime,
            )

    @staticmethod
    def get_cavity_data(bbb):
        """."""
        return dict(
            temperature={
                'cell1': bbb.rfcav.dev_cavmon.temp_cell1,
                'cell2': bbb.rfcav.dev_cavmon.temp_cell2,
                'cell3': bbb.rfcav.dev_cavmon.temp_cell3,
                'cell4': bbb.rfcav.dev_cavmon.temp_cell4,
                'cell5': bbb.rfcav.dev_cavmon.temp_cell5,
                'cell6': bbb.rfcav.dev_cavmon.temp_cell6,
                'cell7': bbb.rfcav.dev_cavmon.temp_cell7,
                'coupler': bbb.rfcav.dev_cavmon.temp_coupler,
                },
            power={
                'cell2': bbb.rfcav.dev_cavmon.power_cell2,
                'cell4': bbb.rfcav.dev_cavmon.power_cell4,
                'cell6': bbb.rfcav.dev_cavmon.power_cell6,
                'forward': bbb.rfcav.dev_cavmon.power_forward,
                'reverse': bbb.rfcav.dev_cavmon.power_reverse,
                'voltage': bbb.rfcav.dev_cavmon.gap_voltage,
                },
            voltage=bbb.rfcav.dev_llrf.voltage_mon,
            phase=bbb.rfcav.dev_llrf.phase_mon,
            detune=bbb.rfcav.dev_llrf.detune,
            detune_error=bbb.rfcav.dev_llrf.detune_error,
            )

    @staticmethod
    def _process_data(data, params, rawdata=None):
        """."""
        per_rev = params.PER_REV
        calib = params.CALIBRATION_FACTOR
        harm_num = params.HARM_NUM
        current = data.get('stored_current', None)
        if current is None:
            current = data['current']

        time = data['time'].copy()
        freq = data['dft_freq'].copy()

        if rawdata is None:
            dataraw = data['rawdata'].copy()
            dataraw *= 1 / (calib * current / harm_num)
        else:
            dataraw = rawdata.copy()

        # remove DC component from bunches
        dataraw -= dataraw.mean(axis=1)[:, None]

        # get the analytic data vector, via discrete hilbert transform
        data_anal = _scysig.hilbert(dataraw, axis=1).copy()

        # calculate DFT:
        data_dft = _np.fft.fft(data_anal, axis=1)

        # compensate the different time samplings of each bunch:
        dts = _np.arange(data_anal.shape[0])/data_anal.shape[0] * per_rev
        comp = _np.exp(-1j*2*_np.pi * freq[None, :]*dts[:, None])
        data_dft *= comp

        # get the processed data by inverse DFT
        data_anal = _np.fft.ifft(data_dft, axis=1)

        # decompose data into even fill eigenvectors:
        data_modes = _np.fft.fft(data_anal, axis=0) / data_anal.shape[0]

        analysis = dict()
        analysis['bunch_numbers'] = _np.arange(1, dataraw.shape[0]+1)
        analysis['dft_freq'] = freq
        analysis['mode_numbers'] = _np.arange(data_modes.shape[0])
        analysis['time'] = time
        analysis['mode_data'] = data_modes
        analysis['bunch_data'] = data_anal

        return analysis

    @staticmethod
    def filter_data(freq, data, params):
        """."""
        center_freq = params.center_frequency
        sigma_freq = params.bandwidth
        ftype = params.filter_type

        data_dft = _np.fft.fft(data, axis=-1)

        if ftype.lower().startswith('gauss'):
            # Apply Gaussian filter to get only the synchrotron frequency
            H = _np.exp(-(freq - center_freq)**2/2/sigma_freq**2)
            H += _np.exp(-(freq + center_freq)**2/2/sigma_freq**2)
            H /= H.max()
            if len(data.shape) > 1:
                data_dft *= H[None, :]
            else:
                data_dft *= H[None, :]
        else:
            indcs = (freq > center_freq - sigma_freq)
            indcs &= (freq < center_freq + sigma_freq)
            if len(data.shape) > 1:
                data_dft[:, ~indcs] = 0
            else:
                data_dft[~indcs] = 0
        return _np.fft.ifft(data_dft, axis=-1)

    @staticmethod
    def estimate_fitting_intervals(infos, int_type='both', clearance=0):
        """."""
        growth_time = infos['growth_time']
        post_time = infos['post_time']
        tim = infos['time']

        change_time = tim[-1] - (post_time - growth_time)
        change_time = max(tim[0], min(tim[-1], change_time))

        grow_int = [0 + clearance, change_time - clearance]
        damp_int = [change_time + clearance, tim[-1] - clearance]
        if int_type == 'both':
            return [grow_int, damp_int]
        if int_type == 'damp':
            return [damp_int, ]
        return [grow_int, ]

    @staticmethod
    def get_strongest_modes(
            data_modes, nr_modes=None, nr_std=4, ignore_mode0=True):
        """."""
        abs_modes = _np.abs(data_modes)

        avg_modes = abs_modes.max(axis=1)
        if ignore_mode0:
            avg_modes = avg_modes[1:]

        if nr_modes is not None:
            modes = _np.argsort(avg_modes)[::-1]
            modes = modes[:nr_modes]
        else:
            avg = avg_modes.mean()
            std = avg_modes.std()
            modes = (avg_modes > (avg + nr_std*std)).nonzero()[0]

        modes += ignore_mode0
        return modes

    @classmethod
    def fit_exponential(cls, times, data, t_ini=None, t_fin=None, offset=True):
        """Fit exponential function."""
        t_ini = t_ini or times.min()
        t_fin = t_fin or times.max()
        idx = (times >= t_ini) & (times <= t_fin)
        tim = times[idx]
        dtim = data[idx]

        # Exponential function without offset
        if not offset:
            log_amp, rate = _np.polynomial.polynomial.polyfit(
                tim, _np.log(dtim), deg=1, rcond=None)
            return tim, (0, _np.exp(log_amp), rate)

        # method to estimate fitting parameters of
        # y = a + b*exp(c*x)
        # based on:
        # https://www.scribd.com/doc/14674814/Regressions-et-equations-integrales
        # pages 16-18
        s = _scyint.cumtrapz(dtim, x=tim, initial=0.0)
        ym = dtim - dtim[0]
        xm = tim - tim[0]
        mat = _np.array([xm, s]).T
        (_, rate), *_ = _np.linalg.lstsq(mat, ym, rcond=None)
        theta = _np.exp(rate*tim)
        mat = _np.ones((theta.size, 2))
        mat[:, 1] = theta
        (off, amp), *_ = _np.linalg.lstsq(mat, dtim, rcond=None)

        # Now use scipy to refine the estimatives:
        coefs = (off, amp, rate)
        try:
            coefs, _ = _scyopt.curve_fit(
                cls.exponential_func, tim, dtim, p0=coefs)
        except RuntimeError:
            pass
        return tim, coefs

    @staticmethod
    def exponential_func(tim, off, amp, rate):
        """Return exponential function with offset."""
        return off + amp*_np.exp(rate*tim)

    @staticmethod
    def calc_instant_frequency(data, dtime):
        """."""
        freq = _np.unwrap(_np.angle(data))
        freq = _np.gradient(freq, axis=-1)
        freq /= 2*_np.pi*dtime
        return freq


class BbBLParams(_ParamsBaseClass):
    """."""

    DAC_NBITS = 14
    SAT_THRES = 2**(DAC_NBITS-1) - 1
    CALIBRATION_FACTOR = 1000  # Counts/mA/degree
    DAMPING_RATE = 1/13.0  # Hz
    FREQ_RF = 499666000
    HARM_NUM = 864
    FREQ_REV = FREQ_RF / HARM_NUM
    PER_REV = 1 / FREQ_REV

    def __init__(self):
        """."""
        super().__init__()
        self.center_frequency = 2090  # [Hz]
        self.bandwidth = 200  # [Hz]
        self.filter_type = 'gauss'  # (gauss, sinc)
        self.acqtype = 'SRAM'

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format
        st = ftmp('center_frequency  [Hz]', self.center_frequency, '')
        st += ftmp('bandwidth [Hz]', self.bandwidth, '')
        st += stmp('filter_type', self.filter_type, '[gauss or sinc]')
        st += stmp('acqtype', self.acqtype, '[SRAM or BRAM]')
        return st


class BbBHParams(BbBLParams):
    """."""

    DAMPING_RATE = 1/16.9e-3  # Hz
    CALIBRATION_FACTOR = 1000  # Counts/mA/um


class BbBVParams(BbBLParams):
    """."""

    DAMPING_RATE = 1/22.0e-3  # Hz
    CALIBRATION_FACTOR = 1000  # Counts/mA/um


class BbBAcqData(_BaseClass, UtilClass):
    """."""

    DEVICES = BunchbyBunch.DEVICES

    def __init__(self, devname):
        """."""
        if devname.endswith('L'):
            params = BbBLParams()
        elif devname.endswith('H'):
            params = BbBHParams()
        elif devname.endswith('V'):
            params = BbBVParams()

        super().__init__(params=params)
        if self.isonline:
            self.devices['bbb'] = BunchbyBunch(devname)

    def get_data(self):
        """Get Raw data to file."""
        acqtype = self.params.acqtype
        bbb = self.devices['bbb']
        self.data = UtilClass.get_data(bbb, acqtype)

    def get_cavity_data(self):
        """."""
        bbb = self.devices['bbb']
        return UtilClass.get_cavity_data(bbb)

    def process_data(self, rawdata=None):
        """."""
        analysis = UtilClass._process_data(
            self.data, self.params, rawdata=rawdata)

        freq = analysis['dft_freq']
        analysis['mode_data'] = UtilClass.filter_data(
            freq, analysis['mode_data'], self.params)
        analysis['bunch_data'] = UtilClass.filter_data(
            freq, analysis['bunch_data'], self.params)

        if rawdata is None:
            self.analysis.update(analysis)
        return analysis

    def load_and_apply_old_data(self, fname):
        """."""
        data = self.load_data(fname)
        if not isinstance(data['data'], _np.ndarray):
            self.load_and_apply(fname)
            return

        data.pop('rf_freq')
        data.pop('harmonic_number')
        data['rawdata'] = _np.array(
            data.pop('data').reshape((-1, self.params.HARM_NUM)).T,
            dtype=float)
        data['cavity_data'] = dict(
            temperature={
                'cell1': 0.0, 'cell2': 0.0, 'cell3': 0.0, 'cell4': 0.0,
                'cell5': 0.0, 'cell6': 0.0, 'cell7': 0.0, 'coupler': 0.0},
            power={
                'cell2': 0.0, 'cell4': 0.0, 'cell6': 0.0, 'forward': 0.0,
                'reverse': 0.0},
            voltage=0.0, phase=0.0, detune=0.0, detune_error=0.0,
            )
        self.data = data

    def get_dac_output(self, coeff=None, shift_gain=None, saturate=True):
        """."""
        if coeff is None:
            coeff = self.data[f"fb_set{self.data['fb_setsel']:d}"]
        if shift_gain is None:
            shift_gain = self.data['fb_shift_gain']
        dac_out = _scysig.convolve(
            self.data['rawdata'], coeff[None, :], mode='valid')
        dac_out *= 2**shift_gain
        if saturate:
            idcs = dac_out > self.params.SAT_THRES
            dac_out[idcs] = self.params.SAT_THRES
            idcs = dac_out < -self.params.SAT_THRES
            dac_out[idcs] = -self.params.SAT_THRES
        return dac_out

    def pca_analysis(self, rawdata=None):
        """."""
        calib = self.params.CALIBRATION_FACTOR
        harm_num = self.params.HARM_NUM
        current = self.data.get('stored_current', None)
        if current is None:
            current = self.data['current']

        if rawdata is None:
            rawdata = self.data['rawdata'].copy()
            rawdata *= 1 / (calib * current / harm_num)
        else:
            rawdata = rawdata.copy()

        rawdata -= rawdata.mean(axis=1)[:, None]
        return _np.linalg.svd(rawdata)

    def get_strongest_modes(
            self, nr_modes=None, nr_std=4, ignore_mode0=True, analysis=None):
        """."""
        if analysis is None:
            analysis = self.analysis
        data_modes = analysis['mode_data']

        return UtilClass.get_strongest_modes(
            data_modes, nr_modes=nr_modes, nr_std=nr_std,
            ignore_mode0=ignore_mode0)

    def estimate_fitting_intervals(self, int_type='both', clearance=0):
        """."""
        return super().estimate_fitting_intervals(
            self.data, int_type, clearance)

    def fit_and_plot_growth_rates(
            self, mode_num, intervals=None, labels=None, title='',
            analysis=None):
        """."""
        if analysis is None:
            analysis = self.analysis
        if intervals is None:
            intervals = self.estimate_fitting_intervals()

        tim = analysis['time']
        dtime = (tim[1] - tim[0]) / 1000

        data_mode = analysis['mode_data'][mode_num]
        abs_mode = _np.abs(data_mode)

        labels = ['']*len(intervals) if labels is None else labels

        fig = _mplt.figure(figsize=(7, 8))
        gsp = _mgs.GridSpec(2, 1)
        gsp.update(left=0.15, right=0.95, top=0.94, bottom=0.1, hspace=0.2)
        aty = _mplt.subplot(gsp[0, 0])
        atx = _mplt.subplot(gsp[1, 0], sharex=aty)

        aty.plot(tim, abs_mode, label='Data')

        gtimes = []
        for inter, label in zip(intervals, labels):
            ini, fin = inter
            tim_fit, coef = self.fit_exponential(
                tim, abs_mode, t_ini=ini, t_fin=fin)
            fit = self.exponential_func(tim_fit, *coef)
            gtimes.append(coef[2] * 1000)
            aty.plot(tim_fit, fit, label=label)
            aty.annotate(
                f'rate = {coef[2]*1000:.2f} Hz', fontsize='x-small',
                xy=(tim_fit[fit.size//2], fit[fit.size//2]),
                textcoords='offset points', xytext=(-100, 10),
                arrowprops=dict(arrowstyle='->'),
                bbox=dict(boxstyle="round", fc="0.8"))

        aty.legend(loc='best', fontsize='small')
        aty.set_title(title, fontsize='small')
        aty.set_xlabel('time [ms]')
        aty.set_ylabel('Amplitude [°]')

        idx = abs_mode > abs_mode.max()/10
        inst_freq = self.calc_instant_frequency(data_mode, dtime)
        inst_freq /= 1e3
        atx.plot(tim[idx], inst_freq[idx])
        atx.set_xlabel('time [ms]')
        atx.set_ylabel('Instantaneous Frequency [kHz]')
        cenf = self.params.center_frequency / 1000
        sig = self.params.bandwidth / 1000
        atx.set_ylim([cenf - sig, cenf + sig])

        fig.show()
        return fig, aty, atx, gtimes

    def plot_modes_evolution(
            self, nr_modes=None, nr_std=4, ignore_mode0=True, title='',
            analysis=None):
        """."""
        if analysis is None:
            analysis = self.analysis

        data_modes = analysis['mode_data']
        abs_modes = _np.abs(data_modes)
        tim = analysis['time']
        dtime = (tim[1] - tim[0]) / 1000

        avg_modes = abs_modes.max(axis=1)

        indcs = self.get_proeminent_modes(
            nr_modes=nr_modes, nr_std=nr_std, ignore_mode0=ignore_mode0,
            analysis=analysis)

        fig = _mplt.figure(figsize=(7, 10))
        gsp = _mgs.GridSpec(3, 1)
        gsp.update(left=0.15, right=0.95, top=0.96, bottom=0.07, hspace=0.23)
        ax = _mplt.subplot(gsp[0, 0])
        aty = _mplt.subplot(gsp[1, 0])
        atx = _mplt.subplot(gsp[2, 0], sharex=aty)

        ax.plot(avg_modes, 'k')
        for idx in indcs:
            data = data_modes[idx, :]
            abs_mode = abs_modes[idx, :]
            ax.plot(idx, avg_modes[idx], 'o')
            aty.plot(tim, abs_mode, label=f'{idx:03d}')
            inst_freq = self.calc_instant_frequency(data, dtime)
            nzer = abs_mode > abs_mode.max()/10
            atx.plot(tim[nzer], inst_freq[nzer]/1e3, label=f'{idx:03d}')

        aty.legend(loc='best', fontsize='small')
        ax.set_title(title)
        ax.set_ylabel('Max Amplitude [°]')
        ax.set_xlabel('Mode Number')
        aty.set_xlabel('time [ms]')
        aty.set_ylabel('Amplitude [°]')
        atx.set_xlabel('time [ms]')
        atx.set_ylabel('Instantaneous Frequency [kHz]')

        cenf = self.params.center_frequency / 1000
        sig = self.params.bandwidth / 1000
        atx.set_ylim([cenf - sig, cenf + sig])

        fig.show()
        return fig, ax, aty, atx

    def plot_average_spectrum(self, rawdata=None, subtract_mean=True):
        """."""
        if rawdata is None:
            rawdata = self.data['rawdata']
        rawdata = rawdata.copy()
        per_rev = self.params.PER_REV
        downsample = self.data['downsample']
        dtime = per_rev*downsample

        if subtract_mean:
            rawdata -= rawdata.mean(axis=1)[:, None]
        dataraw_dft = _np.fft.rfft(rawdata, axis=1)
        rfreq = _np.fft.rfftfreq(rawdata.shape[1], d=dtime)
        avg_dataraw = _np.abs(dataraw_dft).mean(axis=0)/rawdata.shape[1]

        f = _mplt.figure(figsize=(7, 4))
        gs = _mgs.GridSpec(1, 1)
        gs.update(
            left=0.15, right=0.95, top=0.97, bottom=0.18, wspace=0.35,
            hspace=0.2)
        aty = _mplt.subplot(gs[0, 0])

        aty.plot(rfreq, avg_dataraw)
        aty.set_yscale('log')
        f.show()
        return f, aty

    def plot_modes_summary(self, analysis=None):
        """."""
        if analysis is None:
            analysis = self.analysis

        data_modes = analysis['mode_data']
        data_anal = analysis['bunch_data']
        tim = analysis['time']
        mode_nums = analysis['mode_numbers']
        bunch_nums = analysis['bunch_numbers']

        f = _mplt.figure(figsize=(12, 9))
        gs = _mgs.GridSpec(2, 2)
        gs.update(
            left=0.10, right=0.95, top=0.97, bottom=0.10, wspace=0.35,
            hspace=0.2)
        # aty = _mplt.subplot(gs[0, :7], projection='3d')
        # afy = _mplt.subplot(gs[1, :7], projection='3d')
        aty = _mplt.subplot(gs[0, 0])
        afy = _mplt.subplot(gs[1, 0])
        atx = _mplt.subplot(gs[0, 1])
        afx = _mplt.subplot(gs[1, 1])

        abs_modes = _np.abs(data_modes)
        abs_dataf = _np.abs(data_anal)

        afx.plot(mode_nums, abs_modes.mean(axis=1))
        afx.set_xlabel('Mode Number')
        afx.set_ylabel('Average Amplitude [°]')
        # afx.set_yscale('log')

        # waterfall_plot(afy, tim, mode_nums, abs_modes)
        # afy.set_ylabel('\ntime [ms]')
        # afy.set_xlabel('\nmode number')
        # afy.set_zlabel('amplitude')
        T, M = _np.meshgrid(tim, mode_nums)
        cf = afy.pcolormesh(
            T, M, abs_modes, cmap='jet',
            vmin=abs_modes.min(), vmax=abs_modes.max())
        afy.set_xlabel('Time [ms]')
        afy.set_ylabel('Mode Number')
        cb = f.colorbar(cf, ax=afy, pad=0.01)
        cb.set_label('Amplitude [°]')

        atx.plot(bunch_nums, abs_dataf.mean(axis=1))
        atx.set_xlabel('Bunch Number')
        atx.set_ylabel('Average Amplitude [°]')

        # waterfall_plot(aty, tim, bunch_nums, abs_dataf)
        # aty.set_ylabel('\ntime [ms]')
        # aty.set_xlabel('\nbunch number')
        # aty.set_zlabel('amplitude')
        T, M = _np.meshgrid(tim, bunch_nums)
        cf = aty.pcolormesh(
            T, M, abs_dataf, cmap='jet',
            vmin=abs_dataf.min(), vmax=abs_dataf.max())
        aty.set_xlabel('Time [ms]')
        aty.set_ylabel('Bunch Number')
        cb = f.colorbar(cf, ax=aty, pad=0.01)
        cb.set_label('Amplitude [°]')

        f.show()
        return f

    @staticmethod
    def waterfall_plot(axis, xs, zs, data):
        """."""
        vertcs, colors = [], []
        cors = ['b', 'r', 'g', 'y', 'm', 'c']
        for i, y in enumerate(zs):
            ys = data[i, :].copy()
            ys[0], ys[-1] = 0, 0
            vertcs.append(list(zip(xs, ys)))
            colors.append(cors[i % len(cors)])
        poly = _PolyCollection(
            vertcs, closed=False, edgecolors='k',
            linewidths=1, facecolors=colors)

        poly.set_alpha(0.7)
        axis.add_collection3d(poly, zs=zs, zdir='x')
        axis.view_init(elev=35.5, azim=-135)
        axis.set_ylim3d(xs.min(), xs.max())
        axis.set_xlim3d(zs.min(), zs.max())
        axis.set_zlim3d(0, data.max())


class DriveDampLParams(BbBLParams):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.modes_to_measure = _np.arange(self.HARM_NUM//2+1)
        self.drive_num = 0
        self.wait_acquisition = 2  # [s]
        self.fitting_clearance = 5  # [ms]

    def __str__(self):
        """."""
        dtmp = '{0:24s} = {1:9d}\n'.format
        ftmp = '{0:24s} = {1:9.4f}  {2:s}\n'.format
        stmp = '{0:24s} = {1:}\n'.format

        stg = super().__str__()
        stg += dtmp('drive_num', self.drive_num)
        stg += ftmp('wait_acquisition', self.wait_acquisition, '[s]')
        stg += ftmp('fitting_clearance', self.fitting_clearance, '[ms]')

        modes = self.modes_to_measure
        if len(modes) > 6:
            modes_stg = ', '.join([f'{m:03d}' for m in modes[:3]])
            modes_stg += ', ..., '
            modes_stg += ', '.join([f'{m:03d}' for m in modes[-3:]])
        else:
            modes_stg = ', '.join([f'{m:03d}' for m in modes])
        stg += stmp(
            'modes_to_measure', f'[{modes_stg:s}] (size = {len(modes):d})')
        return stg


class DriveDampHParams(DriveDampLParams):
    """."""

    DAMPING_RATE = 1/16.9e-3  # Hz
    CALIBRATION_FACTOR = 1000  # Counts/mA/um


class DriveDampVParams(DriveDampLParams):
    """."""

    DAMPING_RATE = 1/22.0e-3  # Hz
    CALIBRATION_FACTOR = 1000  # Counts/mA/um


class MeasDriveDamp(_ThreadBaseClass, UtilClass):
    """."""

    DEVICES = BbBAcqData.DEVICES

    def __init__(self, devname, isonline=True):
        """."""
        if devname.endswith('L'):
            params = DriveDampLParams()
        elif devname.endswith('H'):
            params = DriveDampHParams()
        elif devname.endswith('V'):
            params = DriveDampVParams()

        super().__init__(
            params=params, target=self._do_measure, isonline=isonline)
        if self.isonline:
            self.devices['bbb'] = BunchbyBunch(devname)

    def process_data(self):
        """."""
        infos = self.data['infos']
        data = self.data['modes_data']

        analysis = dict(coeffs=[], tim_fits=[], fits=[], modes_filt=[])
        for info, datum in zip(infos, data):
            tim = info['time']
            freq = info['dft_freq']
            interval = self.estimate_fitting_intervals(
                info, clearance=self.params.fitting_clearance)
            coeff, tim_fit, fit = self.fit_growth_rates(
                tim, datum, interval=interval, full=True)

            mode_filt = self.filter_data(freq, datum, self.params)

            analysis['coeffs'].append(coeff)
            analysis['tim_fits'].append(tim_fit)
            analysis['fits'].append(fit)
            analysis['modes_filt'].append(mode_filt)
        self.analysis = analysis

    def estimate_fitting_intervals(self, infos, clearance=5):
        """."""
        return super().estimate_fitting_intervals(
            infos, int_type='damp', clearance=clearance)[0]

    def fit_growth_rates(self, tim, data, interval, full=False):
        """."""
        freq = _np.fft.rfftfreq(tim.size, d=(tim[1]-tim[0])/1000)
        data = UtilClass.filter_data(freq, data, self.params)
        abs_data = _np.abs(data)
        if len(abs_data.shape) > 1:
            abs_data = abs_data.reshape(abs_data.size, -1)

        coeffs, fittings = [], []
        for abs_mode in abs_data:
            tim_fit, coef = self.fit_exponential(
                tim, abs_mode, t_ini=interval[0], t_fin=interval[1])
            fit = self.exponential_func(tim_fit, *coef)
            coeffs.append(coef)
            fittings.append(fit)

        coeffs = _np.array(coeffs)
        fittings = _np.array(fittings)
        if not full:
            return coeffs[:, 2]
        return coeffs, tim_fit, fittings

    def plot_growth_rates(self):
        """."""
        fig = _mplt.figure(figsize=(7, 5))
        gsp = _mgs.GridSpec(1, 1)
        gsp.update(left=0.09, right=0.98, top=0.98, bottom=0.1)
        ax = _mplt.subplot(gsp[0, 0])

        modes_meas = self.data['modes_measured']
        coeffs = self.analysis['coeffs']
        for num, coeff in zip(modes_meas, coeffs):
            ax.plot(num, coeff[:, 2]*1000, 'ob')
        ax.set_ylabel('Growth Rates [1/s]')
        ax.set_xlabel('Modes')
        return fig, ax

    def plot_modes_evolution(self, data_index=0, title=''):
        """."""
        mode_meas = self.data['modes_measured'][data_index]
        infos = self.data['infos'][data_index]
        mode_filt = self.analysis['modes_filt'][data_index]
        tfit = self.analysis['tims_fit'][data_index]
        fit = self.analysis['fits'][data_index]
        coeff = self.analysis['coeffs'][data_index]

        abs_mode = _np.abs(mode_filt)

        fig = _mplt.figure(figsize=(7, 8))
        gsp = _mgs.GridSpec(2, 1)
        gsp.update(left=0.15, right=0.95, top=0.94, bottom=0.1, hspace=0.2)
        aty = _mplt.subplot(gsp[0, 0])
        atx = _mplt.subplot(gsp[1, 0], sharex=aty)

        tim = infos['time']
        dtime = (tim[1] - tim[0]) / 1000
        for i, absm in enumerate(abs_mode):
            ffit = fit[i]
            cff = coeff[i]
            num = mode_meas[i]
            aty.plot(tim, absm, label=f'{num:03d}')
            aty.plot(tfit, ffit)

            idx = int(ffit.size/abs_mode.shape[0] * i)
            aty.annotate(
                f'rate = {cff[2]*1000:.2f} Hz', fontsize='x-small',
                xy=(tfit[idx], ffit[idx]),
                textcoords='offset points', xytext=(-100, 10),
                arrowprops=dict(arrowstyle='->'),
                bbox=dict(boxstyle="round", fc="0.8"))

            inst_freq = self.calc_instant_frequency(mode_filt[i], dtime)
            inst_freq /= 1e3
            idx = absm > absm.max()/10
            atx.plot(tim[idx], inst_freq[idx])

        aty.legend(loc='best', fontsize='small')
        aty.set_title(title, fontsize='small')
        aty.set_xlabel('time [ms]')
        aty.set_ylabel('Amplitude [°]')

        atx.set_xlabel('time [ms]')
        atx.set_ylabel('Instantaneous Frequency [kHz]')
        cenf = self.params.center_frequency / 1000
        sig = self.params.bandwidth / 1000
        atx.set_ylim([cenf - sig, cenf + sig])

        fig.show()
        return fig, aty, atx, coeff[:, 2] * 1000

    def _do_measure(self):
        acqtype = self.params.acqtype
        drive_num = self.params.drive_num

        bbb = self.devices['bbb']
        drive = bbb.drive0 if drive_num == 0 else bbb.drive1
        drive = drive if drive_num == 1 else bbb.drive2

        harm_num = bbb.info.harmonic_number
        modes_to_measure = self.params.modes_to_measure
        bunches = _np.arange(harm_num)

        bbb.sram.cmd_data_dump()
        _time.sleep(0.2)
        infos = self.get_data(bbb, acqtype)
        analysis = self._process_data(infos, self.params)
        intervals = self.estimate_fitting_intervals(infos, clearance=5)

        if not self.data:
            self.data = dict(infos=[], modes_data=[], modes_measured=[])
        for mode in modes_to_measure:
            drive.mask = _np.cos(2*_np.pi*bunches*mode/harm_num) > 0
            _time.sleep(self.params.wait_acquisition)
            bbb.sram.cmd_data_dump()

            _time.sleep(0.2)
            infos = self.get_data(bbb, acqtype)
            analysis = self._process_data(infos, self.params)
            modei = sorted({mode, harm_num - mode})
            data = analysis['mode_data'][modei]
            infos.pop('rawdata')

            self.data['modes_measured'].append(modei)
            self.data['modes_data'].append(data)
            self.data['infos'].append(infos)

            grt = self.fit_growth_rates(data, intervals=intervals, full=False)
            print(',   '.join([
                f'mode: {m:03d} --> growth: {gt:.2f}'
                for m, gt in zip(modei, grt.ravel())]))
            if self._stopevt.is_set():
                print('Stopping...')
                break
        print('Finished!!')


class TuneShiftParams(_ParamsBaseClass):
    """."""

    TIME_REV = 864 / 499666000  # s
    WAIT_INJ = 0.2  # s
    DEF_TOL_CURRENT = 0.01  # mA

    def __init__(self):
        """."""
        super().__init__()
        self.plane = 'HV'  # 'H', 'V' or 'HV'
        self.kickh = -25/1000  # mrad
        self.kickv = +20/1000  # mrad
        self.wait_bbb = 9  # s
        self.currents = list()
        self.filename = ''

    def __str__(self):
        """."""
        dtmp = '{0:10s} = {1:9d}  {2:s}\n'.format
        ftmp = '{0:10s} = {1:9.3f}  {2:s}\n'.format
        ltmp = '{0:6.3f},'.format
        stg = f"{'plane':10s} = {self.plane:4s} ('H', 'V' or 'HV')\n"
        stg += ftmp('kickh', self.kickh, '[mrad]')
        stg += ftmp('kickv', self.kickv, '[mrad]')
        stg += dtmp('wait_bbb', self.wait_bbb, '[s]')
        stg += f"{'currents':10s} = ("
        stg += ''.join(map(ltmp, self.currents))
        stg += ' ) mA \n'
        stg += f"{'filename':10s} = '{self.filename:s}'\n"
        return stg


class MeasTuneShift(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        self.devices = dict()
        self.data = dict()
        self.params = TuneShiftParams()
        self.pingers = list()
        if isonline:
            self.devices['bbbh'] = BunchbyBunch(BunchbyBunch.DEVICES.H)
            self.devices['bbbv'] = BunchbyBunch(BunchbyBunch.DEVICES.V)
            self.devices['pingh'] = PowerSupplyPU(
                PowerSupplyPU.DEVICES.SI_INJ_DPKCKR)
            self.devices['pingv'] = PowerSupplyPU(
                PowerSupplyPU.DEVICES.SI_PING_V)
            self.devices['egun'] = EGTriggerPS()
            self.pingers = [self.devices['pingh'], self.devices['pingv']]

    def get_data(self, plane):
        """."""
        bbbtype = 'bbbh' if plane.upper() == 'H' else 'bbbv'
        bbb = self.devices[bbbtype]
        sb_tune = bbb.single_bunch
        data = {
            'timestamp': _time.time(),
            'stored_current': bbb.dcct.current,
            'data': sb_tune.data_raw,
            'spec_mag': sb_tune.spec_mag, 'spec_freq': sb_tune.spec_freq,
            'spec_phase': sb_tune.spec_phase,
            'bunch_id': sb_tune.bunch_id, 'fft_size': sb_tune.fft_size,
            'fft_overlap': sb_tune.fft_overlap,
            'delay_calibration': sb_tune.delay_calibration,
            'nr_averages': sb_tune.nr_averages}
        return data

    def merge_data(data1, data2):
        """."""
        if data1.keys() != data2.keys():
            raise Exception('Incompatible data sets')
        merge = dict()
        for key in data1:
            merge[key] = data1[key] + data2[key]
        return merge

    def turn_on_pingers_pulse(self):
        """."""
        for ping in self.pingers:
            ping.cmd_turn_on_pulse()

    def turn_off_pingers_pulse(self):
        """."""
        for ping in self.pingers:
            ping.cmd_turn_off_pulse()

    def turn_on_pingers(self):
        """."""
        for ping in self.pingers:
            ping.cmd_turn_on()

    def turn_off_pingers(self):
        """."""
        for ping in self.pingers:
            ping.cmd_turn_off()

    def prepare_pingers(self):
        """."""
        self.devices['pingh'].strength = self.params.kickh
        self.devices['pingv'].strength = self.params.kickv
        self.turn_on_pingers()
        self.turn_on_pingers_pulse()

    def inject_in_storage_ring(self, goal_curr):
        """."""
        self.devices['egun'].cmd_enable_trigger()
        while not self._check_stored_current(goal_curr):
            _time.sleep(TuneShiftParams.WAIT_INJ)
        curr = self.devices['bbbh'].dcct.current
        print(
            f'Stored Current: {curr:.3f}/{goal_curr:.3f}mA.')
        self.devices['egun'].cmd_disable_trigger()

    def _check_stored_current(
            self, goal_curr, tol=TuneShiftParams.DEF_TOL_CURRENT):
        dcct_curr = self.devices['bbbh'].dcct.current
        return dcct_curr > goal_curr or abs(dcct_curr - goal_curr) < tol

    def _check_pingers_problem(self):
        for ping in self.pingers:
            if ping.voltage_mon < 0:
                # reset pinger
                ping.cmd_turn_off()
                ping.cmd_turn_on()
                return True
        return False

    def run_meas(self, save=True):
        """."""
        data = dict()
        datah = list()
        datav = list()
        currs = list()
        for curr in self.params.currents:
            t0 = _time.time()
            self.inject_in_storage_ring(curr)

            trial = 0
            while self._check_pingers_problem():
                if trial > 2:
                    print('3 unsucessful reset trials. Exiting...')
                    break
                print('Problem with pingers voltage. Resetting...')
                _time.sleep(5)
                trial += 1

            _time.sleep(self.params.wait_bbb)
            print('Acquiring data...')
            currs.append(self.devices['bbbh'].dcct.current)
            data['stored_current'] = currs
            if 'H' in self.params.plane:
                datah.append(self.get_data(plane='H'))
                data['horizontal'] = datah
            if 'V' in self.params.plane:
                datav.append(self.get_data(plane='V'))
                data['vertical'] = datav
            self.data = data
            if save:
                self.save_data(fname=self.params.filename, overwrite=True)
                print('Data saved!')
            tf = _time.time()
            print(f'Elapsed time: {tf-t0:.2f}s \n')

    def plot_spectrum(
            self, plane, freq_min=None, freq_max=None,
            title=None, fname=None):
        """plane: must be 'H' or 'V'."""
        if plane.upper() == 'H':
            data = self.data['horizontal']
            freq_min = freq_min or 38
            freq_max = freq_max or 52
        elif plane.upper() == 'V':
            data = self.data['vertical']
            freq_min = freq_min or 72
            freq_max = freq_max or 84
        else:
            raise Exception("plane input must be 'H' or 'V'.")

        curr = _np.array(self.data['stored_current'])
        mag = [dta['spec_mag'] for dta in data]
        mag = _np.array(mag, dtype=float)
        freq = _np.array(data[-1]['spec_freq'])

        idcs = (freq > freq_min) & (freq < freq_max)
        freq = freq[idcs]
        idx = _np.argsort(curr)
        curr = curr[idx]
        mag = mag[idx, :]
        mag = mag[:, idcs]

        freq, curr = _np.meshgrid(freq, curr)
        freq, curr, mag = freq.T, curr.T, mag.T

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.pcolormesh(curr, freq, mag)
        ax.set_ylabel('Frequency [kHz]')
        ax.set_xlabel('Current [mA]')
        ax.set_title(title)
        if fname:
            fig.savefig(fname, format='png', dpi=300)
        return fig

    def plot_time_evolution(
            self, plane, title=None, fname=None):
        """plane: must be 'H' or 'V'."""
        if plane.upper() == 'H':
            data = self.data['horizontal']
        elif plane.upper() == 'V':
            data = self.data['vertical']
        else:
            raise Exception("plane input must be 'H' or 'V'.")

        curr = _np.array(self.data['stored_current'])
        mag = [dta['spec_mag'] for dta in data]
        mag = _np.array(mag, dtype=float)
        mag -= _np.mean(mag, axis=1)[:, None]
        mag = _np.abs(mag)
        dtime = _np.arange(0, mag.shape[1]) * TuneShiftParams.TIME_REV

        idx = _np.argsort(curr)
        curr = curr[idx]
        mag = mag[idx, :]

        dtime, curr = _np.meshgrid(dtime, curr)
        dtime, curr, mag = dtime.T, curr.T, mag.T

        fig = _mplt.figure(figsize=(8, 6))
        gs = _mgs.GridSpec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        ax.pcolormesh(curr, dtime * 1e3, mag)
        ax.set_ylabel('Time [ms]')
        ax.set_xlabel('Current [mA]')
        ax.set_title(title)
        if fname:
            fig.savefig(fname, format='png', dpi=300)
        return fig
