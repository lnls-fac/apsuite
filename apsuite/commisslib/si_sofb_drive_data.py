"""Main module."""
import time as _time

import numpy as _np
import scipy.optimize as _scyopt

import matplotlib.pyplot as _mplt
import matplotlib.gridspec as _mgs

from siriuspy.devices import SOFB

from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class SOFBDriveParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.start_idxoffset = 0
        self.stop_idxoffset = None

    def __str__(self):
        """."""
        # ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:20s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:20s} = {1:9s}  {2:s}\n'.format
        st = dtmp('start_idxoffset', self.start_idxoffset, '')
        st += stmp('stop_idxoffset', str(self.stop_idxoffset), '')
        return st


class SOFBDriveData(_BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__(params=SOFBDriveParams())
        self.devices['sofb'] = SOFB(SOFB.DEVICES.SI)
        self.cssofb = self.devices['sofb'].data

    def get_data(self):
        """Get Raw data to file."""
        sofb = self.devices['sofb']
        self.data['nrcycles'] = sofb.drivenrcycles
        self.data['freqdivisor'] = sofb.drivefreqdivisor
        self.data['amplitude'] = sofb.driveamplitude
        self.data['phase'] = sofb.drivephase
        self.data['corridx'] = sofb.drivecorridx
        self.data['bpmidx'] = sofb.drivebpmidx
        self.data['type'] = sofb.drivetype
        self.data['time'] = sofb.drivedata_time
        self.data['corr'] = sofb.drivedata_corr
        self.data['bpm'] = sofb.drivedata_bpm
        self.data['timestamp'] = _time.time()

    def process_data(self, rawdata=None):
        """."""
        data = rawdata or self.data
        if not data:
            return

        slc = slice(self.params.start_idxoffset, self.params.stop_idxoffset)
        tim = data['time'][slc]
        corr = data['corr'][slc]
        bpm = data['bpm'][slc]

        dtim = _np.mean(_np.diff(tim))
        bpms_freq = 1 / dtim
        if data['type'] == self.cssofb.DriveType.Sine:
            freq = bpms_freq / data['freqdivisor']
            omega = 2*_np.pi*freq
            corr_coefs = self.fit_sine(tim, corr, omega)
            bpm_coefs = self.fit_sine(tim, bpm, omega)
            latency = (corr_coefs[-1] - bpm_coefs[-1])/corr_coefs[-2]
        elif data['type'] == self.cssofb.DriveType.Square:
            corr_coefs = self.fit_tanh(tim, corr)
            bpm_coefs = self.fit_tanh(tim, bpm)
            latency = (bpm_coefs[-1] - corr_coefs[-1])
        elif data['type'] == self.cssofb.DriveType.Impulse:
            corr_coefs = self.fit_gauss(tim, corr)
            bpm_coefs = self.fit_gauss(tim, bpm)
            latency = (bpm_coefs[-1] - corr_coefs[-1])

        anl = dict()
        anl['bpms_freq'] = bpms_freq
        anl['corr_coefs'] = corr_coefs
        anl['bpm_coefs'] = bpm_coefs
        anl['latency'] = latency
        if rawdata is not None:
            self.analysis = anl
        return anl

    def plot_results(self):
        """."""
        data = self.data
        anl = self.analysis
        if not data:
            data = self._get_dummy_data()
            anl = self.process_data(data)

        corr_names = self.cssofb.ch_names + self.cssofb.cv_names + ['RF', ]
        bpm_names = self.cssofb.bpm_names * 2
        corr_name = corr_names[data['corridx']]
        bpm_name = bpm_names[data['bpmidx']]
        tim = data['time']
        corr = data['corr']
        bpm = data['bpm']
        if data['type'] == self.cssofb.DriveType.Sine:
            corr_fit = self.sine_func(tim, *anl['corr_coefs'])
            bpm_fit = self.sine_func(tim, *anl['bpm_coefs'])
        elif data['type'] == self.cssofb.DriveType.Square:
            corr_fit = self.tanh_func(tim, *anl['corr_coefs'])
            bpm_fit = self.tanh_func(tim, *anl['bpm_coefs'])
        elif data['type'] == self.cssofb.DriveType.Impulse:
            corr_fit = self.gauss_func(tim, *anl['corr_coefs'])
            bpm_fit = self.gauss_func(tim, *anl['bpm_coefs'])

        fig = _mplt.figure()
        gsc = _mgs.GridSpec(1, 1)
        ax = fig.add_subplot(gsc[0, 0])
        ay = ax.twinx()

        ax.grid(True)
        ax.set_title(f"Latency {anl['latency']*1000:.2f} ms")
        ax.plot(tim, corr, 'o', color='C0')
        ax.plot(tim, corr_fit, '--', color='C0', linewidth=2)
        ax.axvline(tim[self.params.start_idxoffset], linestyle='--', color='k')
        if self.params.stop_idxoffset is not None:
            ax.axvline(
                tim[self.params.stop_idxoffset], linestyle='--', color='k')

        ay.plot(tim, bpm, 'o', color='C1', label='Data')
        ay.plot(tim, bpm_fit, '--', color='C1', label='Fitting')

        ax.set_ylabel(corr_name+' [urad]', color='C0')
        ay.set_ylabel(bpm_name+' [um]', color='C1')
        ax.set_xlabel('Time [s]')
        ax.tick_params(axis='y', labelcolor='C0')
        ay.tick_params(axis='y', labelcolor='C1')
        fig.tight_layout()

        return fig

    @classmethod
    def fit_gauss(cls, times, data):
        """Fit gaussian function."""
        # estimate parameters linearly with the given frequency
        off = data.mean()
        amp = (data.max() - data.min())/2
        tim0 = times[times.size//2]
        scale = 1

        # Now use scipy to refine the estimatives:
        coefs = (off, amp, scale, tim0)
        try:
            coefs, _ = _scyopt.curve_fit(
                cls.gauss_func, times, data, p0=coefs)
        except RuntimeError:
            pass
        return coefs

    @staticmethod
    def gauss_func(tim, off, amp, scale, tim0):
        """Return gaussian function with offset."""
        return off + amp*_np.exp(-scale*(tim - tim0)**2)

    @classmethod
    def fit_tanh(cls, times, data):
        """Fit hyperbolic tangent function."""
        # estimate parameters linearly with the given frequency
        off = data.mean()
        amp = (data.max() - data.min())/2
        tim0 = times[times.size//2]
        scale = 1

        # Now use scipy to refine the estimatives:
        coefs = (off, amp, scale, tim0)
        try:
            coefs, _ = _scyopt.curve_fit(
                cls.tanh_func, times, data, p0=coefs)
        except RuntimeError:
            pass
        return coefs

    @staticmethod
    def tanh_func(tim, off, amp, scale, tim0):
        """Return hyperbolic tangent function with offset."""
        return off + amp*_np.tanh(scale*(tim - tim0))

    @classmethod
    def fit_sine(cls, times, data, omega):
        """Fit sine function."""
        # estimate parameters linearly with the given frequency
        mat = _np.array([
            _np.ones(times.size),
            _np.sin(omega*times),
            _np.cos(omega*times)]).T
        fit, *_ = _np.linalg.lstsq(mat, data, rcond=None)

        off = fit[0]
        amp = fit[1]*fit[1] + fit[2]*fit[2]
        phase = _np.arctan2(fit[2], fit[1])

        # Now use scipy to refine the estimatives:
        coefs = (off, amp, omega, phase)
        try:
            coefs, _ = _scyopt.curve_fit(
                cls.sine_func, times, data, p0=coefs)
        except RuntimeError:
            pass
        return coefs

    @staticmethod
    def sine_func(tim, off, amp, omega, phase):
        """Return sine function with offset."""
        return off + amp*_np.sin(omega*tim + phase)

    def _get_dummy_data(self):
        """."""
        nrcycles = 10
        freqdivisor = 12
        amp = 5
        phase = 0
        typem = self.cssofb.DriveType.Impulse
        bfreq = self.cssofb.BPMsFreq

        tim = _np.arange(nrcycles*freqdivisor) / bfreq
        if typem == self.cssofb.DriveType.Sine:
            omega = 2*_np.pi * bfreq / freqdivisor
            corr = self.sine_func(tim, 30, amp, omega, phase)
            bpm = self.sine_func(tim, 50, 150, omega, phase)
            bpm[-4:] = 50
            bpm = _np.roll(bpm, 4)
        elif typem == self.cssofb.DriveType.Square:
            corr = self.tanh_func(tim, 30, amp, 10000, tim[tim.size//2]+1/50)
            bpm = self.tanh_func(tim, 50, 150, 10000, tim[tim.size//2+4]+1/50)
        elif typem == self.cssofb.DriveType.Impulse:
            corr = self.gauss_func(tim, 30, amp, 10000, tim[tim.size//2])
            bpm = self.gauss_func(tim, 50, 150, 10000, tim[tim.size//2 + 4])

        data = dict()
        data['nrcycles'] = nrcycles
        data['freqdivisor'] = freqdivisor
        data['amplitude'] = amp
        data['phase'] = phase
        data['corridx'] = 0
        data['bpmidx'] = 0
        data['type'] = typem
        data['time'] = tim
        data['corr'] = corr
        data['bpm'] = bpm
        return data
