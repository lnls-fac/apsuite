"""."""

import math as _math
import time as _time
from threading import Event as _Event, Thread as _Thread

import mathphys.constants as _const
import numpy as _np
from siriuspy.devices import (
    BunchbyBunch,
    CurrInfoSI,
    FamBLMs,
    FamGammaMonitors,
    RFGen,
    Tune,
    TuneCorr,
)

from ..utils import ParamsBaseClass, ThreadedMeasBaseClass


class SpinDepolUtils:
    """."""

    ELECTRON_A = 1.15965218059e-3  # anomalous magnetic moment of the electron
    ELECTRON_E = _const.electron_rest_energy / _const.elementary_charge  # [eV]

    @classmethod
    def calc_spin_tune_from_energy(cls, energy):
        """Calculate the spin tune and its fractional parts from beam energy.

        Args:
            energy (float): Beam energy in [GeV]

        Returns:
            tuple of floats (spin_tune, spin_tune_frac, 1-spin_tune_frac)
            containing the spin tune value, its fractional part and the
            its complementary fractional part.
        """
        gamma = (energy * 1e9) / cls.ELECTRON_E
        spin_tune = cls.ELECTRON_A * gamma
        spin_tune_frac = spin_tune - int(spin_tune)
        return spin_tune, spin_tune_frac, 1 - spin_tune_frac

    @classmethod
    def calc_energy_from_spin_tune(cls, spin_tune):
        """Calculate the beam energy from the spin tune value.

        Args:
            spin_tune (float): _description_

        Returns:
            float: Beam energy in [GeV]
        """
        gamma = spin_tune / cls.ELECTRON_A
        energy = gamma * cls.ELECTRON_E
        return energy / 1e9


class MeasureSpinDepolScanParams(ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.bbb_amp_gain = (
            38.0  # [dB], threshold value for amp nonlinearities
        )
        # Note: amplifier gain values above 38 dB results in nonlinearities in
        # the BBB output, revealed by the appearance of higher harmonics in the
        # spectrum of the output signal
        self.bbb_drive_pattern = '2:1'
        self.freq_harmonic = 200  # harmonic number to be excited
        self.freq_start = 400.0  # [kHz]
        self.freq_stop = 500.0  # [kHz]
        self.freq_step = 0.001  # [kHz]
        self.excitation_time = 1.5  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format
        # dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format

        stg = ''
        stg += ftmp('bbb_amp_gain', self.bbb_amp_gain, '[dB]')
        stg += stmp('bbb_drive_pattern', self.bbb_drive_pattern, '')
        stg += ftmp('freq_harmonic', self.freq_harmonic, '')
        stg += ftmp('freq_start', self.freq_start, '[kHz]')
        stg += ftmp('freq_stop', self.freq_stop, '[kHz]')
        stg += ftmp('freq_step', self.freq_step, '[kHz]')
        stg += ftmp('excitation_time', self.excitation_time, '[s]')
        return stg


class MeasureSpinDepolScan(ThreadedMeasBaseClass):
    """."""

    def __init__(self, isonline=True):
        super().__init__(
            params=MeasureSpinDepolScanParams(),
            target=self.do_measurement,
            isonline=isonline,
        )
        self.depol_utils = SpinDepolUtils()
        if self.isonline:
            self.create_devices()

    def create_devices(self):
        """."""
        self.devices['bbbv'] = BunchbyBunch(BunchbyBunch.DEVICES.V)
        self.devices['bbbh'] = BunchbyBunch(BunchbyBunch.DEVICES.H)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['rfgen'] = RFGen(props2init=['Frequency-Mon'])
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['gamma_monitors'] = FamGammaMonitors()
        self.devices['blms'] = FamBLMs()
        # TODO: add also CAX beamsizes, etc

    def get_data(self):
        """."""
        data = {}
        bbbv = self.devices['bbbv']
        bbbh = self.devices['bbbh']
        data['timestamp'] = _time.time()
        data['lifetime'] = self.devices['currinfo'].lifetime
        data['stored_current'] = self.devices['currinfo'].current
        data['bbbv_freq_sp'] = bbbv.drive1['FREQ']
        data['bbbv_freq_rb'] = bbbv.drive1.frequency
        data['bbbv_tune'] = bbbv.single_bunch.spec_marker1_tune
        data['bbbh_tune'] = bbbh.single_bunch.spec_marker1_tune
        data['bbbv_sram_mag'] = bbbv.sram.spec_marker1_mag
        data['bbbh_sram_mag'] = bbbh.sram.spec_marker1_mag
        data['bbbv_sram_tune'] = bbbv.sram.spec_marker1_tune
        data['bbbh_sram_tune'] = bbbh.sram.spec_marker1_tune
        data['rf_freq'] = self.devices['rfgen'].frequency
        data['tunex'] = self.devices['tune'].tunex
        data['tuney'] = self.devices['tune'].tuney
        data['gamma_counts'] = self.devices['gamma_monitors'].counts
        data['blm_counts'] = self.devices['blms'].counts
        return data

    def do_measurement(self):
        """."""
        bbbv = self.devices['bbbv']

        freq_initial = bbbv.drive1.frequency
        gain_initial = bbbv.pwr_amp.gain
        patt_intial = bbbv.drive1.mask_pattern

        self.data = {}

        start = self.params.freq_start
        stop = self.params.freq_stop
        step = self.params.freq_step
        freq_span = _np.arange(
            start=start, stop=stop + step, step=step, dtype=float
        )
        harm_freq = self.params.freq_harmonic
        harm_freq *= bbbv.info.revolution_freq_mon / 1e3

        bbbv.pwr_amp.gain = self.params.bbb_amp_gain
        bbbv.drive1.mask_pattern = self.params.bbb_drive_pattern

        try:
            siz = freq_span.size
            for i, freq in enumerate(freq_span):
                if self._stopevt.is_set():
                    print('Stopped by user. Restoring BBB initial state.')
                    break
                bbbv.drive1.frequency = harm_freq + freq
                print(f' {i + 1:03d}/{siz:03d} -> freq. {freq:.3f} kHz')
                _time.sleep(self.params.excitation_time)

                data = self.get_data()
                for k, v in data.items():
                    if k not in self.data:
                        self.data[k] = []
                    self.data[k].append(v)

            print('Scan finished.')
        finally:
            bbbv.drive1.frequency = freq_initial
            bbbv.pwr_amp.gain = gain_initial
            bbbv.drive1.mask_pattern = patt_intial

        for k, v in self.data.items():
            self.data[k] = _np.array(v)


class MeasureTuneScanParams(ParamsBaseClass):
    """."""

    TUNEY_INT = 49.0
    TUNEX_INT = 16.0

    def __init__(self):
        super().__init__()
        self.wait_time = 1 # [s]
        self.tuney_start = 0.22
        self.tuney_stop = 0.35
        self.tuney_step = 0.0075
        self.tuney_tol = 0.0001

        self.tunex_start = 0.16
        self.tunex_stop = 0.16
        self.tunex_step = 0.0
        self.tunex_tol = 0.0001

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format

        stg = ''
        stg += ftmp('wait_time', self.wait_time, '[s]')
        stg += ftmp('tuney_start', self.tuney_start, '')
        stg += ftmp('tuney_stop', self.tuney_stop, '')
        stg += ftmp('tuney_step', self.tuney_step, '')
        stg += ftmp('tunex_start', self.tunex_start, '')
        stg += ftmp('tunex_stop', self.tunex_stop, '')
        stg += ftmp('tunex_step', self.tunex_step, '')
        return stg


class MeasureTuneScan(ThreadedMeasBaseClass):
    """."""

    def __init__(self, isonline=True):
        super().__init__(
            params=MeasureTuneScanParams,
            target=self.do_measurement,
            isonline=isonline,
        )
        if self.isonline:
            self.create_devices()

    def create_devices(self):
        """."""
        self.devices['bbbv'] = BunchbyBunch(BunchbyBunch.DEVICES.V)
        self.devices['bbbh'] = BunchbyBunch(BunchbyBunch.DEVICES.H)
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['rfgen'] = RFGen(props2init=['Frequency-Mon'])
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['tunecorr'] = TuneCorr(TuneCorr.DEVICES.SI)
        self.devices['gamma_monitors'] = FamGammaMonitors()
        self.devices['blms'] = FamBLMs()

    def scan_tunes(self):
        """."""
        tune_dev = self.devices['tune']
        tunecorr_dev = self.devices['tunecorr']
        tunecorr_dev.cmd_update_reference()

        tunesx = _np.arange(
            self.params.tunex_start,
            self.params.tunex_stop + self.params.tunex_step,
            self.params.tunex_step,
            dtype=float,
        )
        tunesx += self.params.TUNEX_INT

        tunesy = _np.arange(
            self.params.tuney_start,
            self.params.tuney_stop + self.params.tuney_step,
            self.params.tuney_step,
            dtype=float,
        )
        tunesy += self.params.TUNEY_INT

        for i, (tunex, tuney) in enumerate(zip(tunesx, tunesy)):
            if self._stopevt.is_set():
                print('Scan stopped by the user. Exiting.')
                # restore initial tunes and other params
                break
            tunex_meas, tuney_meas = tune_dev.tunex, tune_dev.tuney

            cond = _math.isclose(
                tunex_meas, tunex, abs_tol=self.params.tunex_tol
            ) and _math.isclose(
                tuney_meas, tuney, abs_tol=self.params.tuney_tol
            )

            while not cond:
                print('Applying delta tune')
                deltax = tunex_meas - tunex
                deltay = tuney_meas - tuney
                tunecorr_dev.delta_tunex = deltax
                tunecorr_dev.delta_tuney = deltay
                tunecorr_dev.cmd_apply_delta()

                tunex_meas, tuney_meas = tune_dev.tunex, tune_dev.tuney
                cond = _math.isclose(
                    tunex_meas, tunex, abs_tol=self.params.tunex_tol
                ) and _math.isclose(
                    tuney_meas, tuney, abs_tol=self.params.tuney_tol
                )
            else:
                print('Tunes corrected!')
                print(f'\t(nux, nuy) = ({tune_dev.tunex, tune_dev.tuney}')
                # update BbB masks tune param
                # update tune proc freq offset
        else:
            # restore initial tunes and other params
            return



    def get_data(self):
        """."""
        while not (self._stop_evet.is_set() or self._finished.is_set()):
            data = self.data
            bbbv = self.devices['bbbv']
            bbbh = self.devices['bbbh']
            data['timestamp'].append(_time.time())
            data['lifetime'].append(self.devices['currinfo'].lifetime)
            data['stored_current'].append(self.devices['currinfo'].current)
            data['bbbv_tune'].append(bbbv.single_bunch.spec_marker1_tune)
            data['bbbh_tune'].append(bbbh.single_bunch.spec_marker1_tune)
            data['bbbv_sram_mag'].append(bbbv.sram.spec_marker1_mag)
            data['bbbh_sram_mag'].append(bbbh.sram.spec_marker1_mag)
            data['bbbv_sram_tune'].append(bbbv.sram.spec_marker1_tune)
            data['bbbh_sram_tune'].append(bbbh.sram.spec_marker1_tune)
            data['rf_freq'].append(self.devices['rfgen'].frequency)
            data['tunex'].append(self.devices['tune'].tunex)
            data['tuney'].append(self.devices['tune'].tuney)
            data['gamma_counts'].append(self.devices['gamma_monitors'].counts)
            data['blm_counts'].append(self.devices['blms'].counts)
            _time.sleep(self.params.wait_time)
        else:
            print('Acquisitions finished!')

    def do_measurement(self):
        """."""
        self.data = {
            'timestamp': [],
            'lifetime': [],
            'stored_current': [],
            'bbbv_tune': [],
            'bbbh_tune': [],
            'bbbv_sram_mag': [],
            'bbbh_sram_mag': [],
            'bbbv_sram_tune': [],
            'bbbh_sram_tune': [],
            'rf_freq': [],
            'tunex': [],
            'tuney': [],
            'gamma_counts': [],
            'blm_counts': [],
        }
        acquisition_thread = _Thread(target=self.get_data, daemon=True)

        acquisition_thread.start()
        self.scan_tunes()
