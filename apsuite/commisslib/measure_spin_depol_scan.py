"""."""

import time as _time

import mathphys.constants as _const
import numpy as _np
from siriuspy.devices import CurrInfoSI, RFGen, Tune
from siriuspy.devices.bbb import BunchbyBunch

from ..utils import ParamsBaseClass, ThreadedMeasBaseClass

A_ELECTRON = 1.15965218059e-3
E_electron = _const.electron_rest_energy / _const.elementary_charge  # [eV]


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
        self.bbb_drive_pattern = '2:1'  # not sure this format works
        self.freq_start = 400.0  # [kHz]
        self.freq_stop = 500.0  # [kHz]
        self.freq_step = 1.0  # [kHz]
        self.time_base_update_time = 1.0  # [s]
        self.excitation_time = 1.0  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format
        # dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format

        stg = ''
        stg += ftmp('bbb_amp_gain', self.bbb_amp_gain, '[dB]')
        stg += stmp('bbb_drive_pattern', self.bbb_drive_pattern, '')
        stg += ftmp('freq_start', self.freq_start, '[kHz]')
        stg += ftmp('freq_stop', self.freq_stop, '[kHz]')
        stg += ftmp('freq_step', self.freq_step, '[kHz]')
        stg += ftmp('time_base_update_time', self.time_base_update_time, '[s]')
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
        if self.isonline:
            self.create_devices()

    def create_devices(self):
        """."""
        self.devices['bbbv'] = BunchbyBunch(BunchbyBunch._devices['V'])
        self.devices['currinfo'] = CurrInfoSI()
        self.devices['rfgen'] = RFGen(props2init=['Frequency-Mon'])
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['gamma_counters'] = None
        self.devices['blm_counters'] = None
        # TODO: add also tune from BBB and SRAM peaks, CAX beamsizes, etc

        self.pvs['bbbv_freq_sp'] = self.devices['bbbv'].drive1.pv_object(
            'SI-Glob:DI-BbBProc-V:DRIVE1_FREQ'
        )
        self.pvs['bbbv_freq_rb'] = self.devices['bbbv'].drive1.pv_object(
            'SI-Glob:DI-BbBProc-V:DRIVE1_FREQ_ACT_STRING'
        )

    def get_data(self):
        """."""
        self.data['timestamps'].append(_time.time())
        self.data['lifetime'].append(self.devices['currinfo'].lifetime)
        self.data['stored_current'].append(self.devices['currinfo'].current)
        self.data['bbbv_freq_sp'].append(self.pvs['bbbv_freq_sp'].value)
        self.data['bbbv_freq_rb'].append(self.pvs['bbbv_freq_rb'].value)
        self.data['rf_freq'].append(self.devices['rfgen'].frequency)
        self.data['tunex'].append(self.devices['tune'].tunex)
        self.data['tuney'].append(self.devices['tune'].tuney)
        self.data['gamma_counters'].append(self.get_gamma_data())
        self.data['blm_counters'].append(self.get_blm_data())

    def get_gamma_data(self):
        """."""
        raise NotImplementedError

    def get_blm_data(self):
        """."""
        raise NotImplementedError

    def do_measurement(self):
        """."""
        self._freq_initial = self.devices['bbbv'].drive1.frequency
        self._gain_initial = self.devices['bbbv'].pwr_amp.gain
        self._patt_intial = self.devices['bbbv'].drive1.mask_pattern

        self.data = {
            'timestamps': [],
            'lifetime': [],
            'stored_current': [],
            'bbbv_freq_sp': [],
            'bbbv_freq_rb': [],
            'rf_freq': [],
            'tunex': [],
            'tuney': [],
            'gamma_counters': [],
            'blm_counters': [],
        }

        start = self.params.freq_start
        stop = self.params.freq_stop
        step = self.params.freq_step
        freq_span = _np.arange(
            start=start, stop=stop + step, step=step, dtype=float
        )

        self.devices['bbbv'].pwr_amp.gain = self.params.bbb_amp_gain
        self.devices[
            'bbbv'
        ].drive1.mask_pattern = self.params.bbb_drive_pattern

        try:
            for i, freq in enumerate(freq_span):
                if self._stopevt.is_set():
                    print('Stopped by user. Restoring BBB initial state.')
                    break
                self.devices['bbbv'].drive1.frequency = freq
                print(
                    f'Driving freq. {freq:.3f} kHz'
                    + f' ({i + 1})/{freq_span.size:03d})'
                )
                _time.sleep(self.params.excitation_time)
                self.get_data()
            print('Scan finished.')
        finally:
            self.devices['bbbv'].drive1.frequency = self._freq_initial
            self.devices['bbbv'].pwr_amp.gain = self._gain_initial
            self.devices['bbbv'].drive1.mask_pattern = self._patt_intial

        for k, v in self.data.items():
            self.data[k] = _np.array(v)


class SpinDepolUtils:
    """."""

    @staticmethod
    def calc_spin_tune_from_energy(energy):
        """Calculate the spin tune and its fractional parts from beam energy.

        Args:
            energy (float): Beam energy in [GeV]

        Returns:
            tuple of floats (spin_tune, spin_tune_frac, 1-spin_tune_frac)
            containing the spin tune value, its fractional part and the
            its complementary fractional part.
        """
        gamma = (energy * 1e9) / E_electron
        spin_tune = A_ELECTRON * gamma
        spin_tune_frac = spin_tune - int(spin_tune)
        return spin_tune, spin_tune_frac, 1 - spin_tune_frac

    @staticmethod
    def calc_energy_from_spin_tune(spin_tune):
        """Calculate the beam energy from the spin tune value.

        Args:
            spin_tune (float): _description_

        Returns:
            float: Beam energy in [GeV]
        """
        gamma = spin_tune / A_ELECTRON
        energy = gamma * E_electron
        return energy / 1e9
