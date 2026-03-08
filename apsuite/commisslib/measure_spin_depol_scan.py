"""."""

import time as _time
from threading import Thread as _Thread

import mathphys.constants as _const
import numpy as _np
from siriuspy.devices import (
    BunchbyBunch,
    CurrInfoSI,
    FamBLMs,
    FamGammaMonitors,
    RFGen,
    Tune,
    TuneCorr
)

from ..utils import ParamsBaseClass, ThreadedMeasBaseClass


class _BaseMeasureSpinDepol(ThreadedMeasBaseClass):
    """."""

    _PARAMS_CLASS = ParamsBaseClass

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

    def __init__(self, isonline=True):
        """."""
        super().__init__(
            params=self._PARAMS_CLASS(),
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
        data['tunecorr_delta_tunex'] = self.devices['tunecorr'].delta_tunex
        data['tunecorr_delta_tuney'] = self.devices['tunecorr'].delta_tuney
        data['gamma_counts'] = self.devices['gamma_monitors'].counts
        data['blm_counts'] = self.devices['blms'].counts
        return data

    def do_measurement(self):
        """."""
        pass

    def _update_data_dict(self, data_dict):
        """Update self.data with the values in the provided dictionary.

        Args:
            data_dict (dict): Dictionary containing parameter names and their
                corresponding values.
        """
        for k, v in data_dict.items():
            if k not in self.data:
                self.data[k] = []
            self.data[k].append(v)


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


class MeasureSpinDepolScan(_BaseMeasureSpinDepol):
    """."""

    _PARAMS_CLASS = MeasureSpinDepolScanParams

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
        harm_freq *= bbbv.info.revolution_freq_nom / 1e3

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
                self._update_data_dict(data)

            print('Scan finished.')
        finally:
            bbbv.drive1.frequency = freq_initial
            bbbv.pwr_amp.gain = gain_initial
            bbbv.drive1.mask_pattern = patt_intial

        for k, v in self.data.items():
            self.data[k] = _np.array(v)


class MeasureTuneScanParams(ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.wait_time_change_tune = 40  # [s]
        self.wait_time_acq_data = 1.0  # [s]
        self.nr_steps = 20
        self.tunex_start = 0.16
        self.tunex_stop = 0.16
        self.tuney_start = 0.22
        self.tuney_stop = 0.35

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format

        stg = ''
        stg += ftmp('wait_time_change_tune', self.wait_time_change_tune, '[s]')
        stg += ftmp('wait_time_acq_data', self.wait_time_acq_data, '[s]')
        stg += ftmp('nr_steps', self.nr_steps, '')
        stg += ftmp('tunex_start', self.tunex_start, '')
        stg += ftmp('tunex_stop', self.tunex_stop, '')
        stg += ftmp('tunex_step', self.tunex_step, '(calculated)')
        stg += ftmp('tuney_start', self.tuney_start, '')
        stg += ftmp('tuney_stop', self.tuney_stop, '')
        stg += ftmp('tuney_step', self.tuney_step, '(calculated)')
        stg += f'total scan time (approx.) = {self.total_scan_time:.1f} s\n'
        return stg

    @property
    def tunex_step(self):
        """."""
        return (self.tunex_stop - self.tunex_start) / self.nr_steps

    @property
    def tuney_step(self):
        """."""
        return (self.tuney_stop - self.tuney_start) / self.nr_steps

    @property
    def tunex_grid(self):
        """."""
        return self._calc_grid(
            start=self.tunex_start,
            stop=self.tunex_stop,
            nr_steps=self.nr_steps,
        )

    @property
    def tuney_grid(self):
        """."""
        return self._calc_grid(
            start=self.tuney_start,
            stop=self.tuney_stop,
            nr_steps=self.nr_steps,
        )

    @property
    def total_scan_time(self):
        """."""
        nr_points = 2 * self.nr_steps + 1
        return nr_points * self.wait_time_change_tune

    @staticmethod
    def _calc_grid(start, stop, nr_steps):
        """."""
        grid = _np.linspace(
            start=start, stop=stop, num=nr_steps + 1, dtype=float
        )
        # add the grid in reverse order, excluding the last point:
        grid = _np.r_[grid, grid[-2::-1]]
        return grid


class MeasureTuneScan(_BaseMeasureSpinDepol):
    """."""

    _PARAMS_CLASS = MeasureTuneScanParams

    def do_measurement(self):
        """."""
        self.data = {}

        print('Starting acquisition thread.')
        acquisition_thread = _Thread(target=self._get_data_thread, daemon=True)
        acquisition_thread.start()

        bbbh = self.devices['bbbh']
        bbbv = self.devices['bbbv']
        tune_mon = self.devices['tune']

        gtunesx = self.params.tunex_grid
        gtunesy = self.params.tuney_grid

        # mechanism to compensate for magnet hysteresis. Not ideal, but
        # should help to keep the tune close to the target value:
        errx = erry = 0
        for i, (gtx, gty) in enumerate(zip(gtunesx, gtunesy)):  # noqa: B905
            freqx = gtx * bbbh.info.revolution_freq_nom
            freqy = gty * bbbv.info.revolution_freq_nom

            print(
                f' {i + 1:03d}/{len(gtunesx):03d} -> '
                + f'nux, nuy (freqx, freqy): {gtx:6.4f}, {gty:6.4f} '
                + f'({freqx:6.1f} kHz, {freqy:6.1f} kHz)'
            )
            if self._stopevt.is_set():
                print('Scan stopped by the user. Exiting.')
                break

            dtunex, dtuney = self._change_tune(gtx, gty, errx=errx, erry=erry)

            _time.sleep(self.params.wait_time_change_tune)

            # Calculate the error in this iteration to try to compensate in
            # next iteration. This is needed to try to mitigate the effect
            # of magnet hysteresis, which can cause the tune to deviate
            # from the target value, specially when changing the tune in
            # large steps.
            errx = (gtx - tune_mon.tunex) / dtunex if dtunex != 0 else 0
            erry = (gty - tune_mon.tuney) / dtuney if dtuney != 0 else 0
            print(f'    error: {errx:.2f}, {erry:.2f}')
            # limit the error correction factor to avoid overcompensation:
            errx = min(max(errx, -0.2), 0.2)
            erry = min(max(erry, -0.2), 0.2)

        print('Scan finished! Waiting for acquisition thread to finish.')
        self._stopevt.set()
        acquisition_thread.join()
        print('Done!')

    def change_tune(self, tunex, tuney, errx=0.0, erry=0.0):
        """Change the tune to the specified values.

            An error compensation can be applied, if requested.

        Args:
            tunex (float): Target horizontal tune value.
            tuney (float): Target vertical tune value.
            errx (float, optional): Relative error compensation factor for
                horizontal tune. Defaults to 0.0.
            erry (float, optional): Relative error compensation factor for
                vertical tune. Defaults to 0.0.

        Returns:
            tunex, tuney: A tuple containing the change in horizontal tune
                variation (dtunex) and vertical tune variation (dtuney)
                applied to achieve the target tune values.

        """
        bbbh = self.devices['bbbh']
        bbbv = self.devices['bbbv']
        tune_mon = self.devices['tune']
        tunecorr_dev = self.devices['tunecorr']

        freqx = tunex * bbbh.info.revolution_freq_nom
        freqy = tuney * bbbv.info.revolution_freq_nom

        dtunex = tunex - tune_mon.tunex
        dtuney = tuney - tune_mon.tuney
        # compensate error from previous iteration:
        dtunex *= 1 + errx
        dtuney *= 1 + erry

        tune_mon.center_frequencyx = freqx
        tune_mon.center_frequencyy = freqy

        bbbh.drive0.frequency = freqx
        bbbv.drive0.frequency = freqy

        bbbh.coeffs.edit_freq = tunex * bbbh.feedback.downsample
        bbbv.coeffs.edit_freq = tuney * bbbv.feedback.downsample
        bbbh.coeffs.cmd_edit_apply()
        bbbv.coeffs.cmd_edit_apply()

        tunecorr_dev.cmd_update_reference()
        tunecorr_dev.delta_tunex = dtunex
        tunecorr_dev.delta_tuney = dtuney
        tunecorr_dev.cmd_apply_delta()
        return dtunex, dtuney

    def _get_data_thread(self):
        """."""
        while not (self._stop_evet.is_set() or self._finished.is_set()):
            data = self.get_data()
            self._update_data_dict(data)
            _time.sleep(self.params.wait_time_acq_data)
        print('Acquisition thread finished.')

        for k, v in self.data.items():
            self.data[k] = _np.array(v)
