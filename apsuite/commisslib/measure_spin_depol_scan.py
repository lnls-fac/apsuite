"""."""

import time as _time
from threading import Thread as _Thread

import mathphys.constants as _const
import numpy as _np
import scipy.signal as _scysig
from siriuspy.devices import (
    BunchbyBunch,
    CurrInfoSI,
    DVFImgProc,
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

    # TODO: move this const to mathphy repo
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
        self.devices['rfgen'] = RFGen(
            props2init=['GeneralFreq-SP', 'GeneralFreq-RB']
        )
        self.devices['tune'] = Tune(Tune.DEVICES.SI)
        self.devices['tunecorr'] = TuneCorr(TuneCorr.DEVICES.SI)
        self.devices['gamma_monitors'] = FamGammaMonitors()
        self.devices['blms'] = FamBLMs()
        self.devices['cax'] = DVFImgProc(DVFImgProc.DEVICES.CAX_DVF2)

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

        data['bbbv_sram_mk1_freq'] = bbbv.sram.spec_marker1_freq
        data['bbbv_sram_mk1_mag'] = bbbv.sram.spec_marker1_mag
        data['bbbv_sram_mk2_freq'] = bbbv.sram.spec_marker2_freq
        data['bbbv_sram_mk2_mag'] = bbbv.sram.spec_marker2_mag
        freqs, peaks = self._get_bbb_peaks('v')
        data['bbbv_sram_peaks'] = peaks
        data['bbbv_sram_peaksfreq'] = freqs

        data['bbbh_sram_mk1_freq'] = bbbh.sram.spec_marker1_freq
        data['bbbh_sram_mk1_mag'] = bbbh.sram.spec_marker1_mag
        data['bbbh_sram_mk2_freq'] = bbbh.sram.spec_marker2_freq
        data['bbbh_sram_mk2_mag'] = bbbh.sram.spec_marker2_mag
        freqs, peaks = self._get_bbb_peaks('h')
        data['bbbh_sram_peaks'] = peaks
        data['bbbh_sram_peaksfreq'] = freqs

        data['bbbv_sram_tune'] = bbbv.sram.spec_marker1_tune
        data['bbbh_sram_tune'] = bbbh.sram.spec_marker1_tune

        data['rf_freq'] = self.devices['rfgen'].frequency
        data['tunex'] = self.devices['tune'].tunex
        data['tuney'] = self.devices['tune'].tuney
        data['tunecorr_delta_tunex'] = self.devices['tunecorr'].delta_tunex
        data['tunecorr_delta_tuney'] = self.devices['tunecorr'].delta_tuney
        data['gamma_counts'] = self.devices['gamma_monitors'].counts
        data['blm_counts'] = self.devices['blms'].counts
        data['cax_sigma1'] = self.devices['cax'].fit_sigma1
        data['cax_sigma2'] = self.devices['cax'].fit_sigma2
        data['cax_angle'] = self.devices['cax'].fit_angle

        return data

    def do_measurement(self):
        """."""
        pass

    def _get_bbb_peaks(self, plane):
        bbb = self.devices['bbb' + plane]
        mag = bbb.sram.spec_mag
        freq = bbb.sram.spec_freq
        idx = (freq > 5).nonzero()[0][0]
        idcs, propts = _scysig.find_peaks(mag[idx:], height=-25.5)
        idcs += idx
        return freq[idcs], mag[idcs]

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
    REV_FREQ_NOM = 578.3178240740742  # [kHz] from BbB system
    SPIN_TUNE_INT = 6  # Integer part of the spin tune

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
        stg += ftmp('tune_start', self.tune_start, '(calculated)')
        stg += ftmp('tune_stop', self.tune_stop, '(calculated)')
        stg += ftmp('tune_step', self.tune_step, '(calculated)')
        stg += ftmp('energy_start', self.energy_start, '[GeV] (calculated)')
        stg += ftmp('energy_stop', self.energy_stop, '[GeV] (calculated)')
        stg += ftmp('energy_step', self.energy_step, '[GeV] (calculated)')

        return stg

    @property
    def freq_span(self):
        """."""
        return _np.arange(
            start=self.freq_start,
            stop=self.freq_stop + self.freq_step,
            step=self.freq_step,
            dtype=float
        )

    @property
    def tune_start(self):
        """."""
        return self.freq_start / self.REV_FREQ_NOM

    @property
    def tune_stop(self):
        """."""
        return self.freq_stop / self.REV_FREQ_NOM

    @property
    def tune_step(self):
        """."""
        return self.freq_step / self.REV_FREQ_NOM

    @property
    def energy_start(self):
        """."""
        return _BaseMeasureSpinDepol.calc_energy_from_spin_tune(
            self.SPIN_TUNE_INT + self.tune_start
        )

    @property
    def energy_stop(self):
        """."""
        return _BaseMeasureSpinDepol.calc_energy_from_spin_tune(
            self.SPIN_TUNE_INT + self.tune_stop
        )

    @property
    def energy_step(self):
        """."""
        return _BaseMeasureSpinDepol.calc_energy_from_spin_tune(
            self.tune_step
        )


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

        freq_span = self.params.freq_span

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
        self.change_tunex_error = 0.0
        self.change_tuney_error = 0.0
        self.bbbv_set0_gain = 0.7
        self.bbbv_set0_phase = -120
        self.bbbv_set1_gain = 0.1
        self.bbbv_set1_phase = 60
        self.bbbh_set0_gain = 0.6
        self.bbbh_set0_phase = 150
        self.bbbh_set1_gain = 0.6
        self.bbbh_set1_phase = 150

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.4f}  {2:s}\n'.format

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
        stg += ftmp('change_tunex_error', self.change_tunex_error, '')
        stg += ftmp('change_tuney_error', self.change_tuney_error, '')
        stg += ftmp('bbbv_set0_gain', self.bbbv_set0_gain, '')
        stg += ftmp('bbbv_set0_phase', self.bbbv_set0_phase, '[deg]')
        stg += ftmp('bbbv_set1_gain', self.bbbv_set1_gain, '')
        stg += ftmp('bbbv_set1_phase', self.bbbv_set1_phase, '[deg]')
        stg += ftmp('bbbh_set0_gain', self.bbbh_set0_gain, '')
        stg += ftmp('bbbh_set0_phase', self.bbbh_set0_phase, '[deg]')
        stg += ftmp('bbbh_set1_gain', self.bbbh_set1_gain, '')
        stg += ftmp('bbbh_set1_phase', self.bbbh_set1_phase, '[deg]')
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
        return nr_points * self.wait_time_change_tune * 2

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
        currinfo = self.devices['currinfo']

        gtunesx = self.params.tunex_grid
        gtunesy = self.params.tuney_grid

        # mechanism to compensate for magnet hysteresis. Not ideal, but
        # should help to keep the tune close to the target value:
        errx = self.params.change_tunex_error
        erry = self.params.change_tuney_error
        for i, (gtx, gty) in enumerate(zip(gtunesx, gtunesy)):  # noqa: B905
            freqx = gtx * bbbh.info.revolution_freq_nom / 1e3
            freqy = gty * bbbv.info.revolution_freq_nom / 1e3

            bbbh.feedback.coeff_set = 1
            bbbv.feedback.coeff_set = 1
            currinfo['BuffRst-Cmd'] = 1
            print(
                f' {i + 1:03d}/{len(gtunesx):03d} -> '
                + f'nux, nuy (freqx, freqy): {gtx:6.4f}, {gty:6.4f} '
                + f'({freqx:6.1f} kHz, {freqy:6.1f} kHz)'
            )
            if self._stopevt.is_set():
                print('Scan stopped by the user. Exiting.')
                break

            dtunex, dtuney = self.change_tune(gtx, gty, errx=errx, erry=erry)

            for _ in range(int(self.params.wait_time_change_tune)):
                if self._stopevt.is_set():
                    break
                _time.sleep(1)

            bbbh.feedback.coeff_set = 0
            bbbv.feedback.coeff_set = 0
            currinfo['BuffRst-Cmd'] = 1

            for _ in range(int(self.params.wait_time_change_tune)):
                if self._stopevt.is_set():
                    break
                _time.sleep(1)

            # Calculate the error in this iteration to try to compensate in
            # next iteration. This is needed to try to mitigate the effect
            # of magnet hysteresis, which can cause the tune to deviate
            # from the target value, specially when changing the tune in
            # large steps.
            exn = (gtx - tune_mon.tunex) / dtunex if dtunex != 0 else 0
            eyn = (gty - tune_mon.tuney) / dtuney if dtuney != 0 else 0
            errx += exn * 0.5
            erry += eyn * 0.5
            # limit the error correction factor to avoid overcompensation:
            errx = min(max(errx, -0.2), 0.2)
            erry = min(max(erry, -0.2), 0.2)
            print(
                f'    error (now): {errx:.2f}, {erry:.2f} '
                + f'({exn:.2f}, {eyn:.2f})'
            )

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

        freqx = tunex * bbbh.info.revolution_freq_nom / 1e3
        freqy = tuney * bbbv.info.revolution_freq_nom / 1e3

        dtunex = tunex - tune_mon.tunex
        dtuney = tuney - tune_mon.tuney
        # compensate error from previous iteration:
        dtunex *= 1 + errx
        dtuney *= 1 + erry

        tune_mon.center_frequencyx = freqx
        tune_mon.center_frequencyy = freqy

        bbbh.drive0.frequency = freqx
        bbbv.drive0.frequency = freqy

        bbbh.coeffs.edit_choose_set = 0
        bbbv.coeffs.edit_choose_set = 0
        bbbh.coeffs.edit_freq = tunex * bbbh.feedback.downsample
        bbbv.coeffs.edit_freq = tuney * bbbv.feedback.downsample
        bbbh.coeffs.edit_gain = self.params.bbbh_set0_gain
        bbbh.coeffs.edit_phase = self.params.bbbh_set0_phase
        bbbv.coeffs.edit_gain = self.params.bbbv_set0_gain
        bbbv.coeffs.edit_phase = self.params.bbbv_set0_phase
        bbbh.coeffs.cmd_edit_apply()
        bbbv.coeffs.cmd_edit_apply()

        bbbh.coeffs.edit_choose_set = 1
        bbbv.coeffs.edit_choose_set = 1
        bbbh.coeffs.edit_freq = tunex * bbbh.feedback.downsample
        bbbv.coeffs.edit_freq = tuney * bbbv.feedback.downsample
        bbbh.coeffs.edit_gain = self.params.bbbh_set1_gain
        bbbh.coeffs.edit_phase = self.params.bbbh_set1_phase
        bbbv.coeffs.edit_gain = self.params.bbbv_set1_gain
        bbbv.coeffs.edit_phase = self.params.bbbv_set1_phase
        bbbh.coeffs.cmd_edit_apply()
        bbbv.coeffs.cmd_edit_apply()

        tunecorr_dev.cmd_update_reference()
        tunecorr_dev.delta_tunex = dtunex
        tunecorr_dev.delta_tuney = dtuney
        tunecorr_dev.cmd_apply_delta()
        return dtunex, dtuney

    def _get_data_thread(self):
        """."""
        while not (self._stopevt.is_set() or self._finished.is_set()):
            data = self.get_data()
            self._update_data_dict(data)
            _time.sleep(self.params.wait_time_acq_data)
        print('Acquisition thread finished.')

        for k, v in self.data.items():
            self.data[k] = _np.array(v)
