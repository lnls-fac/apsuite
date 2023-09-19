"""."""

import time as _time
import numpy as _np
from siriuspy.devices import BunchbyBunch, RFCav, ASLLRF, CurrInfoSI
from ..utils import ThreadedMeasBaseClass as _ThreadedMeasBaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class RFCalibrationParams(_ParamsBaseClass):
    """."""

    VoltIncRates = ASLLRF.VoltIncRates

    def __init__(self):
        """."""
        super().__init__()
        self.voltage_timeout = 120  # [s]
        self.voltage_wait = 5  # [s]
        self.initial_voltage = 1.0  # [MV]
        self.final_voltage = 2.0  # [MV]
        self.voltage_nrpoints = 15
        self.voltage_incrate = self.VoltIncRates.vel_0p5
        self.cbmode2drive = 200
        self.cbmode_drive_freq_offset = -0.08  # [kHz]
        self.conv_physics2hardware = 506/1.75  # [mV/MV]
        self.restore_initial_state = True

    def __str__(self):
        """."""
        dtmp = '{0:25s} = {1:9d}\n'.format
        ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        stmp = '{0:25s} = {1:9s}  {2:s}\n'.format
        stg = ftmp('voltage_timeout', self.voltage_timeout, '[s]')
        stg += ftmp('voltage_wait', self.voltage_wait, '[s]')
        stg += ftmp('initial_voltage', self.initial_voltage, '[MV]')
        stg += ftmp('final_voltage', self.final_voltage, '[MV]')
        stg += dtmp('voltage_nrpoints', self.voltage_nrpoints)
        stg += dtmp('cbmode2drive', self.cbmode2drive)
        stg += ftmp(
            'cbmode_drive_freq_offset', self.cbmode_drive_freq_offset, '[kHz]')
        incrate_str = self.VoltIncRates._fields[self.voltage_incrate] \
            if isinstance(self.voltage_incrate, int) else self.voltage_incrate
        stg += stmp(
            'voltage_incrate', incrate_str, '[mV/s]')
        stg += ftmp(
            'conv_physics2hardware', self.conv_physics2hardware, '[mV/MV]')
        stg += dtmp(
            'restore_initial_state', self.restore_initial_state)
        return stg


class RFCalibration(_ThreadedMeasBaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(target=self._meas_func, isonline=isonline)
        self.params = RFCalibrationParams()
        if self.isonline:
            self.devices['bbbl'] = BunchbyBunch(
                BunchbyBunch.DEVICES.L, props2init=[])
            self.devices['rfcav'] = RFCav(RFCav.DEVICES.SI, props2init=[])
            self.devices['currinfo'] = CurrInfoSI(props2init=['Current-Mon'])

    def calc_voltage_span(self):
        """."""
        prms = self.params
        voltage_span = _np.linspace(
            prms.initial_voltage, prms.final_voltage, prms.voltage_nrpoints)
        return voltage_span

    # def configure_bbb_drive_pattern(self, cbmode):
    #     """."""
    #     harm_nr = 864
    #     bunches = _np.arange(harm_nr)
    #     bbbl = self.devices['bbbl']
    #     bbbl.drive0.mask = _np.cos(2*_np.pi*bunches*cbmode/harm_nr) > 0
    #     # drive at synchrotron frequency
    #     bbbl.drive0.frequency = bbbl.sram.modal_marker_freq

    def set_bbb_drive_frequency(self, sync_freq):
        """."""
        harm_nr = 864
        rf_freq = self.devices['rfcav'].dev_rfgen.frequency
        rev_freq = rf_freq/harm_nr * 1e-3  # [Hz -> kHz]
        drive_freq = self.params.cbmode2drive * rev_freq
        drive_freq += sync_freq
        drive_freq += self.params.cbmode_drive_freq_offset
        self.devices['bbbl'].drive0.frequency = drive_freq
        self.devices['bbbl'].sram.modal_sideband_freq = sync_freq

    def _meas_func(self):
        data = dict()
        data_keys = [
            'timestamp', 'sync_freq', 'sync_tune', 'rf_frequency',
            'stored_current', 'voltage_rb', 'amplitude_rb']
        for key in data_keys:
            data[key] = []

        llrf, bbbl = self.devices['rfcav'].dev_llrf, self.devices['bbbl']

        amp0 = llrf.voltage
        prms = self.params

        rfamp_span = self.calc_voltage_span()
        rfamp_span *= prms.conv_physics2hardware

        data['voltage_sp'] = rfamp_span/prms.conv_physics2hardware
        data['amplitude_sp'] = rfamp_span

        inc_rate0 = llrf.voltage_incrate
        llrf.voltage_incrate = prms.voltage_incrate
        timeout = prms.voltage_timeout

        # set first value of voltage
        if not self.set_voltage_and_track_tune(rfamp_span[0], timeout=timeout):
            print('Voltage timeout.')

        for amp in rfamp_span:
            if self._stopevt.is_set():
                print('Stopping...')
                break
            if not self.set_voltage_and_track_tune(amp, timeout=timeout):
                print('Voltage timeout!')
            _time.sleep(prms.voltage_wait)
            sync_freq = bbbl.sram.modal_marker_freq
            sync_tune = bbbl.sram.modal_marker_tune
            rffreq = self.devices['rfcav'].dev_rfgen.frequency
            scurr = self.devices['currinfo'].current
            gap_volt = self.devices['rfcav'].dev_cavmon.gap_voltage
            amp_volt = llrf.voltage

            data['timestamp'].append(_time.time())
            data['sync_freq'].append(sync_freq)
            data['sync_tune'].append(sync_tune)
            data['rf_frequency'].append(rffreq)
            data['stored_current'].append(scurr)
            data['voltage_rb'].append(gap_volt)
            data['amplitude_rb'].append(amp_volt)
            print(
                f'Amp. {amp:.2f} mV, Volt. {gap_volt/1e6:.3f} MV, '
                f'Sync. Freq. {sync_freq:.3f} kHz')

            self.data = data

        for key in data_keys:
            data[key] = _np.array(data[key])

        if prms.restore_initial_state:
            print('Restoring initial RF voltage...')
            llrf.set_voltage(amp0, timeout=2*prms.voltage_timeout)

            print('Restoring initial RF voltage increase rate...')
            llrf.voltage_incrate = inc_rate0

        print('Finished!')
        self.data = data

    def set_voltage_and_track_tune(self, voltage, timeout=100):
        "Set cavity voltage and change drive and mode frequency to match tune."
        llrf, bbbl = self.devices['rfcav'].dev_llrf, self.devices['bbbl']
        success = False
        for _ in range(int(timeout)):
            if llrf.set_voltage(voltage, timeout=1):
                success = True
                break
            self.set_bbb_drive_frequency(
                sync_freq=bbbl.sram.modal_marker_freq)

        self.set_bbb_drive_frequency(
            sync_freq=bbbl.sram.modal_marker_freq)
        return success

    @staticmethod
    def calc_synchrotron_frequency(vgap, E0, U0, frf, alpha, h):
        """."""
        return frf * _np.sqrt(alpha/(2*_np.pi*E0*h)) * (vgap**2 - U0**2)**(1/4)
