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
        self.voltage_min = 1.0  # [MV]
        self.voltage_max = 2.0  # [MV]
        self.voltage_nrpoints = 15
        self.voltage_incrate = self.VoltIncRates.vel_0p5
        self.conv_physics2hardware = 506/1.75  # [mV/MV]

    def __str__(self):
        """."""
        dtmp = '{0:25s} = {1:9d}\n'.format
        ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        stmp = '{0:25s} = {1:9s}  {2:s}\n'.format
        stg = ftmp('voltage_timeout', self.voltage_timeout, '[s]')
        stg += ftmp('voltage_wait', self.voltage_wait, '[s]')
        stg += ftmp('voltage_min', self.voltage_min, '[MV]')
        stg += ftmp('voltage_max', self.voltage_max, '[MV]')
        stg += dtmp('voltage_nrpoints', self.voltage_nrpoints)
        incrate_str = self.VoltIncRates._fields[self.voltage_incrate] \
            if isinstance(self.voltage_incrate, int) else self.voltage_incrate
        stg += stmp(
            'voltage_incrate', incrate_str, '[mV/s]')
        stg += ftmp(
            'conv_physics2hardware', self.conv_physics2hardware, '[mV/MV]')
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
            self.devices['rfcav'] = RFCav(props2init=[])
            self.devices['currinfo'] = CurrInfoSI(props2init=['Current-Mon'])

    def calc_voltage_span(self):
        """."""
        prms = self.params
        voltage_span = _np.linspace(
            prms.voltage_min, prms.voltage_max, prms.voltage_nrpoints)
        return voltage_span

    def configure_bbb_drive_pattern(self, cbmode):
        """."""
        harm_nr = 864
        bunches = _np.arange(harm_nr)
        bbbl = self.devices['bbbl']
        bbbl.drive0.mask = _np.cos(2*_np.pi*bunches*cbmode/harm_nr) > 0
        # drive at synchrotron frequency
        bbbl.drive0.frequency = bbbl.sram.modal_marker_freq

    def _meas_func(self):
        data = dict()
        data_keys = [
            'timestamp', 'sync_freq', 'sync_tune', 'rf_frequency',
            'stored_current', 'voltage_rb', 'amplitude_rb']
        for key in data_keys:
            data[key] = []

        llrf, bbbl = self.devices['rf_cav'].dev_llrf, self.devices['bbbl']

        amp0 = llrf.voltage
        prms = self.params

        rfamp_span = self.calc_voltage_span()
        rfamp_span *= prms.conv_physics2hardware

        data['voltage_sp'] = rfamp_span/prms.conv_physics2hardware
        data['amplitude_sp'] = rfamp_span

        inc_rate0 = llrf.voltage_incrate
        llrf.voltage_incrate = prms.voltage_incrate

        # set first value of voltage
        llrf.set_voltage(rfamp_span[0], timeout=prms.voltage_timeout)
        for amp in rfamp_span:
            if self._stopevt.is_set():
                print('Stopping...')
                break

            llrf.set_voltage(amp, timeout=prms.voltage_timeout)
            _time.sleep(prms.voltage_wait)
            sync_freq = bbbl.sram.modal_marker_freq
            sync_tune = bbbl.sram.modal_marker_tune
            rffreq = self.devices['rfcav'].dev_rfgen.frequency
            scurr = self.devices['currinfo'].current
            gap_volt = self.devices['rf_cav'].dev_cavmon.gap_voltage
            amp_volt = llrf.dev_llrf.voltage

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

            # set drive frequency to excite closer to actual sync. freq.
            bbbl.drive0.frequency = data['sync_freq'][-1]

        for key in data_keys:
            data[key] = _np.array(data[key])

        print('Restoring initial RF voltage...')
        llrf.set_voltage(amp0, timeout=prms.voltage_timeout)

        print('Restoring initial RF voltage increase rate...')
        llrf.voltage_incrate = inc_rate0

        print('Finished!')
        self.data = data

    @staticmethod
    def calc_synchrotron_frequency(vgap, E0, U0, frf, alpha, h):
        """."""
        return frf * _np.sqrt(alpha/(2*_np.pi*E0*h)) * (vgap**2 - U0**2)**(1/4)
