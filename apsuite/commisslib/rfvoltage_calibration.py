"""."""

import time as _time
import numpy as _np
from siriuspy.devices import BunchbyBunch, RFCav, CurrInfoSI
from ..utils import ThreadedMeasBaseClass as _ThreadedMeasBaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class RFCalibrationParams(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.rfcav_timeout = 120  # [s]
        self.rfcav_wait = 5  # [s]
        self.rfvoltage_min = 1.0  # [MV]
        self.rfvoltage_max = 2.0  # [MV]
        self.rfvoltage_nrpoints = 15
        self.conv_physics2hardware = 380/1.72  # [mV/MV]

    def __str__(self):
        """."""
        dtmp = '{0:25s} = {1:9d}\n'.format
        ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        stg = ftmp('rfcav_timeout', self.rfcav_timeout, '[s]')
        stg += ftmp('rfcav_wait', self.rfcav_wait, '[s]')
        stg += ftmp('rfvoltage_min', self.rfvoltage_min, '[MV]')
        stg += ftmp('rfvoltage_max', self.rfvoltage_max, '[MV]')
        stg += dtmp('rfvoltage_nrpoints', self.rfvoltage_nrpoints)
        stg += ftmp(
            'conv_physics2hardware', self.conv_physics2hardware, '[mV/MV]')
        return stg


class RFCalibration(_ThreadedMeasBaseClass):
    """."""

    def __init__(
            self, isonline=True):
        """."""
        super().__init__(target=self._meas_func, isonline=isonline)
        self.params = RFCalibrationParams()
        if self.isonline:
            self.devices['bbbl'] = BunchbyBunch(BunchbyBunch.DEVICES.L)
            self.devices['rfcav'] = RFCav(RFCav.DEVICES.SI)
            self.devices['currinfo'] = CurrInfoSI()

    def calc_rfvoltage_span(self):
        """."""
        prms = self.params
        rfvoltage_span = _np.linspace(
            prms.rfvoltage_min, prms.rfvoltage_max, prms.rfvoltage_nrpoints)
        return rfvoltage_span

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
        props = [
            'timestamp', 'sync_freq', 'sync_tune', 'rf_frequency',
            'stored_current', 'rfvoltage_rb', 'rfamplitude_rb']
        for key in props:
            data[key] = []

        rfcav, bbbl = self.devices['rfcav'], self.devices['bbbl']

        amp0 = rfcav.dev_llrf.voltage
        prms = self.params

        rfamp_span = self.calc_rfvoltage_span()
        rfamp_span *= prms.conv_physics2hardware

        data['rfvoltage_sp'] = rfamp_span/prms.conv_physics2hardware
        data['rfamplitude_sp'] = rfamp_span

        # set first value of voltage
        rfcav.set_voltage(rfamp_span[0], timeout=prms.rfcav_timeout)
        for amp in rfamp_span:
            if self._stopevt.is_set():
                print('Stopping...')
                break

            rfcav.set_voltage(amp, timeout=prms.rfcav_timeout)
            _time.sleep(prms.rfcav_wait)
            sync_freq = bbbl.sram.modal_marker_freq
            sync_tune = bbbl.sram.modal_marker_tune
            rffreq = rfcav.dev_rfgen.frequency
            scurr = self.devices['currinfo'].current
            gap_volt = rfcav.dev_cavmon.gap_voltage
            amp_volt = rfcav.dev_llrf.voltage

            data[props[0]].append(_time.time())
            data[props[1]].append(sync_freq)
            data[props[2]].append(sync_tune)
            data[props[3]].append(rffreq)
            data[props[4]].append(scurr)
            data[props[5]].append(gap_volt)
            data[props[6]].append(amp_volt)
            print(
                f'Amp. {amp:.2f} mV, Volt. {gap_volt/1e6:.3f} MV, '
                f'Sync. Freq. {sync_freq:.3f} kHz')

            # set drive frequency to excite closer to actual sync. freq.
            bbbl.drive0.frequency = data['sync_freq'][-1]

        for key in props:
            data[key] = _np.array(data[key])

        print('Restoring initial RF voltage...')
        rfcav.set_voltage(amp0, timeout=prms.rfcav_timeout)
        print('Finished!')
        self.data = data

    @staticmethod
    def calc_synchrotron_frequency(vgap, E0, U0, frf, alpha, h):
        """."""
        return frf * _np.sqrt(alpha/(2*_np.pi*E0*h)) * (vgap**2 - U0**2)**(1/4)
