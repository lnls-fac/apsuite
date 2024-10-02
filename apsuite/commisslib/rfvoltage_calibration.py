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
        self.initial_voltage = [0.5, 0.5]  # [MV]
        self.final_voltage = [1.0, 1.0]  # [MV]
        self.voltage_nrpoints = 15
        self.voltage_incrate = self.VoltIncRates.vel_0p5
        self.cbmode2drive = 200
        self.cbmode_drive_freq_offset = -0.08  # [kHz]
        # TODO: UPDATE WITH NEW CONVERSION
        self.conv_physics2hardware = 506/1.75  # [mV/MV]
        self.restore_initial_state = True

    def __str__(self):
        """."""
        dtmp = '{0:25s} = {1:9d}\n'.format
        ftmp = '{0:25s} = {1:9.2f}  {2:s}\n'.format
        stmp = '{0:25s} = {1:9s}  {2:s}\n'.format
        stg = ftmp('voltage_timeout', self.voltage_timeout, '[s]')
        stg += ftmp('voltage_wait', self.voltage_wait, '[s]')
        stg += stmp('initial_voltage', str(self.initial_voltage), '[MV]')
        stg += stmp('final_voltage', str(self.final_voltage), '[MV]')
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
            _dev = RFCav.DEVICES
            names = [_dev.SIA, _dev.SIB]
            self.devices['rfcavs'] = [RFCav(nm, props2init=[]) for nm in names]
            self.devices['currinfo'] = CurrInfoSI(props2init=['Current-Mon'])

    def calc_voltage_span(self):
        """."""
        prms = self.params
        cavs = self.devices['rfcavs']
        nrpts = prms.voltage_nrpoints
        voltage_span = _np.zeros((nrpts, len(cavs)))
        for idx, _ in cavs:
            voltage_span[:, idx] = _np.linspace(
                prms.initial_voltage[idx],
                prms.final_voltage[idx],
                nrpts)
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
        rf_freq = self.devices['rfcavs'][0].dev_rfgen.frequency
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

        llrfs = [cav.dev_llrf for cav in self.devices['rfcavs']]
        bbbl = self.devices['bbbl']

        amp0s = [llrf.voltage for llrf in llrfs]
        prms = self.params

        rfamp_span = self.calc_voltage_span()
        rfamp_span *= prms.conv_physics2hardware

        data['voltage_sp'] = rfamp_span/prms.conv_physics2hardware
        data['amplitude_sp'] = rfamp_span

        inc_rate0s = [llrf.voltage_incrate for llrf in llrfs]
        for llrf in llrfs:
            llrf.voltage_incrate = prms.voltage_incrate
        timeout = prms.voltage_timeout

        # set first value of voltage
        if not self.set_voltage_and_track_tune(rfamp_span[0], timeout=timeout):
            print('Voltage timeout.')

        for amps in rfamp_span:
            if self._stopevt.is_set():
                print('Stopping...')
                break
            if not self.set_voltage_and_track_tune(amps, timeout=timeout):
                print('Voltage timeout!')
            _time.sleep(prms.voltage_wait)
            sync_freq = bbbl.sram.modal_marker_freq
            sync_tune = bbbl.sram.modal_marker_tune
            rffreq = self.devices['rfcavs'][0].dev_rfgen.frequency
            scurr = self.devices['currinfo'].current
            gap_volts = [
                cav.dev_cavmon.gap_voltage for cav in self.devices['rfcavs']
                ]
            amp_volts = [llrf.voltage for llrf in llrfs]
            data['timestamp'].append(_time.time())
            data['sync_freq'].append(sync_freq)
            data['sync_tune'].append(sync_tune)
            data['rf_frequency'].append(rffreq)
            data['stored_current'].append(scurr)
            data['voltage_rb'].append(gap_volts)
            data['amplitude_rb'].append(amp_volts)

            for idx, llrf in enumerate(llrfs):
                name = llrf.system_nickname
                print(f'Cavity {name}')
                print(
                    f'Amp. {amps[idx]:.2f} mV, ' +
                    f'Volt. {gap_volts[idx]/1e6:.3f} MV')
            print(f'Sync. Freq. {sync_freq:.3f} kHz')

            self.data = data

        for key in data_keys:
            data[key] = _np.array(data[key])

        if prms.restore_initial_state:
            print('Restoring initial RF voltage...')
            for amp0, llrf in zip(amp0s, llrfs):
                llrf.set_voltage(amp0, timeout=2*prms.voltage_timeout)

            print('Restoring initial RF voltage increase rate...')
            for inc_rate0, llrf in zip(inc_rate0s, llrfs):
                llrf.voltage_incrate = inc_rate0

        print('Finished!')
        self.data = data

    def set_voltage_and_track_tune(self, voltages, timeout=100):
        """Set cavity volt, change drive and mode freq to match tune."""
        llrfs = [cav.dev_llrf for cav in self.devices['rfcavs']]
        bbbl = self.devices['bbbl']
        success = False
        for _ in range(int(timeout)):
            is_ok = []
            for volt, llrf in zip(voltages, llrfs):
                is_ok.append(llrf.set_voltage(volt, timeout=1))
            if all(is_ok):
                success = True
                break
            self.set_bbb_drive_frequency(
                sync_freq=bbbl.sram.modal_marker_freq)
            if self._stopevt.is_set():
                print('Stopping...')
                break

        self.set_bbb_drive_frequency(
            sync_freq=bbbl.sram.modal_marker_freq)
        return success

    @staticmethod
    def calc_synchrotron_frequency(vgap, E0, U0, frf, alpha, h):
        """."""
        return frf * _np.sqrt(alpha/(2*_np.pi*E0*h)) * (vgap**2 - U0**2)**(1/4)
