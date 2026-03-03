"""."""

import time as _time
import numpy as _np
import matplotlib.pyplot as _mplt
from threading import Event as _Event, Thread as _Thread

from .. utils import ThreadedMeasBaseClass, ParamsBaseClass
from siriuspy.devices import CurrInfoSI, RFGen, Tune
from siriuspy.devices.bbb import BunchbyBunch

import mathphys.constants as _const

A_ELECTRON = 1.15965218059e-3
E_electron = _const.electron_rest_energy / _const.elementary_charge  # [eV]


class MeasureSpinDepolScanParams(ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.freq_begin = 400  # [kHz]
        self.freq_end = 500  # [kHz]
        self.freq_step = 1  # [kHz]
        self.time_base_update_time = 1  # [s]
        self.excitation_time = 1  # [s]
        self.update_timeout = 3  # [s]

    def __str__(self):
        """."""
        ftmp = '{0:24s} = {1:9.3f}  {2:s}\n'.format
        dtmp = '{0:24s} = {1:9d}  {2:s}\n'.format
        stmp = '{0:24s} = {1:9s}  {2:s}\n'.format

        stg = ''
        stg += ftmp('freq_begin', self.freq_begin, '[kHz]')
        stg += ftmp('freq_end', self.freq_end, '[kHz]')
        stg += ftmp('freq_step', self.freq_step, '[kHz]')
        stg += ftmp('time_base_update_time', self.time_base_update_time, '[s]')
        stg += ftmp('excitation_time', self.excitation_time, '[s]')
        return stg


class MeasureSpinDepolScan(ThreadedMeasBaseClass):
    """."""

    def __init__(self, params=MeasureSpinDepolScanParams(), isonline=True):
        super().__init__(params, isonline)
        self.target = self.do_measurement
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

    def get_data(self):
        """."""
        flag = _Event()

        def update(*args, **kwargs):
            """."""
            flag.set()

        curr_mon = self.devices['currinfo'].pv_object('Current-Mon')
        curr_mon.auto_monitor = True
        curr_mon.add_callback(update)

        freq_sp = self.devices['bbbv'].pv_object(
            'SI-Glob:DI-BbBProc-V:DRIVE0_FREQ'
        )
        freq_rb = self.devices['bbbv'].pv_object(
            'SI-Glob:DI-BbBProc-V:DRIVE0_FREQ_ACT_STRING'
        )
        while not self._stopevt.is_set():

            if flag.wait(self.params.update_timeout):
                self.data['timestamps'].append(_time.time())
                self.data['lifetime'].append(self.devices['currinfo'].lifetime)
                self.data['stored_current'].append(
                    self.devices['currinfo'].current
                )
                self.data['bbbv_frequency_sp'].append(freq_sp.value)
                self.data['bbbv_frequency_rb'].append(freq_rb.value)
                self.data['rf_frequency'].append(
                    self.devices['rfgen'].frequency
                )
                self.data['tunex'].append(self.devices['tune'].tunex)
                self.data['tuney'].append(self.devices['tune'].tuney)
                self.data['gamma_counters'].append(self.get_gamma_data())
                self.data['blm_counters'].append(self.get_blm_data())
                flag.clear()
            else:
                print('Warning: timeout waiting for current update!')
                break

        curr_mon.clear_callbacks(update)
        curr_mon.auto_monitor = False

        for k, v in self.data.items():
            self.data[k] = _np.array(v)

    def get_gamma_data(self,):
        """."""
        raise NotImplementedError

    def get_blm_data(self,):
        """."""
        raise NotImplementedError

    def do_scan(self,):
        """."""

        beg = self.params.freq_begin
        end = self.params.freq_end
        step = self.params.freq_step
        freq_span = _np.linspace(
            start=beg, stop=end, num=int((end - beg) / step) + 1
        )
        for i, freq in enumerate(freq_span):
            if self._stopevt.is_set():
                break
            self.devices['bbbv'].drive0.frequency = freq
            _time.sleep(self.params.excitation_time)
            print(
                f'Driving freq. {freq:.3f} kHz ({i+1})/{freq_span.size:03d})'
            )
        print('Scan finished.')
        self.stop()


    def do_measurement(self):
        """."""
        self.data = {
            'timestamps': [],
            'lifetime': [],
            'stored_current': [],
            'bbbv_frequency_sp': [],
            'bbbv_frequency_rb': [],
            'rf_frequency': [],
            'tunex': [],
            'tuney': [],
            'gamma_counters': [],
            'blm_counters': [],
            }

        acq_thread = _Thread(target=self.get_data, daemon=True)
        scan_thread = _Thread(target=self.do_scan, daemon=True)

        acq_thread.start()
        scan_thread.start()

        scan_thread.join()
        acq_thread.join()


class SpinDepolUtils:
    """".""""
    @staticmethod
    def calc_spin_tune_from_energy(energy):
        gamma = energy / E_electron
        spin_tune = A_ELECTRON * gamma
        spin_tune_frac = spin_tune - int(spin_tune)
        return spin_tune, spin_tune_frac, 1-spin_tune_frac

    @staticmethod
    def calc_energy_from_spin_tune(spin_tune):
        gamma = spin_tune / A_ELECTRON
        energy = gamma * E_electron
        return energy
