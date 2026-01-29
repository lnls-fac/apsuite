"""."""
import time as _time

import numpy as _np
from siriuspy.devices import CurrInfoLinear, EVG, PowerSupply, Screen, Trigger
from siriuspy.magnet.factory import NormalizerFactory as _Normalizer

from ...utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class Params(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.nr_points = 16  # Number of measurement points
        self.nr_repeat = 2  # Number of repetitions per point
        self.quad_curr_min = -4.5  # [A]
        self.quad_curr_max = 3.5  # [A]
        self.wait_quad = 2  # [s]
        self.wait_repeat = 1  # [s]
        self.bo_dip_energy = 0.144  # [GeV]

    def __str__(self):
        """."""
        st = ''
        st += f'nr_points = {self.nr_points:d}\n'
        st += f'nr_repeat = {self.nr_repeat:d}\n'
        st += f'quad_curr_min = {self.quad_curr_min:.2f}\n'
        st += f'quad_curr_max = {self.quad_curr_max:.2f}\n'
        st += f'wait_quad = {self.wait_quad:d}\n'
        st += f'wait_repeat = {self.wait_repeat:d}\n'
        st += f'bo_dip_energy = {self.bo_dip_energy:.3f}\n'
        return st


class MeasTomography(_BaseClass):
    """."""

    def __init__(self, isonline=True):
        """."""
        super().__init__(params=Params(), isonline=isonline)
        if isonline:
            self.devices['qf'] = PowerSupply('LI-01:PS-QF3')
            self.devices['scrn'] = Screen(Screen.DEVICES.LI_5)
            self.devices['li_curr'] = CurrInfoLinear(CurrInfoLinear.DEVICES.LI)
            self.devices['li_trig'] = Trigger('LI-Fam:TI-Scrn')
            self.devices['evg'] = EVG()

            self.normalizer = _Normalizer.create(
                self.devices['qf'].devname.replace('PS', 'MA')
            )

    @property
    def curr_range(self):
        """."""
        cmin = self.params.quad_curr_min
        cmax = self.params.quad_curr_max
        nr_points = self.params.nr_points
        return _np.linspace(cmin, cmax, nr_points)

    @property
    def kl_range(self):
        """."""
        if not hasattr(self, 'normalizer'):
            raise RuntimeError('Normalizer not available in offline mode.')

        return self.normalizer.conv_current_2_strength(
            self.curr_range, strengths_dipole=self.params.bo_dip_energy
        )

    def make_measurement(self):
        """."""
        curr0 = self.devices['qf'].current

        keys = [
            'timestamp',
            'quad_curr_rb',
            'quad_curr_mon',
            'quad_kl_rb',
            'quad_kl_mon',
            'ict1_curr',
            'ict2_curr',
            'image_raw',
            'image_exposure',
            'image_gain',
            'scalex',
            'scaley',
            'trigger_delay_raw',
        ]

        # Creates nested lists fill with None to store measured data
        for key in keys:
            self.data[key] = [
                [None for _ in range(self.params.nr_repeat)]
                for _ in range(self.params.nr_points)
            ]

        doexit = False
        for idx1, curr in enumerate(self.curr_range):
            self._set_current(curr, timeout=self.params.wait_quad)

            print(f'Measuring {idx1 + 1:02d}/{self.params.nr_points:02d}')
            print(f'QF current: {curr:.4f} A')

            for idx2 in range(self.params.nr_repeat):
                doexit = self._pulse_evg()

                if not idx2 and doexit:
                    print('Exiting...')
                    break

                print(f'    repeat {idx2 + 1}/{self.params.nr_repeat}')
                data = self._get_single_data()

                for key in keys:
                    self.data[key][idx1][idx2] = data[key]

                _time.sleep(self.params.wait_repeat)

            print('..finished point\n')

            # Meas loop is not broken during repetitions of same point.
            # It measures all repetitions and then breaks.
            if doexit:
                break

        self._set_current(curr0, timeout=self.params.wait_quad)
        print('Finished measurement!')

    def _set_current(self, value, timeout=10):
        qf = self.devices['qf']
        qf.current = value
        return qf._wait_float(
            'Current-Mon', value, abs_tol=0.01, timeout=timeout
        )

    def _pulse_evg(self):
        while True:
            self.devices['evg'].cmd_turn_on_injection()
            print('Injection pulse triggered.')
            st = '[Y] pulse again  |'
            st += '  [N] continue measurement  |'
            st += '  [E] exit measurement'
            print(st)
            ret = input('Choose an option [Y/N/E]: ').strip().lower()

            if ret in ('e', 'exit'):
                # abort entire scan
                return True

            if ret in ('n', 'no'):
                # continue measurement
                return False

            if ret in ('y', 'yes', ''):
                # pulse again
                continue

            print('Invalid option. Please type Y, N or E.')

    def _get_single_data(self):
        data = dict()
        data['timestamp'] = _time.time()

        qf = self.devices['qf']
        curr = qf.current
        curr_mon = qf.current_mon
        data['quad_curr_rb'] = curr
        data['quad_curr_mon'] = curr_mon
        data['quad_kl_rb'] = qf.conv_current_2_strength(
            curr, strengths_dipole=self.params.bo_dip_energy
        )
        data['quad_kl_mon'] = qf.conv_current_2_strength(
            curr_mon, strengths_dipole=self.params.bo_dip_energy
        )

        data['ict1_curr'] = self.devices['li_curr'].charge_ict1
        data['ict2_curr'] = self.devices['li_curr'].charge_ict2

        data['image_raw'] = self.devices['scrn'].image
        data['image_exposure'] = self.devices['scrn'].cam_exposure
        data['image_gain'] = self.devices['scrn'].cam_gain
        data['scalex'] = self.devices['scrn'].scale_factor_x
        data['scaley'] = self.devices['scrn'].scale_factor_y
        data['trigger_delay_raw'] = self.devices['li_trig'].delay_raw

        return data
