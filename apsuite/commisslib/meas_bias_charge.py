"""."""
import time as _time

import numpy as np
from epics import PV

from siriuspy.devices import EGBias, ICT, TranspEff, DevLILLRF

from ..utils import MeasBaseClass as _BaseClass, \
    ParamsBaseClass as _ParamsBaseClass


class ParamsBias(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.bias_ini = -41
        self.bias_fin = -55
        self.bias_step = 1
        self.nrpulses = 20
        self.wait_bias = 10

    def __str__(self):
        """."""
        st = '{0:30s}= {1:9.3f}\n'.format('initial bias [V]', self.bias_ini)
        st += '{0:30s}= {1:9.3f}\n'.format('final bias [V]', self.bias_fin)
        st += '{0:30s}= {1:9.3f}\n'.format('step bias [V]', self.bias_step)
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format('Wait Bias [s]', self.wait_bias)
        return st


class MeasCharge(_BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__(ParamsBias())
        self.devices = {
            'bias': EGBias(),
            'ict': ICT(ICT.DEVICES.LI_1),
            'transpeff': TranspEff(TranspEff.DEVICES.LI),
            }
        self.pvs = {
            'energy': PV('LI-Glob:AP-MeasEnergy:Energy-Mon'),
            'spread': PV('LI-Glob:AP-MeasEnergy:Spread-Mon'),
        }
        self.data = {
            'eff': [],
            'charge': [],
            'bias': [],
            'energy': [],
            'spread': [],
            }

    @property
    def bias_span(self):
        """."""
        ini = self.params.bias_ini
        fin = self.params.bias_fin
        dlt = self.params.bias_step
        return self._calc_span(ini, fin, dlt)

    @staticmethod
    def _calc_span(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_meas(self):
        """."""
        var_span = self.bias_span
        self.data['eff'] = []
        self.data['charge'] = []
        self.data['bias'] = []
        self.data['energy'] = []
        self.data['spread'] = []

        print('Setting Initial Value...')
        self.devices['bias'].voltage = var_span[0]
        _time.sleep(self.params.wait_bias)
        print('Starting Loop')
        for val in var_span:
            eff = np.zeros(self.params.nrpulses)
            chrg = np.zeros(self.params.nrpulses)
            bias_val = np.zeros(self.params.nrpulses)
            energy = np.zeros(self.params.nrpulses)
            spread = np.zeros(self.params.nrpulses)
            self.devices['bias'].voltage = val
            _time.sleep(self.params.wait_bias)
            for k in range(self.params.nrpulses):
                eff[k] = self.devices['transpeff'].efficiency
                chrg[k] = self.devices['ict'].charge
                bias_val[k] = self.devices['bias'].voltage
                energy[k] = self.pvs['energy'].value
                spread[k] = self.pvs['spread'].value
                _time.sleep(0.5)
            self.data['eff'].append(eff)
            self.data['charge'].append(chrg)
            self.data['bias'].append(bias_val)
            self.data['energy'].append(energy)
            self.data['spread'].append(spread)
            print(
                ('Bias [V]: {0:8.3f} -> Charge [nC]: {1:8.3f}, ' +
                    'Energy [MeV]: {2:8.3f}, Spread [%]: {3:8.3f}').format(
                        self.devices['bias'].voltage,
                        self.devices['ict'].charge,
                        self.pvs['energy'].value, self.pvs['spread'].value))
        print('Finished!')


class ParamsKly2(_ParamsBaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__()
        self.kly2_ini = 75
        self.kly2_fin = 73
        self.kly2_step = 0.2
        self.nrpulses = 20
        self.wait_kly2 = 10
        self.kly2_timeout = 40

    def __str__(self):
        """."""
        st = '{0:30s}= {1:9.3f}\n'.format(
            'initial klystron2 amp [%]', self.kly2_ini)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'final klystron2 amp [%]', self.kly2_fin)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'step klystron2 amp [V]', self.kly2_step)
        st += '{0:30s}= {1:9d}\n'.format('number of pulses', self.nrpulses)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'wait klystron2 [s]', self.wait_kly2)
        st += '{0:30s}= {1:9.3f}\n'.format(
            'klystron2 timeout [s]', self.kly2_timeout)
        return st


class Kly2Energy(_BaseClass):
    """."""

    def __init__(self):
        """."""
        super().__init__(ParamsKly2())
        self.devices = {
            'kly2': DevLILLRF(DevLILLRF.DEVICES.LI_KLY2),
            'ict': ICT(ICT.DEVICES.LI_1),
            'transpeff': TranspEff(TranspEff.DEVICES.LI),
            }
        self.pvs = {
            'energy': PV('LI-Glob:AP-MeasEnergy:Energy-Mon'),
            'spread': PV('LI-Glob:AP-MeasEnergy:Spread-Mon'),
        }
        self.data = {
            'eff': [],
            'charge': [],
            'kly2amp': [],
            'energy': [],
            'spread': [],
            }

    @property
    def kly2_span(self):
        """."""
        ini = self.params.kly2_ini
        fin = self.params.kly2_fin
        dlt = self.params.kly2_step
        return self._calc_span(ini, fin, dlt)

    @staticmethod
    def _calc_span(ini, fin, dlt):
        npts = abs(int((fin - ini)/dlt)) + 1
        return np.linspace(ini, fin, npts)

    def do_meas(self):
        """."""
        var_span = self.kly2_span
        self.data['eff'] = []
        self.data['charge'] = []
        self.data['kly2amp'] = []
        self.data['energy'] = []
        self.data['spread'] = []

        print('Setting Initial Value...')
        self.devices['kly2'].set_amplitude(
            var_span[0], timeout=self.params.kly2_timeout)
        _time.sleep(self.params.wait_kly2)
        print('Starting Loop')
        for val in var_span:
            eff = np.zeros(self.params.nrpulses)
            chrg = np.zeros(self.params.nrpulses)
            kly2_val = np.zeros(self.params.nrpulses)
            energy = np.zeros(self.params.nrpulses)
            spread = np.zeros(self.params.nrpulses)
            self.devices['kly2'].set_amplitude(
                val, timeout=self.params.kly2_timeout)
            _time.sleep(self.params.wait_kly2)
            for k in range(self.params.nrpulses):
                eff[k] = self.devices['transpeff'].efficiency
                chrg[k] = self.devices['ict'].charge
                kly2_val[k] = self.devices['kly2'].amplitude
                energy[k] = self.pvs['energy'].value
                spread[k] = self.pvs['spread'].value
                _time.sleep(0.5)
            self.data['eff'].append(eff)
            self.data['charge'].append(chrg)
            self.data['kly2amp'].append(kly2_val)
            self.data['energy'].append(energy)
            self.data['spread'].append(spread)
            print(
                ('Klystron2 Amp [%]: {0:8.3f} -> Charge [nC]: {1:8.3f}, ' +
                    'Energy [MeV]: {2:8.3f}, Spread [%]: {3:8.3f}').format(
                        self.devices['kly2'].amplitude,
                        self.devices['ict'].charge,
                        self.pvs['energy'].value, self.pvs['spread'].value))
        print('Finished!')


class Isoenergy:
    """."""

    def __init__(self, qinit=None, qfinal=None):
        """."""
        self.qinit = qinit
        self.qfinal = qfinal
        self.get_params_charge()

    def bias2charge(self, b):
        """."""
        p = np.array(
            [7.90472090e-04, 1.34199022e-01, 7.62099148e+00, 1.44833137e+02])
        return np.polyval(p, b)

    def charge2bias(self, q):
        """."""
        p = np.array(
            [7.90472090e-04, 1.34199022e-01, 7.62099148e+00, 1.44833137e+02])
        newp = np.poly1d(p)
        roots = (newp-q).roots
        return roots[~np.iscomplex(roots)].real[0]

    def charge2energy(self, q):
        """."""
        p = np.array(
            [1.09315839e-03, -3.16996192e-02, -1.72815644e-01, 1.46492452e+02])
        return np.polyval(p, q)

    def charge2spread(self, q):
        """."""
        p = np.array([0.00178367, 0.0177915, 0.05214362, 0.12375059])
        return np.polyval(p, q)

    def kly2en(self, amp, q):
        """."""
        if q > 2.5:
            p = np.array([1.01026423, 71.90322743])  # > 2.5nC
        else:
            p = np.array([0.80518365, 87.56545895])  # < 2.5nC
        return np.polyval(p, amp)

    def en2kly(self, en, q):
        """."""
        if q > 2.5:
            p = np.array([1.01026423, 71.90322743])  # > 2.5nC
        else:
            p = np.array([0.80518365, 87.56545895])  # < 2.5nC
        return (en - p[1])/p[0]

    def calc_deltak2(self, delta_en, q):
        """."""
        if q > 2.5:
            p = np.array([1.01026423, 71.90322743])  # > 2.5nC
        else:
            p = np.array([0.80518365, 87.56545895])  # < 2.5nC
        return delta_en/p[0]

    def calc_deltaen(self, delta_q):
        """."""
        p = np.array(
            [1.09315839e-03, -3.16996192e-02, -1.72815644e-01, 1.46492452e+02])
        return p[0]*delta_q**3 + p[1]*delta_q**2 + p[2]*delta_q

    def get_params_charge(self):
        """."""
        energy_init = self.charge2energy(q=self.qinit)
        energy_final = self.charge2energy(q=self.qfinal)
        denergy = energy_final - energy_init
        bias_init = self.charge2bias(q=self.qinit)
        spread_init = self.charge2spread(q=self.qinit)
        bias_final = self.charge2bias(q=self.qfinal)
        spread_final = self.charge2spread(q=self.qfinal)
        delta_k2 = self.calc_deltak2(-denergy, self.qfinal)
        print(
            'Charge: {0:4f} nC --> {1:4f} nC'.format(
                self.qinit, self.qfinal))
        print(
            'Bias: {0:4f} V --> {1:4f} V'.format(
                bias_init, bias_final))
        print(
            'Energy: {0:4f} MeV --> {1:4f} MeV'.format(
                energy_init, energy_final-denergy))
        print(
            'Spread: {0:4f} % --> {1:4f} %'.format(
                spread_init, spread_final))
        print(
            'Apply Delta Klystron 2 Amp: {0:4f} %'.format(delta_k2))
