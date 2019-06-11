import time as _time
from epics import PV as _PV
import numpy as np
from optimization import PSO, SimulAnneal, SimpleScan


class PSOInjection(PSO):

    def initialization(self):
        self.pv_ioc_prefix_SOFB = 'BO-Glob:AP-SOFB:'
        self.pv_ioc_prefix_PosAng = 'TB-Glob:AP-PosAng:'
        self.pv_orbx = _PV(self.pv_ioc_prefix_SOFB + 'SPassOrbX-Mon')  # um
        self.pv_orby = _PV(self.pv_ioc_prefix_SOFB + 'SPassOrbY-Mon')  # um
        self.pv_sum = _PV(self.pv_ioc_prefix_SOFB + 'SPassSum-Mon')  # counts
        self.pv_dx = _PV(self.pv_ioc_prefix_PosAng + 'DeltaPosX-SP')  # mm
        self.pv_dxl = _PV(self.pv_ioc_prefix_PosAng + 'DeltaAngX-SP')  # mrad
        self.pv_dy = _PV(self.pv_ioc_prefix_PosAng + 'DeltaPosY-SP')  # mm
        self.pv_dyl = _PV(self.pv_ioc_prefix_PosAng + 'DeltaAngY-SP')  # mrad
        self.pv_kckr = _PV('BO-01D:PM-InjKckr:Kick-SP')

        self._upper_limits = np.array([3, 3, 3, 3, 1])
        self._lower_limits = - self._upper_limits

        self.p_bpm = 1.0
        self.p_orb = 1.0e3

        self.reference = np.array([
            self.pv_dx.value,
            self.pv_dxl.value,
            self.pv_dy.value,
            self.pv_dyl.value,
            self.pv_kckr.value,
            ])

    def calc_merit_function(self):
        f_out = np.zeros(self._nswarm)
        ind_bpm = np.arange(1, len(self.pv_sum.value) + 1)

        for i in range(0, self._nswarm):
            change = self.reference + self._position[i, :]
            change = self._set_lim(change)
            self.pv_dx.value = change[0]
            self.pv_dxl.value = change[1]
            self.pv_dy.value = change[2]
            self.pv_dyl.value = change[3]
            self.pv_kckr.value = change[4]
            _time.sleep(3)
            f_bpm = np.dot(ind_bpm, self.pv_sum.value)
            bpm_sel = self.pv_sum.value > self._sum_lim
            orbx_sel = self.pv_orbx.value[bpm_sel]
            sigma_x = orbx_sel * orbx_sel
            orby_sel = self.pv_orby.value[bpm_sel]
            sigma_y = orby_sel * orby_sel
            f_orb = np.sqrt(np.sum(sigma_x + sigma_y))
            f_out[i] = self.p_bpm * f_bpm + self.p_orb / f_orb
        return - f_out


class Test1(PSO):
    def initialization(self):
        self._upper_limits = np.array([10, 10])
        self._lower_limits = - self._upper_limits
        self._max_delta = self._upper_limits
        self._min_delta = self._lower_limits
        self._nswarm = 10 + 2

    def calc_merit_function(self):
        f_out = np.zeros(self._nswarm)

        for i in range(0, self._nswarm):
            v = self._position[i, :]
            v = self._set_lim(v)
            x = v[0]
            y = v[1]
            f_out[i] = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
            # f_out[i] = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return f_out


class Test2(SimulAnneal):
    def initialization(self):
        self._upper_limits = np.array([10, 10])
        self._lower_limits = - self._upper_limits
        self._max_delta = self._upper_limits/2
        self._min_delta = self._lower_limits/2
        self._position = np.array([0, 0])
        self._temperature = 0

    def calc_merit_function(self):
        v = self._position
        v = self._set_lim(v)
        x = v[0]
        y = v[1]
        f_out = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
        # f_out[i] = 0.26 * (x**2 + y**2) - 0.48 * x * y
        return f_out


class Test3(SimpleScan):
    def initialization(self):
        self._upper_limits = np.array([10, 10])
        self._lower_limits = - self._upper_limits
        self._position = np.array([0, 0])

    def calc_merit_function(self):
        f_scan = np.zeros(len(self._delta))
        for j in range(0, len(self._delta)):
            v = self._position
            v[self._curr_dim] = self._delta[j]
            x = v[0]
            y = v[1]
            # f_scan[j] = (x - 1) ** 2
            # f_scan[j] = (x + 2 * y - 7) ** 2 + (2 * x + y - 5) ** 2
            # f_scan[j] = 0.26 * (x**2 + y**2) - 0.48 * x * y
            print('Position ' + str(v) + 'Fig. Merit' + str(f_scan[j]))
        f_out = np.min(f_scan)
        v[self._curr_dim] = self._delta[np.argmin(f_scan)]
        print('Best Position ' + str(v) + 'Fig. Merit' + str(f_out))
        return f_out, v[self._curr_dim]


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="PSO script.")

    parser.add_argument(
        '-sum', '--sum_lim', type=float, default=1e3,
        help='Minimum BPM Sum Signal to calculate merit function (1 kcount).')
    parser.add_argument(
        '-niter', '--niter', type=int, default=30,
        help='Number of Iteractions (30).')

    args = parser.parse_args()
    Pso = PSOInjection()
    Pso._sum_lim = args.sum_lim
    Pso._start_optimization(niter=args.niter)
