import time as _time
from epics import PV as _PV
import numpy as np
from optimization import PSO, SimulAnneal, SimpleScan


class PSOInjection(PSO):

    def initialization(self):
        self._pv_orbx = _PV('BO-Glob:AP-SOFB:SPassOrbX-Mon')  # um
        self._pv_orby = _PV('BO-Glob:AP-SOFB:SPassOrbY-Mon')  # um
        self._pv_sum = _PV('BO-Glob:AP-SOFB:SPassSum-Mon')  # counts
        self._pv_dx = _PV('TB-Glob:AP-PosAng:DeltaPosX-SP')  # mm
        self._pv_dxl = _PV('TB-Glob:AP-PosAng:DeltaAngX-SP')  # mrad
        self._pv_dy = _PV('TB-Glob:AP-PosAng:DeltaPosY-SP')  # mm
        self._pv_dyl = _PV('TB-Glob:AP-PosAng:DeltaAngY-SP')  # mrad
        self._pv_kckr = _PV('BO-01D:PM-InjKckr:Kick-SP')

        self._upper_limits = np.array([3, 3, 3, 3, 1])
        self._lower_limits = - self._upper_limits

        self._p_bpm = 1.0
        self._p_orb = 1.0e3

        self.reference = np.array([
            self._pv_dx.value,
            self._pv_dxl.value,
            self._pv_dy.value,
            self._pv_dyl.value,
            self._pv_kckr.value,
            ])

    def calc_merit_function(self):
        f_out = np.zeros(self._nswarm)
        ind_bpm = np.arange(1, len(self._pv_sum.value) + 1)

        for i in range(0, self._nswarm):
            change = self.reference + self._position[i, :]
            change = self._set_lim(change)
            self._pv_dx.value = change[0]
            self._pv_dxl.value = change[1]
            self._pv_dy.value = change[2]
            self._pv_dyl.value = change[3]
            self._pv_kckr.value = change[4]
            _time.sleep(3)
            f_bpm = np.dot(ind_bpm, self._pv_sum.value)
            bpm_sel = self._pv_sum.value > self._sum_lim
            orbx_sel = self._pv_orbx.value[bpm_sel]
            sigma_x = orbx_sel * orbx_sel
            orby_sel = self._pv_orby.value[bpm_sel]
            sigma_y = orby_sel * orby_sel
            f_orb = np.sqrt(np.sum(sigma_x + sigma_y))
            f_out[i] = self._p_bpm * f_bpm + self._p_orb / f_orb
        return - f_out

if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="PSO script for Booster Injection Optimization.")

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
