#!/usr/bin/env python-sirius

import time as _time
import numpy as np
from epics import PV as _PV


class PSO:

    def __init__(self, niter=None, nswarn=None):
        self._niter = niter or 30
        self._nswarm = nswarn or
        self._c_inertia = 0.7984
        self._c_ind = 1.49618
        self._c_coll = self.c_ind
        self._upper_limits = np.array()
        self._lower_limits = np.array()
        self._best_particle = np.array()
        self._best_global = np.array()
        self.initialization()
        self._check_initialization()

    @property
    def c_ind(self):
        return self._c_ind

    @c_ind.setter
    def c_ind(self, value):
        self._c_ind = value

    def _create_swarm(self):
        self._best_particle = np.zeros(self._nswarn, self._ndim)
        self._best_global = np.zeros(self._ndim)

        delta = self._limits * (1 - 2*np.random.rand(self._nswarn, self._ndim))
        velocity = np.zeros(self._nswarn, self._ndim)
        for j in range(self._nswarm):
            self.vlct[j, :] = np.zeros(self.dim)
            self.p_best[j, :] = self.delta[j, :]
        return delta

    def initialization(self):
        pass

    def calc_merit_function(self, eff_lim, delta):
        pass

    def _update_delta(self, delta, velocity):
        r_ind = np.random.rand()
        r_coll = np.random.rand()

        newvelocity = self.c_inertia * velocity
        newvelocity += self.c_ind * r_ind * (self._best_particle - delta)
        newvelocity += self.c_coll * r_coll * (self._best_global - delta)

        newdelta = delta + newvelocity
        return newdelta, newvelocity

    def _check_initialization(self):
        self._ndim = len(self._limits)
        self._nswarn = self._nswarn or int(10 + 2 * np.sqrt(self._ndim))

    def _start_optimization(self):
        self._create_swarm()

        f_old = np.zeros(self._nswarm)
        f_new = np.zeros(self._nswarm)
        for i in range(0, self._nswarm):
            f_old[i] = self.calc_merit_function(args.sum_lim, i)

        self.g_best = self.p_best[np.argmax(f_old), :]

        k = 0
        while k < args.niter:
            delta, velocity = self._update_delta(delta, velocity)
            for i in range(0, self.nswarm):
                f_new[i] = self.calc_merit_function(args.sum_lim, delta)
                if f_new[i] > f_old[i]:
                    self.p_best[i, :] = self.delta[i, :]
                    f_old[i] = f_new[i]

            self.g_best = pso.p_best[np.argmax(f_old), :]
            k += 1


class PSOInjection(PSO):

    def initialization(self, niter=None):
        self.pv_ioc_prefix_SOFB = 'BO-Glob:AP-SOFB:'
        self.pv_ioc_prefix_PosAng = 'TB-Glob:AP-PosAng'
        self.pv_orbx = _PV(self.pv_ioc_prefix_SOFB + 'SPassOrbX-Mon')  # um
        self.pv_orby = _PV(self.pv_ioc_prefix_SOFB + 'SPassOrbY-Mon')  # um
        self.pv_sum = _PV(self.pv_ioc_prefix_SOFB + 'SPassSum-Mon')  # counts
        self.pv_dx = _PV(self.pv_ioc_prefix_PosAng + 'DeltaPosX-SP')  # mm
        self.pv_dxl = _PV(self.pv_ioc_prefix_PosAng + 'DeltaAngX-SP')  # mrad
        self.pv_dy = _PV(self.pv_ioc_prefix_PosAng + 'DeltaPosY-SP')  # mm
        self.pv_dyl = _PV(self.pv_ioc_prefix_PosAng + 'DeltaAngY-SP')  # mrad
        self.pv_kckr = _PV('BO-01D:PM-InjKckr:Kick-SP')

        self.x_lim = 3
        self.y_lim = 3
        self.xl_lim = 1
        self.yl_lim = 3
        self.kckr_lim = 3

        self.p_bpm = 1.0
        self.p_orb = 1e-7

        self.c_inertia = 0.7984
        self.c_ind = 1.49618
        self.c_coll = self.c_ind

        self.nswarm = int(10 + 2 * np.sqrt(self.dim))

        self.delta = np.zeros(self.nswarm, self.dim)
        self.vlct = np.zeros(self.nswarm, self.dim)
        self.p_best = np.zeros(self.nswarm, self.dim)
        self.g_best = np.zeros(self.nswarm, self.dim)

    def calc_merit_function(self, eff_lim, part):
        self.pv_dx.value += self.delta[part, 0]
        self.pv_dxl.value += self.delta[part, 1]
        self.pv_dy.value += self.delta[part, 2]
        self.pv_dyl.value += self.delta[part, 3]
        self.pv_kckr.value += self.delta[part, 4]
        _time.sleep(3)
        ind_bpm = np.arange(1, len(self.pv_sum.value)+1)
        f_bpm = np.dot(ind_bpm, self.pv_sum.value)
        orbx_sel = self.pv_orbx.value[self.pv_sum.value > eff_lim]
        sigma_x = orbx_sel * orbx_sel
        orby_sel = self.pv_orby.value[self.pv_sum.value > eff_lim]
        sigma_y = orby_sel * orby_sel
        f_orb = np.sqrt(np.sum(sigma_x + sigma_y))
        return self.p_bpm * f_bpm + self.p_orb / f_orb


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
    pso = PSOInjection(niter=args.niter)
