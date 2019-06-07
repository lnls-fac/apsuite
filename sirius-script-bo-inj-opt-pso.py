#!/usr/bin/env python-sirius

import time as _time
import numpy as np
from epics import PV as _PV


class PSO:
    def __init__(self, dim=5, p_sum=1, p_orb=1e3):
        self.pv_ioc_prefix_SOFB = 'BO-Glob:AP-SOFB:'
        self.pv_ioc_prefix_PosAng = 'TB-Glob:AP-PosAng:'
        self.pv_orbx = _PV(self.pv_ioc_prefix_SOFB + 'SPassOrbX-Mon')  # um
        self.pv_orby = _PV(self.pv_ioc_prefix_SOFB + 'SPassOrbY-Mon')  # um
        self.pv_sum = _PV(self.pv_ioc_prefix_SOFB + 'SPassSum-Mon')  # counts
        self.pv_dx = _PV(self.pv_ioc_prefix_PosAng + 'DeltaPosX-SP')  # mm
        self.pv_dxl = _PV(self.pv_ioc_prefix_PosAng + 'DeltaAngX-SP')  # mrad
        self.pv_dy = _PV(self.pv_ioc_prefix_PosAng + 'DeltaPosY-SP')  # mm
        self.pv_dyl = _PV(self.pv_ioc_prefix_PosAng + 'DeltaAngY-SP')  # mrad
        self.pv_resetposang = _PV(
            self.pv_ioc_prefix_PosAng + 'SetNewRefKick-Cmd')
        self.pv_kckr = _PV('BO-01D:PM-InjKckr:Kick-SP')
        _time.sleep(2)
        self.pv_resetposang.value = 1

        self.x_lim = 3
        self.y_lim = 3
        self.xl_lim = 1
        self.yl_lim = 3
        self.kckr_lim = 3

        self.p_sum = p_sum
        self.p_orb = p_orb

        self.c_inertia = 0.7984
        self.c_ind = 1.49618
        self.c_coll = self.c_ind

        self.dim = dim
        self.nswarm = int(10 + 2 * np.sqrt(self.dim))

        self.delta = np.zeros(self.nswarm, self.dim)
        self.vlct = np.zeros(self.nswarm, self.dim)
        self.p_best = np.zeros(self.nswarm, self.dim)
        self.g_best = np.zeros(self.nswarm, self.dim)

        self.create_swarm()
        self.reference = np.array([
            self.pv_dx.value,
            self.pv_dxl.value,
            self.pv_dy.value,
            self.pv_dyl.value,
            self.pv_kckr.value,
            ])

    def create_swarm(self):
        lim = [self.x_lim, self.y_lim, self.xl_lim, self.yl_lim, self.kckr_lim]

        for j in range(0, self.nswarm):
            self.delta[j, :] = lim * (1 - 2 * np.random.rand(self.dim))
            self.vlct[j, :] = np.zeros(self.dim)
            self.p_best[j, :] = self.delta[j, :]

    def merit_func(self, eff_lim, part):
        self.pv_dx.value = self.reference[0] + self.delta[part, 0]
        self.pv_dxl.value = self.reference[1] + self.delta[part, 1]
        self.pv_dy.value = self.reference[2] + self.delta[part, 2]
        self.pv_dyl.value = self.reference[3] + self.delta[part, 3]
        self.pv_kckr.value = self.reference[4] + self.delta[part, 4]
        _time.sleep(3)
        ind_bpm = np.arange(1, len(self.pv_sum.value)+1)
        f_bpm = np.dot(ind_bpm, self.pv_sum.value)
        orbx_sel = self.pv_orbx.value[self.pv_sum.value > eff_lim]
        sigma_x = orbx_sel * orbx_sel
        orby_sel = self.pv_orby.value[self.pv_sum.value > eff_lim]
        sigma_y = orby_sel * orby_sel
        f_orb = np.sqrt(np.sum(sigma_x + sigma_y))
        return self.p_sum * f_bpm + self.p_orb / f_orb

    def update_vlct(self, part):
        r_ind = np.random.rand()
        r_coll = np.random.rand()

        v_inertia = self.c_inertia * self.vlct[part, :]
        v_ind = self.c_ind*r_ind*(self.p_best[part, :] - self.delta[part, :])
        v_coll = self.c_coll * r_coll * (self.g_best - self.delta[part, :])
        self.vlct[part, :] = v_inertia + v_ind + v_coll

    def update_delta(self, part):
        self.delta[part, :] += self.vlct[part, :]


if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="PSO script.")

    parser.add_argument(
        '-dim', '--dim', type=int, default=5,
        help='Dimension in Search Space (5).')
    parser.add_argument(
        '-sum', '--sum_lim', type=float, default=1e3,
        help='Minimum BPM Sum Signal to calculate merit function (1 kcount).')
    parser.add_argument(
        '-niter', '--niter', type=int, default=30,
        help='Number of Iteractions (30).')
    parser.add_argument(
        '-p_sum', '--psum', type=int, default=1,
        help='Weigth for Sum Signal')
    parser.add_argument(
        '-p_orb', '--porb', type=int, default=1e3,
        help='Weigth for Orbit Rms')

    args = parser.parse_args()
    pso = PSO(args.dim, args.psum, args.porb)
    f_old = np.zeros(pso.nswarm)
    f_new = np.zeros(pso.nswarm)

    for i in range(0, pso.nswarm):
        f_old[i] = pso.merit_func(args.sum_lim, i)

    pso.g_best = pso.p_best[np.argmax(f_old), :]

    k = 0
    while k < args.niter:
        for i in range(0, pso.nswarm):
            pso.update_vlct(i)
            pso.update_delta(i)
            f_new[i] = pso.merit_func(args.sum_lim, i)
            if f_new[i] > f_old[i]:
                pso.p_best[i, :] = pso.delta[i, :]
                f_old[i] = f_new[i]

        pso.g_best = pso.p_best[np.argmax(f_old), :]
        k += 1
