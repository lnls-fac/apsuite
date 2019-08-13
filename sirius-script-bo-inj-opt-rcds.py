#!/usr/bin/env python-sirius

import time as _time
import numpy as np
from epics import PV as _PV


class RCDS():
    def __init__(self, dim=5, p_sum=1, p_orb=1e3, sum_lim=None):
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

        self.dim = dim
        self.p_sum = p_sum
        self.p_orb = p_orb
        self.sum_lim = sum_lim
        self.golden = (1 + np.sqrt(5))/2

        self.x_lim = 3
        self.y_lim = 3
        self.xl_lim = 1
        self.yl_lim = 3
        self.kckr_lim = 3
        lim = np.array([self.x_lim, self.y_lim, self.xl_lim,
                        self.yl_lim, self.kckr_lim])
        self.vrange = np.array([-lim, lim]).transpose()
        self.delta = (1 - 2 * np.random.rand(self.dim)) * lim
        self.reference = np.array([
            self.pv_dx.value,
            self.pv_dxl.value,
            self.pv_dy.value,
            self.pv_dyl.value,
            self.pv_kckr.value,
            ])

    def merit_func(self, x):
        self.pv_dx.value = self.reference[0] + self.delta[0]
        self.pv_dxl.value = self.reference[1] + self.delta[1]
        self.pv_dy.value = self.reference[2] + self.delta[2]
        self.pv_dyl.value = self.reference[3] + self.delta[3]
        self.pv_kckr.value = self.reference[4] + self.delta[4]
        _time.sleep(3)
        ind_bpm = np.arange(1, len(self.pv_sum.value) + 1)
        f_bpm = np.dot(ind_bpm, self.pv_sum.value)
        orbx_sel = self.pv_orbx.value[self.pv_sum.value > self.sum_lim]
        sigma_x = orbx_sel * orbx_sel
        orby_sel = self.pv_orby.value[self.pv_sum.value > self.sum_lim]
        sigma_y = orby_sel * orby_sel
        f_orb = np.sqrt(np.sum(sigma_x + sigma_y))
        return -(self.p_sum * f_bpm + self.p_orb / f_orb)

    def golden_search(self, nint, x):
        k = 0
        x_upper = x[1]
        x_lower = x[0]
        d = self.golden * (x_upper - x_lower)
        x1 = x_lower + d
        x2 = x_upper - d
        f1 = self.merit_func(x1)
        f2 = self.merit_func(x2)

        while k < nint:
            if f1 > f2:
                x_lower = x2
                x2 = x1
                f2 = f1
                x1 = x_lower + self.golden * (x_upper - x_lower)
                f1 = self.merit_func(x1)
            elif f2 > f1:
                x_upper = x1
                x1 = x2
                f1 = f2
                x2 = x_upper - self.golden * (x_upper - x_lower)
                f2 = self.merit_func(x2)
            k += 1
        return x_lower, x_upper

if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="RCDS script.")

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
