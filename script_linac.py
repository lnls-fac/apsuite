#!/usr/bin/env python-sirius
"""."""

import time as _time
from epics import PV as _PV
import numpy as np
from optimization import PSO


class PSOLinac(PSO):

    ENERGY = 0
    SPREAD = 1
    ICT1_CHG = 2
    ICT2_CHG = 3

    def __init__(self, low_lim, up_lim):
        self._upper_limits = up_lim
        self._lower_limits = low_lim

    def initialization(self):
        _pv_phase_shb = _PV('LA-RF:LLRF:BUN1:SET_PHASE')
        _pv_phase_kly1 = _PV('LA-RF:LLRF:KLY1:SET_PHASE')
        _pv_phase_kly2 = _PV('LA-RF:LLRF:KLY2:SET_PHASE')

        _pv_amp_shb = _PV('LA-RF:LLRF:BUN1:SET_AMP')
        _pv_amp_kly1 = _PV('LA-RF:LLRF:KLY1:SET_AMP')
        _pv_amp_kly2 = _PV('LA-RF:LLRF:KLY2:SET_AMP')

        self.params = [_pv_phase_shb,
                       _pv_amp_shb,
                       _pv_phase_kly1,
                       _pv_amp_kly1,
                       _pv_phase_kly2,
                       _pv_amp_kly2]

        _pv_energy = _PV('Linac_Energy')  # Fake PV
        _pv_spread = _PV('Linac_Energy_Spread')  # Fake PV
        _pv_ict1 = _PV('ICT1_charge')  # Fake PV
        _pv_ict2 = _PV('ICT2_charge')  # Fake PV

        self.diag = [_pv_energy,
                     _pv_spread,
                     _pv_ict1,
                     _pv_ict2]

        self._wait = 0

        self.p_energy = 1
        self.p_spread = 75
        self.p_transmit = 150

        self._max_delta = self._upper_limits/10
        self._min_delta = - self._max_delta

        self._nswarm = 10 + 2 * int(np.sqrt(len(self._upper_limits)))

    def calc_merit_function(self):
        f_out = np.zeros(self._nswarm)

        for i in range(0, self._nswarm):
            for k in range(0, len(self.params)):
                self.params[k].value = self._position[i, k]

            _time.sleep(self._wait)
            transmit = self.diag[ICT2_CHG].value / self.diag[ICT1_CHG].value
            f_out[i] = self.p_energy * self.diag[ENERGY].value +
                       self.p_spread / self.diag[SPREAD].value +
                       self.p_transmit * transmit
        return - f_out

if __name__ == "__main__":
    import argparse as _argparse

    parser = _argparse.ArgumentParser(
        description="PSO script for LLRF Linac Optimization.")

    parser.add_argument(
        '-niter', '--niter', type=int, default=30,
        help='Number of Iteractions (30).')
    parser.add_argument(
        '-upper_lim', '--up_lim', nargs='+', type=float,
        help='Upper limits [SHB_Phase, SHB_Amp, Kly1_Phase, Kly1_Amp, Kly2_Phase, Kly2_Amp]')
    parser.add_argument(
        '-lower_lim', '--low_lim', nargs='+', type=float,
        help='Lower limits [SHB_Phase, SHB_Amp, Kly1_Phase, Kly1_Amp, Kly2_Phase, Kly2_Amp]')

    args = parser.parse_args()
    Pso = PSOLinac(low_lim=args.low_lim, up_lim=args.up_lim)
    Pso._start_optimization(niter=args.niter)
